import os
import lightning as L
import torch
import time
from snac import SNAC
from litgpt import Tokenizer
from litgpt.utils import (
    num_parameters,
)
from litgpt.generate.base import (
    generate_AA,
    generate_ASR,
    generate_TA,
    generate_TT,
    generate_AT,
    generate_TA_BATCH,
    next_token_batch
)
import soundfile as sf
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import whisper
from tqdm import tqdm
from huggingface_hub import snapshot_download


torch.set_printoptions(sci_mode=False)


# 词表大小定义
text_vocabsize = 151936  # 文本词表大小,包含所有可能的文本token
text_specialtokens = 64  # 文本特殊标记数量
audio_vocabsize = 4096   # 音频词表大小,用于音频的离散表示
audio_specialtokens = 64 # 音频特殊标记数量

# 填充后的总词表大小
padded_text_vocabsize = text_vocabsize + text_specialtokens   # 文本词表+特殊标记的总大小
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens # 音频词表+特殊标记的总大小

# 文本相关的特殊标记
_eot = text_vocabsize      # End of Text,文本结束标记
_pad_t = text_vocabsize + 1  # Text Padding,文本填充标记
_input_t = text_vocabsize + 2  # Text Input,文本输入标记
_answer_t = text_vocabsize + 3  # Text Answer,文本回答标记
_asr = text_vocabsize + 4  # ASR专用标记,用于语音识别任务

# 音频相关的特殊标记
_eoa = audio_vocabsize      # End of Audio,音频结束标记
_pad_a = audio_vocabsize + 1  # Audio Padding,音频填充标记
_input_a = audio_vocabsize + 2  # Audio Input,音频输入标记
_answer_a = audio_vocabsize + 3  # Audio Answer,音频回答标记
_split = audio_vocabsize + 4  # 音频分隔标记,用于分隔不同的音频片段


def get_input_ids_TA(text, text_tokenizer):
    input_ids_item = [[] for _ in range(8)]
    text_tokens = text_tokenizer.encode(text)
    for i in range(7):
        input_ids_item[i] = [layershift(_pad_a, i)] * (len(text_tokens) + 2) + [
            layershift(_answer_a, i)
        ]
        input_ids_item[i] = torch.tensor(input_ids_item[i]).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens.tolist() + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    return input_ids_item


def get_input_ids_TT(text, text_tokenizer):
    """为文本到文本(T1T2)任务准备模型输入
    
    由于模型使用统一的多模态架构,即使是纯文本任务也需要构建8层结构:
    - 前7层是音频层(但实际是填充)
    - 第8层是实际的文本输入层
    
    参数:
        text (str): 输入文本
        text_tokenizer: 用于将文本转换为token的分词器
    
    返回:
        list[tensor]: 8个tensor组成的列表,每个tensor shape为(1, seq_len)
    """
    # 创建8层结构(7层音频+1层文本)
    input_ids_item = [[] for i in range(8)]
    
    # 将输入文本转换为token序列
    text_tokens = text_tokenizer.encode(text).tolist()

    # 构建前7层(音频层),全部使用填充token
    # 每一层的token都要经过layershift处理以区分不同层
    for i in range(7):
        # 长度为 len(text_tokens) + 3,是为了匹配最后文本层的长度
        # +3 对应文本层的 [_input_t], [_eot], [_answer_t] 三个特殊标记
        input_ids_item[i] = torch.tensor(
            [layershift(_pad_a, i)] * (len(text_tokens) + 3)
        ).unsqueeze(0)  # 增加batch维度变成(1, seq_len)
    
    # 构建第8层(文本层),格式为:
    # [输入标记] + [文本token序列] + [结束标记] + [回答标记]
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)  # 增加batch维度

    return input_ids_item


def get_input_ids_whisper(
    mel, leng, whispermodel, device, 
    special_token_a=_answer_a, special_token_t=_answer_t,
):
    """使用Whisper模型处理音频输入并准备模型输入格式
    
    该函数完成两个主要任务:
    1. 使用Whisper模型从mel频谱图中提取音频特征
    2. 构建8层的输入ID结构(7层音频+1层文本)
    
    参数:
        mel: 输入的mel频谱图
        leng: 音频长度
        whispermodel: Whisper模型实例
        device: 计算设备(CPU/GPU)
        special_token_a: 音频特殊标记,默认为_answer_a
        special_token_t: 文本特殊标记,默认为_answer_t
    
    返回:
        tuple: (audio_feature, input_ids)
            - audio_feature: Whisper提取的音频特征, shape为(1, T, dim)
            - input_ids: 8个tensor的列表,每个tensor shape为(1, seq_len)
    """
    # 使用Whisper模型提取音频特征
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)  # 增加batch维度并移至目标设备
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]  # 提取特征并截取指定长度

    # 获取序列长度T(用于构建填充序列)
    T = audio_feature.size(0)
    
    # 构建8层输入ID结构
    input_ids = []
    
    # 构建前7层(音频层)
    # 每层格式: [输入标记] + [填充标记]*T + [结束标记] + [特殊标记]
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))  # 音频输入标记
        input_ids_item += [layershift(_pad_a, i)] * T   # 填充标记
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]  # 结束标记和特殊标记
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    
    # 构建第8层(文本层)
    # 格式: [输入标记] + [填充标记]*T + [结束标记] + [特殊标记]
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    
    # 返回音频特征和处理后的输入ID
    return audio_feature.unsqueeze(0), input_ids


def get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device):
    """批处理模式下的Whisper输入准备函数
    
    该函数为批处理模式准备音频特征和输入ID。它同时构建AA(音频到音频)和AT(音频到文本)
    两种任务的输入,每种任务包含7层音频和1层文本的输入ID。
    
    参数:
        mel: 梅尔频谱图输入
        leng: 音频长度
        whispermodel: Whisper模型实例,用于提取音频特征
        device: 计算设备(CPU/GPU)
    
    返回:
        tuple: (音频特征, 输入ID列表)
            - 音频特征: shape为(2, T, dim)的tensor,包含两份相同的特征
            - 输入ID列表: 8个tensor组成的列表,每个tensor包含两个批次的输入
    
    工作流程:
    1. 使用Whisper模型提取音频特征
    2. 构建AA任务的输入ID(7层音频+1层文本)
    3. 构建AT任务的输入ID(7层音频+1层文本)
    4. 将两个任务的输入堆叠为批处理格式
    """
    # 使用Whisper模型提取音频特征
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # 提取音频特征并截取指定长度
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
        
    # 获取序列长度
    T = audio_feature.size(0)
    
    # 构建AA(音频到音频)任务的输入ID
    input_ids_AA = []
    for i in range(7):
        input_ids_item = []
        # 添加输入标记(经过layershift处理)
        input_ids_item.append(layershift(_input_a, i))
        # 添加T个填充标记
        input_ids_item += [layershift(_pad_a, i)] * T
        # 添加结束标记和回答标记
        input_ids_item += [(layershift(_eoa, i)), layershift(_answer_a, i)]
        input_ids_AA.append(torch.tensor(input_ids_item))
    # 添加文本层的输入ID
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AA.append(input_id_T)

    # 构建AT(音频到文本)任务的输入ID
    input_ids_AT = []
    for i in range(7):
        input_ids_item = []
        # 添加输入标记(经过layershift处理)
        input_ids_item.append(layershift(_input_a, i))
        # 添加T个填充标记
        input_ids_item += [layershift(_pad_a, i)] * T
        # 添加结束标记和填充标记
        input_ids_item += [(layershift(_eoa, i)), layershift(_pad_a, i)]
        input_ids_AT.append(torch.tensor(input_ids_item))
    # 添加文本层的输入ID
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AT.append(input_id_T)

    # 将AA和AT任务的输入组合
    input_ids = [input_ids_AA, input_ids_AT]
    
    # 重新组织为批处理格式:每层包含两个批次的输入
    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):  # 遍历两个任务
        for j in range(8):  # 遍历8层输入
            stacked_inputids[j].append(input_ids[i][j])
    # 将每层的输入堆叠为tensor
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]
    
    # 返回两份相同的音频特征和堆叠后的输入ID
    return torch.stack([audio_feature, audio_feature]), stacked_inputids


def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1


def A1_A2_batch(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                snacmodel, out_dir=None):
    """批处理模式下的音频到音频转换函数(Audio to Audio Batch)
    
    该函数实现了批处理模式下的音频到音频转换功能。它同时处理两个任务:
    1. A1A2: 音频到音频的转换
    2. A1T2: 音频到文本的转换
    通过批处理方式同时执行这两个任务,提高了处理效率。
    
    参数:
        fabric: PyTorch Lightning的fabric实例,用于模型计算管理
        audio_feature: Whisper提取的音频特征, shape为(2, T, dim)
        input_ids: 8层输入ID列表,每层包含两个批次的输入
        leng: 音频序列长度
        model: 预训练的转换模型
        text_tokenizer: 文本分词器,用于将token转换回文本
        step: 当前处理的步骤编号
        snacmodel: SNAC声码器模型,用于将token解码为波形
        out_dir: 输出目录路径,默认为None时使用默认路径
    
    返回:
        str: 生成的文本描述
    
    工作流程:
    1. 初始化模型的KV缓存(batch_size=2)
    2. 使用generate_TA_BATCH生成token序列
    3. 处理生成的文本token
    4. 重构音频token为SNAC格式
    5. 使用SNAC模型解码为波形
    6. 保存音频文件
    """
    # 初始化模型的KV缓存,设置batch_size=2用于并行处理两个任务
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=2)
        
    # 调用generate_TA_BATCH函数生成token序列
    # 参数说明:
    # - model: 预训练模型
    # - audio_feature: 音频特征(包含两个批次)
    # - input_ids: 分层输入ID(每层包含两个批次)
    # - [leng, leng]: 两个批次的音频长度
    # - ["A1A2", "A1T2"]: 两个任务的标识
    tokenlist = generate_TA_BATCH(
        model,
        audio_feature,
        input_ids,
        [leng, leng],
        ["A1A2", "A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    
    # 获取并处理文本token序列
    text_tokenlist = tokenlist[-1]
    if text_vocabsize in text_tokenlist:
        text_tokenlist = text_tokenlist[: text_tokenlist.index(text_vocabsize)]
    # 将token解码为文本
    text = text_tokenizer.decode(torch.tensor(text_tokenlist)).strip()

    # 重构音频token为SNAC格式并生成音频
    audio_tokenlist = tokenlist[:-1]
    audiolist = reconscruct_snac(audio_tokenlist)
    audio = reconstruct_tensors(audiolist)
    
    # 设置输出目录
    if out_dir is None:
        out_dir = "./output/default/A1-A2-batch"
    else:
        out_dir = out_dir + "/A1-A2-batch"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 使用SNAC模型解码生成音频波形
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
        
    # 保存生成的音频文件
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,  # 采样率24kHz
    )
    
    # 清理KV缓存并返回文本描述
    model.clear_kv_cache()
    return text


def A1_T2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    """音频到文本的转换函数(Audio to Text)
    
    该函数将输入的音频转换为对应的文本响应。这是一个对话式转换,
    不同于ASR的直接转录,它会根据音频内容生成相应的回答。
    
    参数:
        fabric: PyTorch Lightning的fabric实例,用于模型计算管理
        audio_feature: Whisper提取的音频特征, shape为(1, T, dim)
        input_ids: 8层输入ID列表,包含7层音频和1层文本
        leng: 音频序列长度
        model: 预训练的转换模型
        text_tokenizer: 文本分词器,用于将token转换回文本
        step: 当前处理的步骤编号
    
    返回:
        str: 生成的文本响应
    
    工作流程:
    1. 初始化模型的KV缓存
    2. 调用generate_AT进行音频到文本的生成
    3. 将生成的token解码为文本
    """
    # 初始化模型的KV缓存,设置batch_size=1
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    
    # 调用generate_AT函数生成文本token序列
    # generate_AT的参数说明:
    # - model: 预训练模型
    # - audio_feature: 音频特征
    # - input_ids: 分层输入ID
    # - [leng]: 音频长度列表
    # - ["AT"]: 任务类型标识
    # - max_returned_tokens: 最大返回token数
    # - temperature: 采样温度,控制生成的随机性
    # - top_k: 只保留概率最高的k个token
    # - eos_id_a/t: 音频/文本的结束标记
    # - pad_id_t: 文本的填充标记
    # - shift: token的偏移量
    # - include_prompt: 是否包含提示在输出中
    # - generate_text: 是否生成文本
    tokenlist = generate_AT(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["AT"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    
    # 将生成的token序列解码为文本,并去除首尾空白
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_A2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    """音频到音频的转换函数(Audio to Audio)
    
    该函数实现了音频到音频的转换功能,将输入音频转换为新的音频。
    它使用预训练模型生成音频token序列,然后通过SNAC模型解码为波形。
    
    参数:
        fabric: PyTorch Lightning的fabric实例,用于模型计算管理
        audio_feature: Whisper提取的音频特征, shape为(1, T, dim)
        input_ids: 8层输入ID列表,包含7层音频和1层文本
        leng: 音频序列长度
        model: 预训练的转换模型
        text_tokenizer: 文本分词器,用于将token转换回文本
        step: 当前处理的步骤编号
        snacmodel: SNAC声码器模型,用于将token解码为波形
        out_dir: 输出目录路径,默认为None时使用默认路径
    
    返回:
        str: 生成的文本描述
    
    工作流程:
    1. 初始化模型的KV缓存
    2. 使用generate_AA生成音频token序列
    3. 重构音频token为SNAC格式
    4. 使用SNAC模型解码为波形
    5. 保存音频文件
    """
    # 初始化模型的KV缓存,设置batch_size=1
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
        
    # 调用generate_AA函数生成token序列
    tokenlist = generate_AA(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    
    # 重构音频token为SNAC格式
    audiolist = reconscruct_snac(tokenlist)
    # 获取文本token序列
    tokenlist = tokenlist[-1]
    # 移除词表大小之后的token
    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
        
    # 设置输出目录
    if out_dir is None:
        out_dir = "./output/default/A1-A2"
    else:
        out_dir = out_dir + "/A1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 重构音频tensor并使用SNAC模型解码
    audio = reconstruct_tensors(audiolist)
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
        
    # 保存生成的音频文件
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    
    # 清理KV缓存并返回文本描述
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_T1(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    """音频到文本的转录函数(Audio Speech Recognition)
    
    该函数实现了音频到文本的直接转录功能。不同于A1_T2的对话式转换,
    这个函数专注于准确转录音频中的语音内容,类似传统的ASR功能。
    
    参数:
        fabric: PyTorch Lightning的fabric实例,用于模型计算管理
        audio_feature: Whisper提取的音频特征, shape为(1, T, dim)
        input_ids: 8层输入ID列表,包含7层音频和1层文本
        leng: 音频序列长度
        model: 预训练的转换模型
        text_tokenizer: 文本分词器,用于将token转换回文本
        step: 当前处理的步骤编号
    
    返回:
        str: 转录的文本内容
    
    工作流程:
    1. 初始化模型的KV缓存
    2. 调用generate_ASR进行音频转录
    3. 清理缓存并返回转录文本
    """
    # 初始化模型的KV缓存,设置batch_size=1
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
        
    # 调用generate_ASR函数进行音频转录
    # 参数说明:
    # - model: 预训练模型
    # - audio_feature: 音频特征
    # - input_ids: 分层输入ID
    # - [leng]: 音频长度列表
    # - ["A1T1"]: ASR任务标识
    # - max_returned_tokens: 最大返回token数
    # - temperature: 采样温度,控制生成的随机性
    # - top_k: 只保留概率最高的k个token
    # - eos_id_a/t: 音频/文本的结束标记
    # - pad_id_t: 文本的填充标记
    # - shift: token的偏移量
    # - include_prompt: 是否包含提示在输出中
    # - generate_text: 是否生成文本
    tokenlist = generate_ASR(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T1"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    
    # 清理KV缓存,将token序列解码为文本并返回
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_A2(fabric, input_ids, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    """文本到音频的转换函数(Text to Audio)
    
    该函数实现了文本到音频的转换功能,将输入文本转换为对应的音频。
    它首先使用预训练模型生成音频token序列,然后通过SNAC模型解码为波形。
    
    参数:
        fabric: PyTorch Lightning的fabric实例,用于模型计算管理
        input_ids: 8层输入ID列表,包含7层音频和1层文本
        model: 预训练的转换模型
        text_tokenizer: 文本分词器,用于将token转换回文本
        step: 当前处理的步骤编号
        snacmodel: SNAC声码器模型,用于将token解码为波形
        out_dir: 输出目录路径,默认为None时使用默认路径
    
    返回:
        str: 生成的文本描述
    
    工作流程:
    1. 初始化模型的KV缓存
    2. 使用generate_TA生成音频token序列
    3. 重构音频token为SNAC格式
    4. 使用SNAC模型解码为波形
    5. 保存音频文件
    """
    # 初始化模型的KV缓存,设置batch_size=1
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
        
    # 调用generate_TA函数生成token序列
    # 注意这里audio_features为None,因为是文本到音频的转换
    tokenlist = generate_TA(
        model,
        None,
        input_ids,
        None,
        ["T1A2"],  # 使用T1A2任务标识
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )

    # 重构音频token为SNAC格式
    audiolist = reconscruct_snac(tokenlist)
    # 获取文本token序列
    tokenlist = tokenlist[-1]

    # 移除词表大小之后的token
    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
        
    # 重构音频tensor
    audio = reconstruct_tensors(audiolist)
    
    # 设置输出目录
    if out_dir is None:
        out_dir = "./output/default/T1-A2"
    else:
        out_dir = out_dir + "/T1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 使用SNAC模型解码生成音频波形
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
        
    # 保存生成的音频文件
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,  # 采样率24kHz
    )
    
    # 清理KV缓存并返回文本描述
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_T2(fabric, input_ids, model, text_tokenizer, step):

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_TT(
        model,
        None,
        input_ids,
        None,
        ["T1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

    
def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("small").to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False

    with fabric.init_module(empty_init=False):
        model = GPT(config)

    model = fabric.setup(model)
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel

    
def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

    
class OmniInference:

    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)

    def warm_up(self, sample='./data/samples/output1.wav'):
        for _ in self.run_AT_batch_stream(sample):
            pass

    @torch.inference_mode()
    def run_AT_batch_stream(self, 
                            audio_path, 
                            stream_stride=4,
                            max_returned_tokens=2048, 
                            temperature=0.9, 
                            top_k=1, 
                            top_p=1.0,
                            eos_id_a=_eoa,
                            eos_id_t=_eot,
        ):
        """批处理模式下的音频到文本流式生成函数
        
        该函数实现了音频到文本的流式生成功能,使用批处理模式同时处理多个序列,
        并支持流式输出音频数据。这种方式可以实现实时的音频生成和文本转换。
        
        参数:
            audio_path: 输入音频文件的路径
            stream_stride: 流式生成的步长,控制每次生成多少个token
            max_returned_tokens: 最大返回的token数量
            temperature: 采样温度,控制生成的随机性
            top_k: 只保留概率最高的k个token
            top_p: 累积概率阈值,用于nucleus sampling
            eos_id_a: 音频序列的结束标记ID
            eos_id_t: 文本序列的结束标记ID
        
        生成过程:
        1. 加载并处理输入音频
        2. 初始化批处理模式的KV缓存
        3. 生成初始token
        4. 循环生成后续token
        5. 定期输出音频流
        
        返回:
            generator: 生成器对象,用于流式输出音频数据
        """
        # 检查音频文件是否存在
        assert os.path.exists(audio_path), f"audio file {audio_path} not found"
        model = self.model

        # 初始化模型的KV缓存,设置batch_size=2
        with self.fabric.init_tensor():
            model.set_kv_cache(batch_size=2,device=self.device)

        # 加载音频文件并准备输入数据
        mel, leng = load_audio(audio_path)
        audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, self.whispermodel, self.device)
        T = input_ids[0].size(1)  # 获取序列长度
        device = input_ids[0].device

        # 检查最大token数是否合适
        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than audio length {T}"
        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        # 初始化位置编码和输出列表
        input_pos = torch.tensor([T], device=device)
        list_output = [[] for i in range(8)]  # 8个列表用于存储不同层的输出
        
        # 生成初始token
        tokens_A, token_T = next_token_batch(
            model,
            audio_feature.to(torch.float32).to(model.device),
            input_ids,
            [T - 3, T - 3],  # 两个批次的位置偏移
            ["A1T2", "A1T2"],  # 两个相同的任务标识
            input_pos=torch.arange(0, T, device=device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # 保存初始token到输出列表
        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        # 准备下一步的模型输入
        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            # 对音频token进行位置编码处理
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * padded_audio_vocabsize
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=device))
            model_input_ids[i] = torch.stack(model_input_ids[i])

        # 处理文本层的输入
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        # 初始化生成控制变量
        text_end = False  # 文本生成是否结束
        index = 1  # 当前生成的位置
        nums_generate = stream_stride  # 每次流式生成的token数
        begin_generate = False  # 是否开始生成
        current_index = 0  # 当前流式生成的索引

        # 主循环:生成token序列
        for _ in tqdm(range(2, max_returned_tokens - T + 1)):
            # 生成下一个token
            tokens_A, token_T = next_token_batch(
                model,
                None,
                model_input_ids,
                None,
                None,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # 如果文本生成结束,使用填充token
            if text_end:
                token_T = torch.tensor([_pad_t], device=device)

            # 检查是否到达音频序列结束
            if tokens_A[-1] == eos_id_a:
                break

            # 检查是否到达文本序列结束
            if token_T == eos_id_t:
                text_end = True

            # 保存生成的token
            for i in range(7):
                list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])

            # 准备下一步的模型输入
            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() +padded_text_vocabsize + i * padded_audio_vocabsize
                model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
                model_input_ids[i].append(
                    torch.tensor([layershift(4097, i)], device=device)
                )
                model_input_ids[i] = torch.stack(model_input_ids[i])

            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            # 控制流式生成的开始
            if index == 7:
                begin_generate = True

            # 流式生成音频数据
            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    # 获取SNAC格式的音频数据
                    snac = get_snac(list_output, index, nums_generate)
                    # 生成音频流并输出
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)
                    yield audio_stream

            # 更新位置编码和索引
            input_pos = input_pos.add_(1)
            index += 1
            
        # 生成完成后输出文本结果
        text = self.text_tokenizer.decode(torch.tensor(list_output[-1]))
        print(f"text output: {text}")
        # 清理KV缓存
        model.clear_kv_cache()
        return list_output


def test_infer():
    # device = "cuda:0"
    device = "cpu"
    out_dir = f"./output/{get_time_str()}"
    ckpt_dir = f"./checkpoint"
    if not os.path.exists(ckpt_dir):
        print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

    # task = ['A1A2', 'asr', "T1A2", "AA-BATCH", 'T1T2', 'AT']
    task = ["AA-BATCH"]

    # prepare test data
    # TODO
    test_audio_list = sorted(os.listdir('./data/samples'))
    test_audio_list = [(os.path.join('./data/samples', path)).replace('\\', '/') for path in test_audio_list]
    test_audio_transcripts = [
        "What is your name?",
        "what are your hobbies?",
        "Do you like beijing",
        "How are you feeling today?",
        "what is the weather like today?",
    ]
    test_text_list = [
        "What is your name?",
        "How are you feeling today?",
        "Can you describe your surroundings?",
        "What did you do yesterday?",
        "What is your favorite book and why?",
        "How do you make a cup of tea?",
        "What is the weather like today?",
        "Can you explain the concept of time?",
        "Can you tell me a joke?",
    ]

    # LOAD MODEL
    with torch.no_grad():
        if "A1A2" in task:
            print("===============================================================")
            print("                       testing A1A2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                try:
                    mel, leng = load_audio(path)
                    audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device)
                    text = A1_A2(
                        fabric,
                        audio_feature,
                        input_ids,
                        leng,
                        model,
                        text_tokenizer,
                        step,
                        snacmodel,
                        out_dir=out_dir,
                    )
                    print(f"input: {test_audio_transcripts[step]}")
                    print(f"output: {text}")
                    step += 1
                    print(
                        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                    )
                except:
                    print(f"[error] failed to process {path}")
            print("===============================================================")

        if 'asr' in task:
            print("===============================================================")
            print("                       testing asr")
            print("===============================================================")

            index = 0
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_pad_a, special_token_t=_asr)
                output = A1_T1(fabric, audio_feature, input_ids ,leng, model, text_tokenizer, index).lower().replace(',','').replace('.','').replace('?','')
                print(f"audio_path: {path}")
                print(f"audio transcript: {test_audio_transcripts[index]}")
                print(f"asr output: {output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                index += 1

        if "T1A2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1A2")
            print("===============================================================")
            for text in test_text_list:
                input_ids = get_input_ids_TA(text, text_tokenizer)
                text_output = T1_A2(fabric, input_ids, model, text_tokenizer, step,
                                    snacmodel, out_dir=out_dir)
                print(f"input: {text}")
                print(f"output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                step += 1
            print("===============================================================")

        if "T1T2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1T2")
            print("===============================================================")

            for text in test_text_list:
                input_ids = get_input_ids_TT(text, text_tokenizer)
                text_output = T1_T2(fabric, input_ids, model, text_tokenizer, step)
                print(f" Input: {text}")
                print(f"Output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AT" in task:
            print("===============================================================")
            print("                       testing A1T2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(
                    mel, leng, whispermodel, device, 
                    special_token_a=_pad_a, special_token_t=_answer_t
                )
                text = A1_T2(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step
                )
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AA-BATCH" in task:
            print("===============================================================")
            print("                       testing A1A2-BATCH")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device)
                text = A1_A2_batch(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                    snacmodel, out_dir=out_dir
                )
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        print("*********************** test end *****************************")



if __name__ == "__main__":
    test_infer()
