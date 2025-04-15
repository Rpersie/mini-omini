# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Any, Literal, Optional

import torch
# import torch._dynamo.config
# import torch._inductor.config

from litgpt.model import GPT
from utils.snac_utils import layershift, snac_config
from tqdm import tqdm


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """top-p(nucleus)采样的具体实现
    
    只保留累积概率超过top_p的最小token集合,将其他token的概率设为-inf。
    
    参数:
        logits: 原始logits
        top_p: 累积概率阈值(0.0~1.0)
    
    返回:
        torch.Tensor: 过滤后的logits
    
    实现步骤:
    1. 对logits按概率从小到大排序
    2. 计算softmax概率的累积和
    3. 找出累积概率<=1-top_p的token
    4. 将这些token的概率设为-inf
    """
    # 按概率从小到大排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    
    # 找出需要过滤的token
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    
    # 保证至少保留一个token(概率最大的)
    sorted_indices_to_remove[-1:] = 0
    
    # 将过滤结果映射回原始顺序
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    
    # 将被过滤token的概率设为-inf
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """token采样函数
    
    该函数实现了多种采样策略来从logits中生成下一个token。
    支持greedy、temperature和top-p采样,可以组合使用。
    
    参数:
        logits: 模型输出的logits,shape为(batch_size, seq_len, vocab_size)
        temperature: 控制采样随机性的温度参数
                    - temperature=0.0: greedy采样
                    - temperature>0.0: 使用temperature对logits进行缩放
        top_k: 只保留概率最高的k个候选,其他设为-inf
        top_p: 累积概率阈值采样(nucleus sampling)
               - 只保留累积概率超过top_p的最小token集合
               - top_p=1.0表示保留所有token
    
    返回:
        torch.Tensor: 采样得到的token id
    
    采样流程:
    1. 取最后一个位置的logits
    2. 可选的top-k过滤
    3. 可选的temperature缩放
    4. 可选的top-p过滤
    5. 根据最终的概率分布采样
    """
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    
    # 只使用最后一个位置的logits
    logits = logits[0, -1]
    
    # top-k过滤:只保留概率最高的k个候选
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    
    # temperature采样和top-p采样
    if temperature > 0.0 or top_p > 0.0:
        # temperature缩放
        if temperature > 0.0:
            logits = logits / temperature
            
        # top-p(nucleus)采样
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
            
        # 计算softmax概率并采样
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
        
    # 如果temperature=0且top_p=0,使用greedy采样
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(
    model: GPT, input_pos: torch.Tensor, x: list, **kwargs: Any
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    logits_a, logit_t = model(x, input_pos)

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=x[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=x[0].dtype)
    return next_audio_tokens, next_t


def next_token_asr(
    model: GPT,
    input_pos: torch.Tensor,
    audio_features: torch.tensor,
    lens: int,
    input_ids: list,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(audio_features, input_ids, input_pos, whisper_lens=lens)

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_audio_tokens, next_t


def next_token_A1T2(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_audio_tokens, next_t


def next_token_A1T1(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """生成下一个文本token的函数(用于音频到文本任务)
    
    该函数接收当前状态的输入,使用模型生成下一个文本token。
    它主要用于ASR和AT等音频到文本的任务。
    
    参数:
        model: GPT模型实例
        audio_features: Whisper提取的音频特征,在自回归阶段可以为None
        input_ids: 8层输入ID列表
        whisper_lens: 音频特征的长度列表,在自回归阶段可以为None
        task: 任务类型标识列表
        input_pos: 当前位置的tensor
        **kwargs: 其他参数,包括temperature、top_k等采样参数
    
    返回:
        torch.Tensor: 生成的下一个文本token
    
    工作流程:
    1. 将输入移动到正确的设备
    2. 通过模型前向传播获取logits
    3. 对文本logits进行采样得到下一个token
    """
    # 将输入移动到模型所在设备
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    
    # 模型前向传播,获取音频和文本的logits
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )
    
    # 对文本logits进行采样,生成下一个token
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_t


def next_token_batch(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """批处理模式下的token生成函数
    
    该函数实现了批处理模式下的token生成功能,可以同时处理多个序列。
    主要用于音频到文本的转换任务,支持同时生成7层音频token和1层文本token。
    
    参数:
        model: GPT模型实例
        audio_features: Whisper提取的音频特征,在自回归阶段可以为None
        input_ids: 8层输入ID列表,每层包含两个批次的输入
        whisper_lens: 音频特征的长度列表,在自回归阶段可以为None
        task: 任务类型标识列表,如["A1T2", "A1T2"]
        input_pos: 当前位置的tensor
        **kwargs: 其他参数,包括temperature、top_k等采样参数
    
    返回:
        tuple: (音频token列表, 文本token)
        - 音频token列表包含7个tensor,对应7层音频
        - 文本token为单个tensor
    """
    # 将输入移动到模型所在设备
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    
    # 模型前向传播,获取音频和文本的logits
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )

    # 处理音频logits的维度,保留第一个批次的结果
    for i in range(7):
        logits_a[i] = logits_a[i][0].unsqueeze(0)
    # 处理文本logits的维度,取第二个批次的结果
    logit_t = logit_t[1].unsqueeze(0)

    # 对每层音频logits进行采样,生成下一个token
    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    # 对文本logits进行采样,生成下一个token
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    
    return next_audio_tokens, next_t


# torch._dynamo.config.automatic_dynamic_shapes = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.coordinate_descent_tuning = True
# next_token = torch.compile(next_token, mode="reduce-overhead")


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: list,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    # print("eos_id_a:", eos_id_a)
    # print("eos_id_t:", eos_id_t)
    # print("pad_id:", pad_id)
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """
    T = input_ids[0].size(0)
    device = input_ids[0].device
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    for input_id in input_ids:
        input_id = [input_id]
    (
        tokens_A1,
        tokens_A2,
        tokens_A3,
        tokens_A4,
        tokens_A5,
        tokens_A6,
        tokens_A7,
        tokens_T,
    ) = input_ids

    tokens_A1_output = [tokens_A1]
    tokens_A2_output = [tokens_A2]
    tokens_A3_output = [tokens_A3]
    tokens_A4_output = [tokens_A4]
    tokens_A5_output = [tokens_A5]
    tokens_A6_output = [tokens_A6]
    tokens_A7_output = [tokens_A7]
    tokens_T_output = [tokens_T]

    list_output = [
        tokens_A1_output,
        tokens_A2_output,
        tokens_A3_output,
        tokens_A4_output,
        tokens_A5_output,
        tokens_A6_output,
        tokens_A7_output,
        tokens_T_output,
    ]

    input_pos = torch.tensor([T], device=device)
    model_input_ids = [
        tokens_A1.view(1, -1),
        tokens_A2.view(1, -1),
        tokens_A3.view(1, -1),
        tokens_A4.view(1, -1),
        tokens_A5.view(1, -1),
        tokens_A6.view(1, -1),
        tokens_A7.view(1, -1),
        tokens_T.view(1, -1),
    ]

    tokens_A, token_T = next_token(
        model,
        torch.arange(0, T, device=device),
        model_input_ids,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    for i in range(7):
        list_output[i].append(tokens_A[i].clone())
    list_output[7].append(token_T.clone())

    # prepare the input for the next iteration
    for i in range(7):
        tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
    token_T = token_T.clone()

    text_end = False
    max_returned_tokens = 1000
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        model_input_ids = [
            token_a.view(1, -1).to(torch.int32) for token_a in tokens_A
        ] + [token_T.view(1, -1).to(torch.int32)]
        tokens_A, token_T = next_token(
            model,
            input_pos,
            model_input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if text_end:
            token_T = torch.tensor([pad_id], device=device)

        for i in range(7):
            list_output[i].append(tokens_A[i].clone())
        list_output[7].append(token_T.clone())

        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            if generate_text:
                break
            text_end = True

        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
        token_T = token_T.clone()
        input_pos = input_pos.add_(1)

    for i in range(len(list_output)):
        list_output[i] = torch.cat(list_output[i])
    return list_output


@torch.inference_mode()
def generate_TA_BATCH(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 1000,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    """批处理模式下的文本到音频生成函数
    
    该函数实现了批处理模式下的文本到音频转换功能。它可以同时处理多个序列,
    支持并行生成7层音频token和1层文本token,提高了生成效率。
    
    参数:
        model: GPT模型实例
        audio_features: Whisper提取的音频特征,shape为(batch_size, T, dim)
        input_ids: 8层输入ID列表,每层包含batch_size个序列
        leng: 音频长度列表
        task: 任务类型标识列表
        max_returned_tokens: 最大生成token数,默认1000
        temperature: 采样温度,控制随机性,默认1.0
        top_k: 只保留概率最高的k个候选,默认None
        top_p: 累积概率阈值采样,默认1.0
        eos_id_a: 音频结束标记
        eos_id_t: 文本结束标记
        pad_id_t: 文本填充标记
        shift: token偏移量
        include_prompt: 是否包含提示在输出中
        generate_text: 是否生成文本
    
    返回:
        list[list]: 8个列表组成的列表,包含7层音频token和1层文本token
    
    工作流程:
    1. 初始化模型状态和输出列表
    2. 使用next_token_batch生成第一组token
    3. 进入自回归生成循环:
       - 准备模型输入(7层音频+1层文本)
       - 生成下一组token
       - 处理文本结束标记
       - 检查音频结束条件
       - 更新输出和位置编码
    """
    # 获取输入序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    # 初始化位置编码和输出列表
    input_pos = torch.tensor([T], device=device)
    model_input_ids = input_ids
    list_output = [[] for i in range(8)]  # 8个列表用于存储不同层的输出

    # 生成第一组token
    tokens_A, token_T = next_token_batch(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3, T - 3],  # 两个批次的位置偏移
        ["A1T2", "A1T2"],  # 两个相同的任务标识
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # 保存第一组token到输出列表
    for i in range(7):
        list_output[i].append(tokens_A[i].tolist()[0])
    list_output[7].append(token_T.tolist()[0])

    # 准备下一步的模型输入
    model_input_ids = [[] for i in range(8)]
    for i in range(7):
        # 对音频token进行位置编码处理
        tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
        model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
        model_input_ids[i].append(torch.tensor([layershift(snac_config.end_of_audio, i)], device=device))
        model_input_ids[i] = torch.stack(model_input_ids[i])

    # 处理文本层的输入
    model_input_ids[-1].append(token_T.clone().to(torch.int32))
    model_input_ids[-1].append(token_T.clone().to(torch.int32))
    model_input_ids[-1] = torch.stack(model_input_ids[-1])

    # 标记文本是否结束
    text_end = False

    # 主生成循环
    for _ in range(2, max_returned_tokens - T + 1):
        # 生成下一组token
        tokens_A, token_T = next_token_batch(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # 如果文本已结束,使用padding
        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        # 检查是否达到结束条件
        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            text_end = True

        # 保存生成的token
        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        # 准备下一步的输入
        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            # 对音频token进行位置编码处理
            tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(
                torch.tensor([layershift(snac_config.end_of_audio, i)], device=device)
            )
            model_input_ids[i] = torch.stack(model_input_ids[i])

        # 处理文本层的输入
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        # 更新位置编码
        input_pos = input_pos.add_(1)

    return list_output


@torch.inference_mode()
def generate_TT(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    """文本到文本的生成函数(Text to Text)
    
    该函数实现了文本到文本的生成功能。它只生成文本层的token,
    而音频层使用end_of_audio标记填充。这种设计使得模型可以专注于
    文本生成任务,同时保持与其他生成函数的接口一致性。
    
    参数:
        model: GPT模型实例
        audio_features: 此处为None,因为是文本到文本转换
        input_ids: 8层输入ID列表(7层音频+1层文本)
        leng: 此处为None,因为是文本输入
        task: 任务类型标识
        max_returned_tokens: 最大生成token数,默认2048
        temperature: 采样温度,控制随机性,默认1.0
        top_k: 只保留概率最高的k个候选,默认None
        top_p: 累积概率阈值采样,默认1.0
        eos_id_a: 音频结束标记(此处未使用)
        eos_id_t: 文本结束标记
        pad_id_t: 文本填充标记
        shift: token偏移量(此处未使用)
        include_prompt: 是否包含提示在输出中
        generate_text: 是否生成文本
    
    返回:
        list: 生成的文本token序列
    
    工作流程:
    1. 获取输入序列长度和设备信息
    2. 使用next_token_A1T1生成第一个token
    3. 进入自回归生成循环:
       - 为音频层填充end_of_audio标记
       - 生成下一个文本token
       - 检查是否遇到结束标记
       - 更新位置编码
    """
    # 获取输入序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device

    # 初始化输出列表
    output = []
    
    # 生成第一个token
    token_T = next_token_A1T1(
        model,
        None,  # 文本到文本转换不需要音频特征
        input_ids,
        None,  # 不需要音频长度
        None,  # 自回归阶段不需要任务标识
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # 保存第一个token
    output.append(token_T.clone().tolist()[0])
    
    # 初始化位置编码
    input_pos = torch.tensor([T], device=device)

    # 自回归生成循环
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        # 准备模型输入:为每一层构建输入ID
        model_input_ids = []
        for i in range(7):
            # 对每一层添加音频结束标记
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        # 添加上一步生成的文本token
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        
        # 生成下一个token
        token_T = next_token_A1T1(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 检查是否遇到结束标记
        if token_T == eos_id_t:
            break
            
        # 保存生成的token并更新位置编码
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
        
    return output


@torch.inference_mode()
def generate_AT(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    """音频到文本的自回归生成函数
    
    该函数实现了音频到文本的自回归生成过程。它首先处理音频输入生成初始token,
    然后逐步生成后续的文本token,直到遇到结束标记或达到最大长度。

    参数:
        model: GPT模型实例
        audio_features: Whisper提取的音频特征, shape为(1, T, dim)
        input_ids: 8层输入ID列表
        leng: 音频序列长度列表
        task: 任务类型标识列表(如["AT"])
        max_returned_tokens: 最大生成token数,默认2048
        temperature: 采样温度,控制随机性,默认1.0
        top_k: 只保留概率最高的k个候选,默认None
        top_p: 累积概率阈值采样,默认1.0
        eos_id_a: 音频结束标记
        eos_id_t: 文本结束标记
        pad_id_t: 文本填充标记
        shift: token偏移量
        include_prompt: 是否包含提示在输出中
        generate_text: 是否生成文本

    返回:
        torch.Tensor: 生成的token序列
    
    工作流程:
    1. 获取输入序列长度T和设备信息
    2. 使用next_token_A1T1生成第一个token
    3. 进入自回归生成循环:
       - 准备模型输入
       - 生成下一个token
       - 检查是否遇到结束标记
       - 更新位置编码
    """
    # 获取输入序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device

    # 初始化输出列表并生成第一个token
    output = []
    token_T = next_token_A1T1(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],  # 减3是因为特殊标记占用的位置
        ["AT"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    output.append(token_T.clone().tolist()[0])
    
    # 初始化位置编码
    input_pos = torch.tensor([T], device=device)
    
    # 自回归生成循环
    text_end = False
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        # 准备模型输入:为每一层构建输入ID
        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        
        # 生成下一个token
        token_T = next_token_A1T1(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 检查是否遇到结束标记
        if token_T == eos_id_t:
            break
            
        # 保存生成的token并更新位置编码
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
        
    return output


@torch.inference_mode()
def generate_TA(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    """文本到音频的生成函数(Text to Audio)
    
    该函数实现了文本到音频的生成功能,将输入文本转换为对应的音频token序列。
    它使用自回归方式同时生成7层音频token和1层文本token。
    
    参数:
        model: GPT模型实例
        audio_features: 此处为None,因为是文本到音频的转换
        input_ids: 8层输入ID列表(7层音频+1层文本)
        leng: 此处为None,因为是文本输入
        task: 任务类型标识
        max_returned_tokens: 最大生成token数,默认2048
        temperature: 采样温度,控制随机性,默认1.0
        top_k: 只保留概率最高的k个候选,默认None
        top_p: 累积概率阈值采样,默认1.0
        eos_id_a: 音频结束标记
        eos_id_t: 文本结束标记
        pad_id_t: 文本填充标记
        shift: token偏移量
        include_prompt: 是否包含提示在输出中
        generate_text: 是否生成文本
    
    返回:
        list[list]: 8个列表组成的列表,包含7层音频token和1层文本token
    
    工作流程:
    1. 获取输入序列长度和设备信息
    2. 使用next_token_A1T2生成第一组token
    3. 进入自回归生成循环:
       - 准备模型输入(7层音频+1层文本)
       - 生成下一组token
       - 处理文本结束标记
       - 检查音频结束条件
       - 更新输出和位置编码
    """
    # 获取输入序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device

    # 初始化8层输出列表(7层音频+1层文本)
    output = [[] for _ in range(8)]
    
    # 生成第一组token
    tokens_A, token_T = next_token_A1T2(
        model,
        None,  # 文本到音频转换不需要音频特征
        input_ids,
        None,  # 不需要音频长度
        None,  # 自回归阶段不需要任务标识
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # 保存第一组token到输出列表
    for i in range(7):
        output[i].append(tokens_A[i].clone().tolist()[0])
    output[7].append(token_T.clone().tolist()[0])

    # 初始化位置编码
    input_pos = torch.tensor([T], device=device)
    
    # 标记文本是否结束
    text_end = False
    
    # 自回归生成循环
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        # 准备下一步的输入
        model_input_ids = []
        for i in range(7):
            # 对每一层的音频token进行layershift处理
            model_input_ids.append(
                layershift(tokens_A[i].clone(), i)
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        # 添加文本层的输入
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))

        # 生成下一组token
        tokens_A, token_T = next_token_A1T2(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # 如果文本已结束,使用padding
        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        # 检查音频是否结束
        if tokens_A[-1] == eos_id_a:
            break

        # 检查文本是否结束
        if token_T == eos_id_t:
            text_end = True

        # 保存生成的token
        for i in range(7):
            output[i].append(tokens_A[i].clone().tolist()[0])
        output[7].append(token_T.clone().tolist()[0])
        
        # 更新位置编码
        input_pos = input_pos.add_(1)

    return output


@torch.inference_mode()
def generate_AA(
    model: GPT,
    audio_features: torch.Tensor,  # 音频特征输入
    input_ids: list,              # 8层输入ID列表(7层音频+1层文本)
    leng,                         # 音频长度
    task,                         # 任务类型标识
    max_returned_tokens: int = 2048,  # 最大生成token数
    *,
    temperature: float = 1.0,     # 采样温度
    top_k: Optional[int] = None,  # top-k采样参数
    top_p: float = 1.0,          # top-p采样参数
    eos_id_a: Optional[int] = None,  # 音频结束标记
    eos_id_t: Optional[int] = None,  # 文本结束标记
    pad_id_t: Optional[int] = None,  # 文本填充标记
    shift: Optional[int] = None,     # token偏移量
    include_prompt: bool = True,     # 是否包含提示
    generate_text=False,             # 是否生成文本
) -> torch.Tensor:
    """音频到音频的自回归生成函数
    
    该函数实现了音频到音频的自回归生成过程,支持7层并行生成。
    主要用于A1A2等音频生成任务。
    """
    # 获取序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device

    # 初始化输出列表(7层音频+1层文本)
    output = [[] for _ in range(8)]
    
    # 生成第一个token
    tokens_A, token_T = next_token_A1T2(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],  # 减3是因为特殊标记占用的位置
        ["A1T2"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # 保存第一个token到输出列表
    for i in range(7):
        output[i].append(tokens_A[i].clone().tolist()[0])
    output[7].append(token_T.clone().tolist()[0])

    # 初始化位置编码
    input_pos = torch.tensor([T], device=device)

    # 标记文本是否结束
    text_end = False
    
    # 主生成循环
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        # 准备下一步的输入
        model_input_ids = []
        for i in range(7):
            # 对每一层的token进行layershift处理
            model_input_ids.append(
                layershift(tokens_A[i].clone(), i)
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        # 添加文本层的输入
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))

        # 生成下一个token
        tokens_A, token_T = next_token_A1T2(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # 如果文本结束,使用padding
        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        # 检查是否达到结束条件
        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            text_end = True

        # 保存生成的token
        for i in range(7):
            output[i].append(tokens_A[i].clone().tolist()[0])
        output[7].append(token_T.clone().tolist()[0])
        
        # 更新位置编码
        input_pos = input_pos.add_(1)

    return output


@torch.inference_mode()
def generate_ASR(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 1200,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    """语音识别生成函数(Automatic Speech Recognition)
    
    该函数实现了音频到文本的直接转录功能,将输入的音频特征转换为对应的文本序列。
    它使用自回归方式逐token生成,直到遇到结束标记或达到最大长度。
    
    参数:
        model: GPT模型实例
        audio_features: Whisper提取的音频特征, shape为(1, T, dim)
        input_ids: 8层输入ID列表(7层音频+1层文本)
        leng: 音频长度
        task: 任务类型标识
        max_returned_tokens: 最大生成token数,默认1200
        temperature: 采样温度,控制随机性,默认1.0
        top_k: 只保留概率最高的k个候选,默认None
        top_p: 累积概率阈值采样,默认1.0
        eos_id_a: 音频结束标记
        eos_id_t: 文本结束标记
        pad_id_t: 文本填充标记
        shift: token偏移量
        include_prompt: 是否包含提示在输出中
        generate_text: 是否生成文本
    
    返回:
        torch.Tensor: 生成的文本token序列
    
    工作流程:
    1. 获取输入序列长度和设备信息
    2. 使用next_token_A1T1生成第一个token
    3. 进入自回归生成循环:
       - 准备模型输入(7层音频end标记+1层文本)
       - 生成下一个token
       - 检查是否遇到结束标记
       - 更新位置编码
    """
    # 获取输入序列长度和设备信息
    T = input_ids[0].size(1)
    device = input_ids[0].device
    
    # 初始化输出列表
    output = []
    
    # 生成第一个token
    token_T = next_token_A1T1(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],  # 减3是因为特殊标记占用的位置
        ["asr"],  # 使用asr任务标识
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    output.append(token_T.clone().tolist()[0])
    
    # 初始化位置编码
    input_pos = torch.tensor([T], device=device)
    
    # 标记文本是否结束
    text_end = False
    
    # 自回归生成循环
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        # 准备模型输入:为每一层构建输入ID
        model_input_ids = []
        for i in range(7):
            # 对每一层添加音频结束标记
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        # 添加上一步生成的文本token
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        
        # 生成下一个token
        token_T = next_token_A1T1(
            model,
            None,  # 自回归阶段不需要音频特征
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 检查是否遇到结束标记
        if token_T == eos_id_t:
            break
            
        # 保存生成的token并更新位置编码
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
        
    return output
