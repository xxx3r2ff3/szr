# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# R069 重建物:modules/wenet/utils/mask.cp310-win_amd64.pyd
#   source_sha256 = 25efa0f968db9809c5c61e1e4d907a82b9fe9cb6362a73f52df5ea93be38ec6b
#
# 二进制出处(逐行;本模块无 import 除 torch 外任何符号 → 逐行同构上游):
#   [pyd imports.txt]  mask.subsequent_mask / mask.subsequent_chunk_mask /
#                      mask.add_optional_chunk_mask / mask.make_pad_mask /
#                      mask.make_non_pad_mask / mask.mask_finished_scores /
#                      mask.mask_finished_preds / mask.causal_or_lookahead_mask /
#                      mask.__defaults__ / torch / mask.py
#   [pyd strtab]      8 个函数全部 docstring 与上游**逐字节相同**
#                     (ast.get_docstring 原文 in pyd bytes → True,8/8);
#                     常量串 '0'/'1'/'eos'/'masked_fill_'/'repeat'/'clamp'
#                     之外无本地新增串
#   [pyd line-tab]     'subsequent_mask (line 52)'      'subsequent_chunk_mask (line 88)'
#                      'make_pad_mask (line 201)'       'make_non_pad_mask (line 230)'
#                      'causal_or_lookahead_mask (line 307)'
#                      → 与上游 def 行号 **完全一致**(5/5),故本文件按上游原文
#                        保留第 17-49 行那段被注释掉的旧 subsequent_mask 实现,
#                        使函数体文本与上游逐行对齐;(line N) 表项只出现在含
#                        doctest 例子的函数上,其数值与该函数 def 行号相等。
#                        注意:本候选按规范插入了行内 `# [E1 …]` 证据注释,
#                        因此候选文件的绝对 def 行号 ≠ pyd 行号(偏移非恒定),
#                        行号一致性以「与上游原文对齐」为准,见报告 §6。
#   [runtime ev]       evidence/modules/wenet__utils__mask/runtime/*.json
#                      8 个函数签名/默认值(含 enable_full_context=True、
#                      max_chunk_size=25、num_left_chunks=-1、left_t_valid=0、
#                      max_len=0、device=cpu)与上游逐字相同
#   [E1]               evidence/modules/wenet__utils__mask/probes/probe_mask.py
#                      + probe_mask_out.json(51 条探针)
#   [upstream]         wenet-e2e/wenet @d170596 wenet/utils/mask.py
#
# 关键 E1 判定(逐条见 probe_mask_out.json):
#   * subsequent_mask: arange→expand→unsqueeze(-1)→`<=`,返回 torch.bool;
#     size=0 → (0,0)。
#   * subsequent_chunk_mask: 逐行 for 循环(非向量化);num_left_chunks<0 → start=0;
#     chunk_size<=0 走 Python 整除语义(chunk_size=0 → ZeroDivisionError;
#     chunk_size=-1 → 首行 ret[0, 0:-1] 等实测形态)。无 dtype 形参
#     (probe :: scm_dtype_has_dtype_param → False)。
#   * add_optional_chunk_mask: use_dynamic_chunk 且 decoding_chunk_size==0 时
#     消费 torch RNG(randint(1,max_len) → 可能再 randint(0,max_left_chunks)),
#     probe :: rng_replay_seed0 实测 seed0/L=40 首个 randint=30 → 30>20 且
#     enable_full_context=True → chunk_size=max_len(第二次 randint 不消费);
#     max_len==1 → RuntimeError('random_ expects from < to');
#     masks 非 bool → RuntimeError('bitwise_and_cpu not implemented for Float')。
#   * make_pad_mask: max_len<=0 时用 lengths.max();len=0 的 batch → RuntimeError。
#   * mask_finished_scores: beam_size>1 用 cat((zero,flag.repeat))/cat((flag,zero.repeat)),
#     beam_size==1 直接用 zero_mask/flag;先填 -inf 再填 0。
#   * mask_finished_preds: flag.repeat([1, beam_size]) 后 masked_fill_(…, eos)。
#   * causal_or_lookahead_mask: start/end 用 torch.where 构造,返回
#     (gt & lt) * mask.transpose(1, 2) * mask(bool 相乘仍为 bool)。
# ============================================================================

import torch
'''
def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
'''


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    # [E1 probe_mask :: subsequent_mask_3 / _0 / _1 / _device]
    #   (3,3) 下三角 bool;[0,0];(2,2) device=cpu 可传。
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    # [E1 probe_mask :: scm_4_2 / _4_2_l0 / _4_2_l1 / _6_3_lm1 / scm_0_2 /
    #   scm_5_1 / scm_5_10 / scm_neg_chunk(chunk_size=-1) / scm_dev /
    #   scm_5_0 → ZeroDivisionError / scm_dtype_has_dtype_param → False]
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True,
                            max_chunk_size: int = 25):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, max_chunk_size] or full context(max_len)
            False: chunk size ~ U[1, max_chunk_size]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    # [E1 probe_mask :: aocm_static_only(static=4) / aocm_static_zero_nomask
    #   (static=0 且非动态 → 原样返回 masks) / aocm_dyn_dec_neg(dec<0 →
    #   chunk_size=max_len,num_left_chunks=-1) / aocm_dyn_dec_pos(dec=3,
    #   left=2 / left=4 形状相同,仅 start 列不同) / aocm_dyn_seed0/1 /
    #   aocm_dyn_seed0_full_False / aocm_dyn_seed0_left /
    #   aocm_zero_maxlen → RuntimeError / aocm_static_nonbool_mask → RuntimeError]
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, max_chunk_size] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % max_chunk_size + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    # [E1 probe_mask :: make_pad_mask([5,3,2]) → (3,5); make_pad_mask_maxlen(6)
    #   → (3,6); _maxlen_neg(-1) → 退回 lengths.max(); _maxlen_small(3) → (3,3);
    #   _zero_lens([0,2]) 首行全 True; _empty → RuntimeError(max 空张量);
    #   _list → AttributeError]
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    # [E1 probe_mask :: make_non_pad_mask([5,3,2]) → make_pad_mask 的取反;
    #   _empty → RuntimeError]。
    return ~make_pad_mask(lengths)


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    # [E1 probe_mask :: mfs_beam3 → [[0,-inf,-inf],[4,5,6]];mfs_beam1 →
    #   [[0],[2]];mfs_dtype(float32);mfs_int_score(int 输入)→ RuntimeError;
    #   mfs_noncontig_flag 正常]
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    # [E1 probe_mask :: mfp_beam3(flag=True 行整行 → 11)/ mfp_eos0(→0)/
    #   mfp_dtype(int32 保持)]
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def causal_or_lookahead_mask(
    mask: torch.Tensor,
    right_context: int,
    left_context: int,
    left_t_valid: int = 0,
) -> torch.Tensor:
    """Create mask (B, T, T) with history or future or both,
       this is for causal or noncausal streaming encoder

    Args:
        mask (torch.Tensor): size of mask shape (B, 1, T)
        right_context (int): future context size
        left_context (int): history context size
        left_t_valid (int): valid start offset

    Returns:
        torch.Tensor: mask shape (B, T, T)

    Examples:
        >>> seq_len  = torch.tensor([2,3,4])
        >>> seq_mask = make_non_pad_mask(seq_len)
        [[1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 0, 2)
        [[[1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 1, 1, 1]]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 1, 2)
        [[[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1],
         [0, 1, 1, 1]]]
    """
    # [E1 probe_mask :: colm_r0_l2 / colm_r1_l2 / colm_r0_l0 / colm_left_valid
    #   (left_t_valid=2)/ colm_left_valid_lt / colm_bool_input]
    #   → 返回 (B,T,T) torch.bool;`*` 为 bool 乘法(仍 bool)。
    _, _, T = mask.size()
    indices = torch.arange(T, device=mask.device)
    start = torch.where(indices > left_context, indices - left_context, 0)
    start = torch.where(indices < left_t_valid, indices, start).unsqueeze(1)

    end = indices + right_context + 1
    end = end.unsqueeze(1)
    indices_expand = indices.unsqueeze(0)
    gt = (indices_expand >= start)
    lt = (indices_expand < end)

    return (gt & lt) * mask.transpose(1, 2) * mask
