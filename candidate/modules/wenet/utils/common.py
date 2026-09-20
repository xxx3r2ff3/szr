# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""
#
# ============================================================================
# R068 重建物:modules/wenet/utils/common.cp310-win_amd64.pyd
#   source_sha256 = 2b344d4af5b84e226170babede4cd85b671811852b8f656a1d1752d37bf87c45
#
# 二进制出处(逐行):
#   [pyd imports.txt]  common.pad_list / common.add_blank / common.add_sos_eos /
#                      common.reverse_pad_list / common.th_accuracy /
#                      common.get_subsample / common.log_add(.genexpr) /
#                      common.mask_to_bias / common.get_nested_attribute /
#                      common.lrs_to_str / common.is_torch_npu_available /
#                      common.__pyx_scope_struct__log_add /
#                      common.__pyx_scope_struct_1_genexpr /
#                      common.__pyx_scope_struct_2_genexpr /
#                      torch / torch.nn.utils.rnn / collections.abc / common.py
#   [pyd strtab]       'IGNORE_ID' 'Unsupported ndim: ' 'pad_res' '_blank'
#                      'pad_pred' 'r_ys_pad' 'input_layer' 'cur_step'
#                      'TORCH_NPU_AVAILABLE'(运行期名) 'torch_npu'
#                      'Module "torch_npu" not found. "pip install torch_npu"'
#                      + 17 空格 + 'if you are using Ascend NPU, otherwise,
#                      ignore it'(与上游反斜杠续行文本逐字节一致)
#                      文档串全文:pad_list / add_blank / add_sos_eos /
#                      reverse_pad_list / th_accuracy / is_torch_npu_available
#   [pyd line-tab]     'pad_list (line 30)' 'add_blank (line 76)'
#                      'add_sos_eos (line 108)' 'reverse_pad_list (line 153)'
#                      → 上游对应 30 / 79 / 113 / 241;见报告 §6「行号偏差」
#   [runtime ev]       evidence/modules/wenet__utils__common/runtime/*.json
#                      15 个 callable 的签名/默认值与上游逐字相同
#   [E1]               evidence/modules/wenet__utils__common/probes/
#                      probe_common.py + probe_common_out.json(70 条探针)
#   [upstream]         wenet-e2e/wenet @d170596 wenet/utils/common.py
#
# [本地改动] 上游 common.py 有 `from whisper.tokenizer import LANGUAGES as
#   WhiserLanguages` + `WHISPER_LANGS` + `add_whisper_tokens`(含 NotImplementedError
#   占位)。本 pyd 的 imports.txt 无 whisper 依赖、strtab 无 'sot_prev'/
#   'transcribe'/'translate'/'no_timestamps'/'eot'/'WHISPER_LANGS',
#   runtime module_dir 也无该符号 → 本重建物删去 whisper 相关三名。
# [本地改动·docstring] 逐字节比对(ast.get_docstring 原文 in pyd bytes):
#   本文件 9/9 段 docstring 与上游逐字节相同,**两处例外**,已按 pyd 实证修正:
#     1) add_blank:上游为 `""" Prepad blank for transducer predictor`(引号后
#        一个空格),pyd 内实际串为 `Prepad blank for transducer predictor`
#        (无前导空格)→ 本文件写作 `"""Prepad ...`;
#     2) is_torch_npu_available:上游 docstring 内层缩进 8 空格,pyd 内实际串为
#        '\n    check if torch_npu is available.\n    torch_npu is a npu adapter
#        of PyTorch\n    '(内层 4 空格)→ 本文件按 4 空格书写。
#   二者均不影响可观测行为(函数对象不可编码),仅作保真登记。
# ============================================================================

import math
import time
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    # [E1 probe_common :: pad_list_1d/2d/3d/int_dtype/float_padvalue]
    #   ndim==1 → (B,Tmax);ndim==2 → (B,Tmax,xs[0].shape[1]);
    #   ndim==3 → (B,Tmax,shape[1],shape[2]);dtype/device 取 xs[0]。
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        # [E1 probe_common :: pad_list_ndim4 → ValueError 'Unsupported ndim: 4';
        #  pad_list_ndim0 → TypeError 'len() of a 0-d tensor'(先算 max_len 再取 ndim);
        #  pad_list_empty → ValueError 'max() arg is an empty sequence']。
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_blank(ys_pad: torch.Tensor, blank: int,
              ignore_id: int) -> torch.Tensor:
    """Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    # [E1 probe_common :: add_blank_basic] 输出 dtype int64(与入参 int32 不同);
    #   [add_blank_no_ignore_hit] ignore_id=-100 时不改写 -1;
    #   [add_blank_blank_eq_ignore] blank=-1/ignore_id=-1 时前导位为 -1;
    #   [add_blank_float] 比较用 `out == ignore_id`(float 输入同样成立);
    #   [add_blank_0rows] (0,L) 输入 → (0,L+1);
    #   [add_blank_1d] 1-D 输入 → RuntimeError(维度不匹配)。
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    # [E1 probe_common :: add_sos_eos_basic / add_sos_eos_all_pad]
    #   全 pad 行 → ys_in=[sos,eos,eos](pad_list 用 eos 填充)、
    #   ys_out=[eos,ignore_id,...];dtype 统一 int64。
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    # [E1 probe_common :: reverse_pad_list_default(pad_value 缺省 = -1.0,
    #   输出 dtype int32 来自 .int())/ _pad0 / _lens_list(ys_lens 可为 list)/
    #   _len0(长度 0 行整行 pad)]。
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    # [E1 probe_common :: th_accuracy_basic → 0 维 float32 tensor 1.0;
    #   th_accuracy_all_ignore → nan(0/0);th_accuracy_returns_tensor → 'Tensor'
    #   (返回 (numerator/denominator).detach(),不是 Python float)]。
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_subsample(config):
    # [E1 probe_common :: get_subsample_conv2d/6/8 → 4/6/8;
    #   get_subsample_conv1d → AssertionError;get_subsample_missing_key →
    #   KeyError 'encoder_conf']。
    input_layer = config["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"]
    if input_layer == "conv2d":
        return 4
    elif input_layer == "conv2d6":
        return 6
    elif input_layer == "conv2d8":
        return 8


def log_add(*args) -> float:
    """
    Stable log add
    """
    # [E1 probe_common :: log_add_two → -0.6867383124817772;
    #   log_add_one → -3.5;log_add_three → -0.5923940355556196;
    #   log_add_all_neginf → -inf;log_add_empty → ValueError(max());
    #   log_add_list_arg → TypeError(形参为 *args,不是单个 list)]。
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [E1 probe_common :: mask_to_bias_f32 → [(1-0)*-1e10 ...] 且 True 位为 -0.0;
    #   _bf16 → -9999220736.0; _f16 → -inf(溢出);
    #   _bad_dtype(float64) → AssertionError; _nonbool(int64) → AssertionError]。
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def get_nested_attribute(obj, attr_path):
    # [E1 probe_common :: get_nested_attr_plain → 7; _single → 内层对象;
    #   _missing → AttributeError "'Outer' object has no attribute 'nope'";
    #   attr_path 以 '.' 切分,DistributedDataParallel 先取 .module]。
    if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
        obj = obj.module
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def lrs_to_str(lrs: List):
    # [E1 probe_common :: lrs_to_str → '1.0000e-03 2.5000e-05 0.0000e+00';
    #   lrs_to_str_empty → '']。
    return " ".join(["{:.4e}".format(lr) for lr in lrs])


class StepTimer:
    """Utility class for measuring steps/second."""

    # [E1 probe_common :: StepTimer_default_last_iteration → 0.0;
    #   StepTimer_step_arg(step=2.5) → 2.5;StepTimer_attrs →
    #   ['start','steps_per_second'];StepTimer_has_last_time → True;
    #   StepTimer_sps_restart_true(step=1.0, cur=3.0) → last_iteration 1.0→3.0;
    #   StepTimer_sps_restart_false → last_iteration 保持 1.0]。
    def __init__(self, step=0.0):
        self.last_iteration = step
        self.start()

    def start(self):
        self.last_time = time.time()

    def steps_per_second(self, cur_step, restart=True):
        value = ((float(cur_step) - self.last_iteration) /
                 (time.time() - self.last_time))
        if restart:
            self.start()
            self.last_iteration = float(cur_step)
        return value


def tensor_to_scalar(x):
    # [E1 probe_common :: tensor_to_scalar_tensor → 3.5; _int → 7;
    #   _2d → 1.0(item() 对单元素张量成立);_str → 原样返回]。
    if torch.is_tensor(x):
        return x.item()
    return x


def is_torch_npu_available() -> bool:
    '''
    check if torch_npu is available.
    torch_npu is a npu adapter of PyTorch
    '''
    # [E1 probe_common :: is_torch_npu_available → False;stdout 恰为
    #   'Module "torch_npu" not found. "pip install torch_npu"' + 17 空格 +
    #   'if you are using Ascend NPU, otherwise, ignore it\n'(pyd strtab 内
    #   该串逐字节相同,故下面反斜杠续行的缩进必须保持 16 空格);
    #   cuda_is_available → False(本机无 CUDA,走 print 分支)]。
    try:
        import torch_npu  # noqa
        return True
    except ImportError:
        if not torch.cuda.is_available():
            print("Module \"torch_npu\" not found. \"pip install torch_npu\" \
                if you are using Ascend NPU, otherwise, ignore it")
    return False


# [E1 probe_common :: TORCH_NPU_AVAILABLE → False]
TORCH_NPU_AVAILABLE = is_torch_npu_available()
