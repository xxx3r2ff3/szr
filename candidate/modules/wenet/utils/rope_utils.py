#
# ============================================================================
# R070 重建物:modules/wenet/utils/rope_utils.cp310-win_amd64.pyd
#   source_sha256 = 0a2be51d847a1a736fe7802bd938ee2d9f304d7f2ed322f51ce01b98c5386189
#
# 二进制出处(逐行;除 torch 外无任何依赖 → 与上游逐行同构):
#   [pyd imports.txt]  rope_utils.precompute_freqs_cis /
#                      rope_utils.google_apply_rotary_emb /
#                      rope_utils.llama_apply_rotary_emb / torch / rope_utils.py
#   [pyd strtab]      'Applies the rotary embedding to the query and key tensors.'
#                     'freqs_cis' 'device' 'precompute_freqs_cis'
#                     + google/llama 两函数名;三个函数 docstring 与上游
#                     **逐字节相同**(ast.get_docstring 原文 in pyd bytes → 3/3)
#   [pyd 无 line-tab] 本模块无 "(line N)" 表项(三个函数 docstring 均无 doctest
#                     例子,与 mask/common 的规律一致)
#   [runtime ev]      evidence/modules/wenet__utils__rope_utils/runtime/*.json
#                     google_apply_rotary_emb(x: torch.Tensor, freqs_cis:
#                     torch.Tensor) -> torch.Tensor / llama 同 /
#                     precompute_freqs_cis(dim: int, end: int, theta: float =
#                     10000.0) -> torch.Tensor / WENET_APPLY_ROTARY_EMB 存在
#   [E1]              evidence/modules/wenet__utils__rope_utils/probes/
#                     probe_rope_utils.py + probe_rope_utils_out.json(20 条探针)
#   [upstream]        wenet-e2e/wenet @d170596 wenet/utils/rope_utils.py
#
# 关键 E1 判定(详见 probe_rope_utils_out.json):
#   * precompute_freqs_cis(8,4) → complex64 (4,4)=(end, dim//2);
#     dim=0 → (end,0);end=0 → (0,dim//2);dim=5(奇数)→(2, dim//2=2);
#     theta 可为 float/int。
#   * google/llama 的输出 dtype 跟随输入(.type_as(x)),复数中间量为 complex64;
#     两个函数在 freqs_cis 与 x 的可广播维度上语义不同:
#       - google: chunk(x.float(),2,-1)+stack → view_as_complex;乘后
#         cat(chunk(x_out,2,-1),dim=-2) 再 reshape(...,-1) —— 即
#         "后半段实部/虚部交错"布局;
#       - llama : reshape(*x.shape[:-1],-1,2) → view_as_complex;乘后
#         view_as_real().flatten(3) —— 即 "前半段/后半段分块"布局。
#     广播失败时二者报错维度不同(google: non-singleton dimension 2 或 3;
#     llama: dimension 2),见 probe :: google_apply_dim_mismatch /
#     google_apply_3d / llama_apply_dim_mismatch,可作为布局差异的判别证据。
#   * 真实调用形态(上游 attention.py:660 + embedding.py:222):
#     q/k 为 (B,T,H,D),pos_emb = 复数 (1,T,1,D/2) → 本重建物保持原式,
#     不额外 reshape。
# ============================================================================

import torch


# copy from:https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L84
def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    # [E1 probe_rope_utils :: precompute_freqs_cis_8_4 / _4_3 / _theta(100.0) /
    #   _odd_dim(5,2) / _dim0 / _end0 / _dtype(complex64) / _int_theta]
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# modified from:
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L95
def google_apply_rotary_emb(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    # [E1 probe_rope_utils :: google_apply_freqs_mismatch(dtype 保持 float32,
    #   形状不变 (1,3,2,8))/ google_apply_dim_mismatch / google_apply_3d]
    # [E1 probe_rope_utils :: google_apply_f16(freqs 复数与 fp16 相乘后在
    #   view_as_real 前升为 float32,最后 .type_as(x) 落回 fp16)]
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


def llama_apply_rotary_emb(x: torch.Tensor,
                           freqs_cis: torch.Tensor) -> torch.Tensor:
    # [E1 probe_rope_utils :: llama_apply_* 与 google 同签名同 dtype 规则,
    #   但广播失败维度不同 → 布局不同(见文件头说明)]。
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


# [E1 probe_rope_utils :: WENET_APPLY_ROTARY_EMB → 键 ['google','llama'],
#  值身份与模块级同名函数相同(dict_values_identity → [True, True])]
WENET_APPLY_ROTARY_EMB = {
    'google': google_apply_rotary_emb,
    'llama': llama_apply_rotary_emb,
}
