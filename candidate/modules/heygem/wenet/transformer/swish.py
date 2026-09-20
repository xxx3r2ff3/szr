# -*- coding: utf-8 -*-
# 出处:S003 恢复件(candidate/modules/heygem/wenet/transformer/swish.pyc,py3.8)。
# 反汇编:evidence/modules/_heygem_orphans/disasm/wenet_transformer_swish.pyc.dis
"""Swish() activation function for Conformer."""
import torch


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)
