# -*- coding: utf-8 -*-
# 出处:S003 恢复件(candidate/modules/heygem/wenet/transformer/convolution.pyc,py3.8)。
# 反汇编:evidence/modules/_heygem_orphans/disasm/wenet_transformer_convolution.pyc.dis
# 字节码全局名对照:check_argument_types <- typeguard;AssertionError <- 本文件 assert
# 语句隐式加载(py3.8 编译 assert 时 LOAD_GLOBAL AssertionError)。
"""ConvolutionModule definition."""
from typing import Optional, Tuple

import torch
from torch import nn
from typeguard import check_argument_types


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = 'batch_norm',
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        assert check_argument_types()
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder 标记是否为 causal 卷积:> 0 时是 causal 卷积。
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        assert norm in ['batch_norm', 'layer_norm']
        if norm == 'batch_norm':
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # 交换时间维与特征维
        x = x.transpose(1, 2)

        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache is None:
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder:]
        else:
            # 非 causal 卷积,不需要 cache。
            new_cache = torch.tensor([], dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache
