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
"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = False,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(
                idim, hidden_units, dropout_rate, activation, bias=bias)
            for _ in range(n_expert))
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.size(
        )  # batch size, sequence length, embedding dimension (idim)
        xs = xs.view(-1, D)  # (B*L, D)
        router = self.gate(xs)  # (B*L, n_expert)
        logits, selected_experts = torch.topk(
            router, self.n_expert_activated
        )  # probs:(B*L, n_expert_activated), selected_exp: (B*L, n_expert_activated)
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_activated)
        output = torch.zeros_like(xs)  # (B*L, D)
        for i, expert in enumerate(self.experts):
            mask = selected_experts == i
            token_ids, ith_expert = torch.where(mask)
            output[token_ids] += weights[token_ids, ith_expert, None] * expert(
                xs[token_ids])
        return output.view(B, L, D)


class GatedVariantsMLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.GELU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(GatedVariantsMLP, self).__init__()
        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        # w_1 as up proj
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # w_2 as down proj
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, x) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        gate = self.activation(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        return self.w_2(self.dropout(fuse))

# ---------------------------------------------------------------------------
# 重建出处与证据(R062/R063/R064/R065 通用规范,见 docs/PARALLEL_PLAYBOOK.md §2/§3)
#   目标 pyd : modules/wenet/transformer/positionwise_feed_forward.cp310-win_amd64.pyd
#              sha256 = 63ff1de07aa2439c5621e9932222d7170ba05ecbf82b530fb758ed467f213c5a
#   上游对齐 : wenet-e2e/wenet @ 6480b8fffe372a170989fd65f9b9bd97ff368df3
#              https://github.com/wenet-e2e/wenet/blob/6480b8fffe372a170989fd65f9b9bd97ff368df3/wenet/transformer/positionwise_feed_forward.py
#              (逐方法行号对齐,E1 依据见下)
#   行号对齐 : PositionwiseFeedForward.__init__ L33 / .forward L50; MoEFFNLayer.__init__ L78 / .forward L97; GatedVariantsMLP.__init__ L128 / .forward L148
#   E1 证据  : 共享目录 ns_wenet2/probe_surface.json、probe_detail.json、
#              probe_lines.json、probe_code.json(探针脚本 probes/*.py,
#              已抄入 evidence/modules/wenet__transformer__positionwise_feed_forward/runtime/)
#              P3(pff.MoEFFNLayer.__init__:89, .forward:105, GatedVariantsMLP.forward:156, PositionwiseFeedForward.forward:58 回溯)/P4
#   E2 证据  : evidence/modules/wenet__transformer__positionwise_feed_forward/static/{api_surface.md,imports.txt,constants.txt}
#              (字符串表含本文件全部标识符:类名/方法名/注册键)
#   约束     : 仅依赖 torch 与同树候选源码;不 import 任何业务 .pyd;
#              行内不含 NotImplementedError/pass 占位。
#   说明     : 本块为重建登记信息,置于源文件末尾以保持方法 def 行号与
#              上游(及 oracle co_firstlineno)完全一致。
# ---------------------------------------------------------------------------
