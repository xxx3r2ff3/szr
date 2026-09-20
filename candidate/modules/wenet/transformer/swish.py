# Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
#               2020 Northwestern Polytechnical University (Pengcheng Guo)
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
"""Swish() activation function for Conformer."""

import torch


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

# ---------------------------------------------------------------------------
# 重建出处与证据(R062/R063/R064/R065 通用规范,见 docs/PARALLEL_PLAYBOOK.md §2/§3)
#   目标 pyd : modules/wenet/transformer/swish.cp310-win_amd64.pyd
#              sha256 = 334d4293c83f0af3e2818a17e38453e3c80b67faffe87ba013cab221807b1bda
#   上游对齐 : wenet-e2e/wenet @ 6480b8fffe372a170989fd65f9b9bd97ff368df3
#              https://github.com/wenet-e2e/wenet/blob/6480b8fffe372a170989fd65f9b9bd97ff368df3/wenet/transformer/swish.py
#              (逐方法行号对齐,E1 依据见下)
#   行号对齐 : Swish.forward L24
#   E1 证据  : 共享目录 ns_wenet2/probe_surface.json、probe_detail.json、
#              probe_lines.json、probe_code.json(探针脚本 probes/*.py,
#              已抄入 evidence/modules/wenet__transformer__swish/runtime/)
#              P1/P2/P3(swish.Swish.forward:26 回溯)/P4
#   E2 证据  : evidence/modules/wenet__transformer__swish/static/{api_surface.md,imports.txt,constants.txt}
#              (字符串表含本文件全部标识符:类名/方法名/注册键)
#   约束     : 仅依赖 torch 与同树候选源码;不 import 任何业务 .pyd;
#              行内不含 NotImplementedError/pass 占位。
#   说明     : 本块为重建登记信息,置于源文件末尾以保持方法 def 行号与
#              上游(及 oracle co_firstlineno)完全一致。
# ---------------------------------------------------------------------------
