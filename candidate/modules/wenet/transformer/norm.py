import torch


class RMSNorm(torch.nn.Module):
    """ https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return x * (1 + self.weight)
        else:
            return x * self.weight

# ---------------------------------------------------------------------------
# 重建出处与证据(R062/R063/R064/R065 通用规范,见 docs/PARALLEL_PLAYBOOK.md §2/§3)
#   目标 pyd : modules/wenet/transformer/norm.cp310-win_amd64.pyd
#              sha256 = 5f3f95ad8314c31a9e4ee5e11e57b33e9739aedc337a3c041d1972c7059cbba2
#   上游对齐 : wenet-e2e/wenet @ 6480b8fffe372a170989fd65f9b9bd97ff368df3
#              https://github.com/wenet-e2e/wenet/blob/6480b8fffe372a170989fd65f9b9bd97ff368df3/wenet/transformer/norm.py
#              (逐方法行号对齐,E1 依据见下)
#   行号对齐 : RMSNorm.__init__ L8 / RMSNorm._norm L19 / RMSNorm.forward L22
#   E1 证据  : 共享目录 ns_wenet2/probe_surface.json、probe_detail.json、
#              probe_lines.json、probe_code.json(探针脚本 probes/*.py,
#              已抄入 evidence/modules/wenet__transformer__norm/runtime/)
#              P1 probe_surface.json / P2 probe_detail.json / P3 probe_lines.json (norm.RMSNorm.__init__:16, _norm:20, forward:23 回溯)/ P4 probe_code.json
#   E2 证据  : evidence/modules/wenet__transformer__norm/static/{api_surface.md,imports.txt,constants.txt}
#              (字符串表含本文件全部标识符:类名/方法名/注册键)
#   约束     : 仅依赖 torch 与同树候选源码;不 import 任何业务 .pyd;
#              行内不含 NotImplementedError/pass 占位。
#   说明     : 本块为重建登记信息,置于源文件末尾以保持方法 def 行号与
#              上游(及 oracle co_firstlineno)完全一致。
# ---------------------------------------------------------------------------
