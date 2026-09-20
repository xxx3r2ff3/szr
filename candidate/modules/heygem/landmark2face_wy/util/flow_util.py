# 出处:landmark2face_wy/util/flow_util.pyc(py3.8,反汇编见
# evidence/modules/_heygem_orphans/disasm/landmark2face_wy_util_flow_util.pyc.dis)。
# 该 .dis 模块 [Constants] 第一项是 0(不是字符串)=> 模块级没有 docstring,出处写在注释里。
# 恢复说明:三个函数 docstring 逐字取自 [Constants];函数体按 [Disassembly] 逐指令等价重建
# (含 make_coordinate_grid(flow) 的单参数签名、2*(x/(w-1))-1 归一化、warp_image 的
# 尺寸不等时才 interpolate 的分支、grid_sample 的调用)。未还原细节:无。
import torch


def convert_flow_to_deformation(flow):
    """convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b, c, h, w = flow.shape
    flow_norm = 2 * torch.cat([flow[:, :, :, :1] / (w - 1), flow[:, :, :, 1:] / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0, 2, 3, 1)
    return deformation


def make_coordinate_grid(flow):
    """obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b, c, h, w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    meshed = meshed.expand(b, -1, -1, -1)
    return meshed


def warp_image(source_image, deformation):
    """warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation)
