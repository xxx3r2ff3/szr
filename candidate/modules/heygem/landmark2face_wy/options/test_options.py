# -*- coding: utf-8 -*-
# 出处:landmark2face_wy/options/test_options.pyc(py3.8,反汇编见
# evidence/modules/_heygem_orphans/disasm/landmark2face_wy_options_test_options.pyc.dis)。
# 该 .dis 模块 [Constants] 第一项是 1(不是字符串)=> 模块级没有 docstring,出处写在注释里。
# 恢复说明:initialize() 的 9 个 add_argument(参数名/type/default/help)逐字取自
# [Disassembly] 的 LOAD_CONST 与 CALL_FUNCTION_KW 关键字元组;末尾 self.isTrain = False
# 与 `return parser` 亦按字节码还原。未还原细节:无。
from base_options import BaseOptions
from y_utils.config import GlobalConfig


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--model_path', type=str, default='./landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--test_muban', type=str)
        parser.add_argument('--test_audio_path', type=str)

        self.isTrain = False
        return parser
