import importlib
import os
import shutil


def load(module_name):
    old_file = module_name + ".cp310-win_amd64.pyd"
    new_file = "new_" + module_name + ".cp310-win_amd64.pyd"

    if os.path.exists(new_file):
        if os.path.exists(old_file):
            os.remove(old_file)

        shutil.move(new_file, old_file)

    importlib.import_module(module_name)

# ============================================================================
# loader —— 自更新交换加载器(重建自 loader.cp310-win_amd64.pyd,Cython 0.29.37)
#
# 行号契约:上方 1-16 行与原版 loader.py 逐行对齐(E2 回溯串 py_line 证据:
# def=6, old_file=7, new_file=8, if=10, if=11, remove=12, move=14, import=16)。
# E1 修正(oracle 差分证据):行 16 的 import_module 不受 if 门控——
# 无新文件时同样执行导入(No module named 'X' 即其可观察行为)。
# 修改任何可执行行前,必须同步核对 contracts/modules/loader.json 的块/边编号。
# ============================================================================
