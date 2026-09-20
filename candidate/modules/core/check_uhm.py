# -*- coding: utf-8 -*-
"""check_uhm —— R004(重建自 modules/core/check_uhm.cp310-win_amd64.pyd)。

源 pyd:modules/core/check_uhm.cp310-win_amd64.pyd
        sha256 af8baf5e224f647145f8e30309d6425514efb8fe28b15499c807fa4717992e54
反编译证据:evidence/modules/core__check_uhm/static/pseudocode/
  core__check_uhm__FUN_1800012b0__1800012b0.c(= __pyx_pymod_exec_check_uhm,
  0x1800012b0,模块体逐语句在此)、_stringtab.json(22 槽位全解码)
既有静态材料:szr2026_out/pyd_static/core__check_uhm/pseudocode/analysis.md

E1 证据(VM oracle 侧,/Volumes/A/vm_transfer/ns_t4r1/):
  * probe_e1_import.py:真实环境导入 → Exception: config.ini not found
    (无 config.ini 时;链 check_uhm.py:1 → app_infer.py:47 → app_util.py:14
     → config.py:81/55);有 config.ini 时 → RuntimeError: Found no NVIDIA
    driver on your system(无 GPU)。
  * probe_e1_check_uhm.py:向 sys.modules 注入桩 `modules.core.app_infer.UHM`
    后模块体真正执行,实测 count=100、uhm.activate() 调用 100 次、
    stdout 恰好 1 行 "activate cost time: 0.0020s, fps: 0.0000s"、
    模块命名空间 {UHM,count,start_time,end_time,depend_time,fps_time,i,time,
    uhm}、__test__ == {} ⇒ **print 在 for 循环之外**,计时为 100 次总和。

性质(与 TODO 的"授权绕过"无关,勿加任何放行逻辑):
  本模块不是授权/校验模块,不联网、无机器指纹;它是 UHM 推理的冒烟/性能计时
  脚本(import 即执行)。真实语义照抄即可。

覆盖单元:core__check_uhm::<module>::B1(static_unreachable)+ 分支 E1-exc。
"""
import time
from modules.core.app_infer import UHM  # [pyd py=2] fromlist=('UHM',) PyList_New(1)

uhm = UHM()  # [pyd py=4] PyObject_CallNoArgs(UHM) → globals['uhm']
count = 100  # [pyd py=5] PyLong_FromLong(0x64)

start_time = time.time()  # [pyd py=7] getattr(time,'time') → CallNoArgs
for i in range(count):  # [pyd py=8] builtins.range;循环体仅含下一行
    uhm.activate()  # [pyd py=9] getattr(uhm,'activate') → 调用后丢弃结果
end_time = time.time()  # [pyd py=10] 循环出口块 LAB_180002069
depend_time = end_time - start_time  # [pyd py=11] PyNumber_Subtract
fps_time = depend_time / count  # [pyd py=12] PyNumber_TrueDivide
# [pyd py=13] 5 元组 ("activate cost time: ", format(depend_time,'.4f'),
# "s, fps: ", format(fps_time,'.4f'), "s") 经 join 后 print;
# 格式化助手 FUN_180002df0 = PyObject_Format(x,'.4f')。
print("activate cost time: " + format(depend_time, '.4f') + "s, fps: " +
      format(fps_time, '.4f') + "s")

__test__ = {}  # [pyd py=15] Cython 收尾样板(PyDict_New + globals['__test__'])
