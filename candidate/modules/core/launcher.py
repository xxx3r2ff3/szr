# -*- coding: utf-8 -*-
"""launcher —— R007(重建自 modules/core/launcher.cp310-win_amd64.pyd)。

源 pyd:modules/core/launcher.cp310-win_amd64.pyd
        sha256 4b3c5217177a3b58a79a7862c5952c98f09f334e8d11963fc053b7f5e7784ca0
反编译证据:evidence/modules/core__launcher/static/pseudocode/
  core__launcher__FUN_180001010__180001010.c(= __pyx_pf_7launcher_main,0x180001010,13875 B)
  core__launcher__FUN_180004900__180004900.c(= __pyx_pymod_exec_launcher,0x180004900)
  _stringtab.json(StringTab 槽位 → 字符串)

E1 证据(VM oracle 侧探针,脚本/输出存 /Volumes/A/vm_transfer/ns_t4r1/):
  probe_e1_launcher.py / probe_e1_launcher2.py / probe_e1_launcher3.py /
  probe_e1_launcher4.py(+ *_out.json):
  * runtime_dir 取自 sys.executable(basename 测试:把 launcher.pyd 复制到
    C:\\szr2026_testdata\\ns_t4r1\\probe_launcher_isofile\\ 再导入,结果路径完全
    不变 → 与 __file__ 无关);
  * Popen 实参 = [<runtime_dir>\\runtime\\python.exe, '-u', <runtime_dir>\\start.py],
    cwd=runtime_dir,env=os.environ.copy() 追加 5 个键;
  * 正常路径 stdout 两行 + 返回子进程退出码(int);
  * Popen 抛异常 → stdout '启动过程中出错: {e}' + traceback.print_exc() → 返回 1;
  * env 里无 'PATH' 时 PATH 不加尾部 pathsep。

覆盖单元:core__launcher::main::B0(契约 functions[0].blocks)。
"""
import os
import subprocess
import sys
import traceback


def main():
    # [pyd main 0x180001010 +0x653..+0x66d / py=8]
    # os.path.dirname(sys.executable) → runtime_dir。
    # E1(probe_e1_launcher3):换 __file__ 位置不影响结果 ⇒ 不依赖 __file__。
    runtime_dir = os.path.dirname(sys.executable)
    # [pyd +0x2ae0..+0x2b18 / py=9] os.path.join(runtime_dir, 'runtime')
    one_path = os.path.join(runtime_dir, 'runtime')
    # [pyd +0x... / py=10] os.path.join(one_path, 'python.exe')
    python_exe = os.path.join(one_path, 'python.exe')
    # [pyd py=11] os.path.join(runtime_dir, 'start.py')
    start_script = os.path.join(runtime_dir, 'start.py')
    # [pyd py=12] os.path.join(runtime_dir, 'bin')
    bin_dir = os.path.join(runtime_dir, 'bin')
    # [pyd py=13] os.path.join(one_path, 'Scripts')
    scripts_dir = os.path.join(one_path, 'Scripts')
    # [pyd py=14] os.path.join(one_path, 'Library', 'bin')
    library_dir = os.path.join(one_path, 'Library', 'bin')
    # [pyd py=15] os.path.join(one_path, 'cuda', 'v11.8', 'bin')
    cuda_dir = os.path.join(one_path, 'cuda', 'v11.8', 'bin')
    # [pyd py=16] os.path.join(one_path, 'cuda', 'v12.8', 'bin')
    cuda12_dir = os.path.join(one_path, 'cuda', 'v12.8', 'bin')
    # [pyd +0x1800027f0 区域 / py=17-19] os.path.exists(cuda12_dir) 为真时改用 v12.8;
    # E1(probe_e1_launcher2):本机无该目录 ⇒ PATH 用 v11.8。
    if os.path.exists(cuda12_dir):
        cuda_dir = cuda12_dir
    # [pyd py=20-22] env = os.environ.copy();['LAUNCHED_FROM_EXE']=sys.executable;
    # ['PYTHON_EXECUTABLE']=python_exe(E1 实测:os.environ 本身未被修改)
    # —— 逐条对应 pyd 内 PyObject_GetAttr(...,'environ')→'copy' 与
    #    PyObject_SetItem(env,'LAUNCHED_FROM_EXE'/'PYTHON_EXECUTABLE')。
    env = os.environ.copy()
    env['LAUNCHED_FROM_EXE'] = sys.executable
    env['PYTHON_EXECUTABLE'] = python_exe
    # [pyd py=25] os.path.join(one_path, 'Library', 'lib', 'tcl8.6')
    env['TCL_LIBRARY'] = os.path.join(one_path, 'Library', 'lib', 'tcl8.6')
    # [pyd py=26] os.path.join(one_path, 'Library', 'lib', 'tk8.6')
    env['TK_LIBRARY'] = os.path.join(one_path, 'Library', 'lib', 'tk8.6')
    # [pyd PyList_New(5) @0x1800029.. / py=28-29] 顺序固定为
    #   [one_path, bin_dir, scripts_dir, library_dir, cuda_dir]
    # (E1: PATH 前缀逐项核对一致)
    paths = os.pathsep.join([one_path, bin_dir, scripts_dir, library_dir, cuda_dir])
    # [pyd PySequence_Contains(env,'PATH') @…+0x9b4 / py=31-34]
    if 'PATH' in env:
        env['PATH'] = paths + os.pathsep + env['PATH']
    else:
        env['PATH'] = paths
    # [pyd __Pyx_ExceptionSave @+0x…(LAB_1800043..) / py=36] try:
    try:
        # [pyd PyObject_GetAttr(subprocess,'Popen')+PyList_New(3)/PyDict_New
        #  +0x9ff..+0xa3e / py=37-40]
        process = subprocess.Popen([python_exe, '-u', start_script],
                                   cwd=runtime_dir, env=env)
        # [pyd PyUnicode_Concat(STR('进程已启动，PID: '),str(pid)) +0xa4d / py=42]
        print('进程已启动，PID: ' + str(process.pid))
        # [pyd PyObject_GetAttr(process,'wait') +0xa61 / py=44]
        return_code = process.wait()
        # [pyd PyUnicode_Concat(STR('进程已结束，返回代码: '),str(rc)) +0xa7c / py=45]
        print('进程已结束，返回代码: ' + str(return_code))
        # [pyd 尾部 return plVar21 / py=46]
        return return_code
    # [pyd LAB_18000xxxx 异常匹配 PyExc_Exception / py=47]
    except Exception as e:
        # [pyd PyUnicode_Concat(STR('启动过程中出错: '),str(e)) +0xab9 / py=48]
        print('启动过程中出错: ' + str(e))
        # [pyd PyObject_GetAttr(traceback,'print_exc') +0xaca / py=49]
        traceback.print_exc()
        # [pyd 常量 1 返回 / py=50;E1 probe_e1_launcher4 'popen_raise' → 1]
        return 1


# [pyd pymod_exec +0xcf1..+0xcf8:PyObject_SetAttr(module,'__name__','__main__')
#  + 'exit';py=53-55]
if __name__ == '__main__':
    sys.exit(main())
