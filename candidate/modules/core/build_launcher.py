# -*- coding: utf-8 -*-
"""build_launcher —— R006(重建自 modules/core/build_launcher.cp310-win_amd64.pyd)。

源 pyd:modules/core/build_launcher.cp310-win_amd64.pyd
        sha256 3089cde8dbb0926ad63c2bdcfab90c0b75d49008c5e5b8c3511494fa67a506e3
反编译证据:evidence/modules/core__build_launcher/static/pseudocode/
  core__build_launcher__FUN_180013400__180013400.c  = __pyx_pymod_exec_build_launcher
  core__build_launcher__FUN_180011540__180011540.c  = main(0x180011540 / py 328-338)
  core__build_launcher__FUN_180001260__180001260.c  = ExeGenerator.__init__(py 10-22)
  core__build_launcher__FUN_180001c80__180001c80.c  = create_widgets(0x180001c80)
  core__build_launcher__FUN_180006e80/76c0/7f00/8e00/a9a0/cbe0/e9b0/ff20 = 其余方法
  core__build_launcher__FUN_1800126d0__1800126d0.c  = init_cached_constants
        (PyTuple_Pack 全部消息元组:('错误','请选择ICO图标文件') / '请选择Python文件'
         / '选择的Python文件不存在' / '请输入EXE程序名称'
         / '未找到pyinstaller，请先安装: pip install pyinstaller'
         / ('警告: 未找到pyinstaller，请先安装: pip install pyinstaller',)
         / ('PYTHON_EXECUTABLE',) / (False,False) / ('Arial',16,'bold')
         / ('ICO files','*.ico') / ('All files','*.*') / ('Python files','*.py'))
  _stringtab.json(230 槽位)

E1 证据(VM oracle 侧,probe_e1_surface / probe_e1_build + _out.json):
  * 模块命名空间 = ExeGenerator/main/filedialog/messagebox/os/python_exe/
    subprocess/sys/threading/tk/ttk(+dunder),无模块级 Tk root;
  * python_exe = os.getenv('PYTHON_EXECUTABLE') or sys.executable
    (probe:置 PYTHON_EXECUTABLE=C:\\fake\\python.exe → python_exe 取该值;
     未置时 = sys.executable);
  * 方法签名/文档字符串逐条实测(py=函数 def 行,co_varnames 全量);
  * main().co_varnames=(root,app)、co_firstlineno=328;
  * oracle runtime 内 pyinstaller 6.17.0 可用
    (`python -m PyInstaller --version` → rc=0)——因此 main() 的
    可用性检查在 VM 内恒走成功分支。

★ GUI:本模块是 Tkinter 工具,main() 会 mainloop 常驻;正式用例只验证
  "GUI 前置行为"(pyinstaller 可用性检查 + tk.Tk() 入口),用 setattr 把
  tk 置 None 使入口立即失败,不拉起真实窗口(见 cases/)。

覆盖单元:core__build_launcher::main::B0。
"""
import os
import sys
import subprocess  # noqa: F401  (E1 名表:模块级 subprocess)
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox  # noqa: F401  (E1 名表)

# [pyd pymod_exec py=8:os.getenv(*('PYTHON_EXECUTABLE',)) 为真则用之,否则
#  sys.executable;缓存常量 DAT_180020220 = PyTuple_Pack(1,'PYTHON_EXECUTABLE')]
python_exe = os.getenv('PYTHON_EXECUTABLE') or sys.executable


class ExeGenerator:
    # [pyd 0x180001260 py=10-22;co_varnames=(self,root)]
    def __init__(self, root):
        self.root = root
        self.root.title('EXE生成器')  # [py=13]
        self.root.geometry('500x700')  # [py=14]
        self.root.resizable(False, False)  # [py=15,常量 (False,False)]
        self.create_widgets()  # [py=16]
        self.ico_path = None  # [py=18]
        # [py=22] os.path.join(os.getcwd(), 'start_launcher.py')
        self.py_path = os.path.join(os.getcwd(), 'start_launcher.py')

    # [pyd 0x180001c80 py=24-119;co_varnames=(self,main_frame,title_label,
    #  ico_frame,ico_select_frame,name_frame,options_frame,button_frame)]
    def create_widgets(self):
        """创建界面组件"""
        # [py=27-28] ttk.Frame(self.root, padding=20) + pack(fill=BOTH, expand=True)
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # [py=31-34] 标题 Label('EXE文件生成器', font=('Arial',16,'bold')) + pack(pady=10)
        self.title_label = ttk.Label(self.main_frame, text='EXE文件生成器',
                                     font=('Arial', 16, 'bold'))
        # [py=31-34] 标题 Label('EXE文件生成器', font=('Arial',16,'bold')) + pack
        # [U007 定谳 u007-fix-w02-1] pack 几何以 U003 oracle 控件树实测为准
        # (ui/baseline/1920x1080/100/u002_exe_generator/normal/capture_win32.json):
        # 段间距(逻辑 px)= title→ico 30 / ico→name 20 / name→options 20 /
        # options→button 30 / button→labelframe 30,即各段 pady=(0,N);
        # 候选旧 pady(10/5/5/5/5)导致 132 节点全树纵向漂移 -130..+20。
        self.title_label.pack(pady=(0, 30))
        # [py=37-40] ICO 区
        self.ico_frame = ttk.Frame(self.main_frame)
        self.ico_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(self.ico_frame, text='选择ICO图标:').pack(anchor=tk.W)
        # [py=42-49] 输入框 + 浏览按钮
        # [U007 定谳 u007-fix-w02-1] oracle ico_entry = padx=(0,5)(左缘与段框
        # 对齐 x+0、宽 368;旧 padx=5 使输入框 363 且右移 5,与 oracle 不符)。
        self.ico_select_frame = ttk.Frame(self.ico_frame)
        self.ico_select_frame.pack(fill=tk.X, pady=5)
        self.ico_entry = ttk.Entry(self.ico_select_frame)
        self.ico_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(self.ico_select_frame, text='浏览',
                   command=self.select_ico_file).pack(side=tk.RIGHT)
        # [py=69-76] EXE 名称区(默认名 '启动',常量 DAT_180020010=(0,'启动'))
        self.name_frame = ttk.Frame(self.main_frame)
        self.name_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(self.name_frame, text='EXE程序名称:').pack(anchor=tk.W)
        self.name_entry = ttk.Entry(self.name_frame)
        self.name_entry.pack(fill=tk.X, pady=5)
        self.name_entry.insert(0, '启动')
        # [py=79-90] 选项区(两个 Checkbutton,初始 state=disabled)
        self.options_frame = ttk.Frame(self.main_frame)
        self.options_frame.pack(fill=tk.X, pady=(0, 30))
        self.console_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text='无控制台窗口 (--noconsole)',
                        variable=self.console_var,
                        state='disabled').pack(anchor=tk.W)
        self.admin_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text='请求管理员权限 (--uac-admin)',
                        variable=self.admin_var,
                        state='disabled').pack(anchor=tk.W)
        # [py=93-98] 生成按钮(width=20)
        # [U007 定谳 u007-fix-w02-1] button_frame pady=(0,30)(同上,oracle 树实测)。
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 30))
        self.generate_btn = ttk.Button(self.button_frame, text='生成EXE文件',
                                       command=self.generate_exe, width=20)
        self.generate_btn.pack()
        # [py=103] 进度条
        self.progress = ttk.Progressbar(self.main_frame, mode='indeterminate')
        # [py=106] 状态标签
        self.status_label = ttk.Label(self.main_frame, text='', foreground='blue')
        # [py=109-119] 输出位置区
        self.exe_location_frame = ttk.LabelFrame(self.main_frame,
                                                 text='生成的EXE文件位置')
        # [U007 定谳 u007-fix-w02-1] oracle 树实测 labelframe 顶距 button 段 30,
        # 即本段 pady top=0(pady=(0,5));旧 pady=5 使 labelframe@368,oracle@363。
        self.exe_location_frame.pack(fill=tk.X, pady=(0, 5))
        self.exe_location_label = ttk.Label(self.exe_location_frame,
                                            text='尚未生成EXE文件',
                                            wraplength=450, foreground='gray',
                                            justify=tk.LEFT)
        self.exe_location_label.pack(fill=tk.X, padx=5, pady=5)

    # [pyd 0x180006e80 py=121-130;co_varnames=(self,file_path)]
    def select_ico_file(self):
        """选择ICO文件"""
        # [py=123-125] filedialog.askopenfilename(title=..., filetypes=[...])
        file_path = filedialog.askopenfilename(
            title='选择ICO图标文件',
            filetypes=[('ICO files', '*.ico'), ('All files', '*.*')])
        if file_path:  # [py=126]
            self.ico_path = file_path  # [py=127]
            self.ico_entry.delete(0, tk.END)  # [py=128]
            self.ico_entry.insert(0, file_path)  # [py=129]

    # [pyd 0x1800076c0 py=132-141;co_varnames=(self,file_path)]
    def select_py_file(self):
        """选择Python文件"""
        file_path = filedialog.askopenfilename(
            title='选择Python文件',
            filetypes=[('Python files', '*.py'), ('All files', '*.*')])
        if file_path:
            self.py_path = file_path
            self.py_entry.delete(0, tk.END)
            self.py_entry.insert(0, file_path)

    # [pyd 0x180007f00 py=143-162;co_varnames=(self,name)]
    # 四条校验各以 messagebox.showerror(('错误', <文本>)) 报错并 return False
    def validate_inputs(self):
        """验证输入"""
        if not self.ico_path:  # [py=145]
            messagebox.showerror('错误', '请选择ICO图标文件')  # [py=146]
            return False
        if not self.py_path:  # [py=149]
            messagebox.showerror('错误', '请选择Python文件')  # [py=150]
            return False
        if not os.path.exists(self.py_path):  # [py=153]
            messagebox.showerror('错误', '选择的Python文件不存在')  # [py=154]
            return False
        name = self.name_entry.get().strip()  # [py=157]
        if not name:  # [py=158]
            messagebox.showerror('错误', '请输入EXE程序名称')  # [py=159]
            return False
        return True  # [py=161]

    # [pyd 0x180008e00 py=164-186;co_varnames=(self,exe_name,possible_dirs,
    #  extensions,directory,ext,exe_path)]
    def find_exe_file(self, exe_name):
        """查找生成的EXE文件"""
        # [py=167-171] 候选目录: cwd/dist、脚本所在目录/dist、cwd、脚本所在目录
        possible_dirs = [
            os.path.join(os.getcwd(), 'dist'),
            os.path.join(os.path.dirname(self.py_path), 'dist'),
            os.getcwd(),
            os.path.dirname(self.py_path),
        ]
        # [py=173] 候选后缀:.exe 与无后缀
        extensions = ['.exe', '']
        for directory in possible_dirs:  # [py=174]
            for ext in extensions:  # [py=175]
                exe_path = os.path.join(directory, exe_name + ext)  # [py=181-182]
                if os.path.exists(exe_path):  # [py=183]
                    return os.path.abspath(exe_path)  # [py=184]
        return None  # [py=186]

    # [pyd 0x18000a9a0 py=188-245;co_varnames=(self,cmd,name,result,exe_path,
    #  e,error_msg)]
    def run_pyinstaller(self):
        """在单独线程中运行pyinstaller命令"""
        try:  # [py=190]
            # [py=192] PyList_New(8) = [python_exe,'-m','PyInstaller','-F',
            #   '--clean','--noconfirm','--exclude-module','core.launcher'] + '--icon'
            cmd = [python_exe, '-m', 'PyInstaller', '-F', '--clean', '--noconfirm',
                   '--exclude-module', 'core.launcher', '--icon']
            if self.ico_path:  # [py=194]
                cmd.append(self.ico_path)  # [py=195]
            name = self.name_entry.get().strip()  # [py=198]
            cmd.append('--name')  # [py=199]
            cmd.append(name)  # [py=200]
            if self.admin_var.get():  # [py=202]
                cmd.append('--uac-admin')  # [py=203]
            if self.console_var.get():  # [py=206]
                cmd.append('--noconsole')  # [py=207]
            cmd.append(self.py_path)  # [py=208]
            # [py=216] subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            exe_path = self.find_exe_file(name)  # [py=217]
            if exe_path:  # [py=220]
                self.root.after(0, lambda: self.generation_complete(
                    True, 'EXE文件生成成功: ' + exe_path, exe_path))  # [py=222-223]
            else:
                self.root.after(0, lambda: self.generation_complete(
                    True, 'EXE文件生成成功，但无法定位文件位置', None))  # [py=231-233]
        except subprocess.CalledProcessError as e:  # [py=239]
            self.root.after(0, lambda: self.generation_complete(
                False, '发生错误: ' + (e.stderr or str(e)), None))  # [py=240-241]
        except Exception as e:  # [py=242]
            self.root.after(0, lambda: self.generation_complete(
                False, '发生错误: ' + str(e), None))  # [py=243-244]

    # [pyd 0x18000cbe0 py=246-277;co_varnames=(self,thread)]
    def generate_exe(self):
        """生成EXE文件"""
        # [py=248] 先校验 pyinstaller 可用性(与 main 同一命令行)
        try:
            subprocess.run([python_exe, '-m', 'PyInstaller', '--version'],
                           capture_output=True, check=True)  # [py=253-255]
        except (subprocess.CalledProcessError, FileNotFoundError):  # [py=258]
            messagebox.showerror('错误', '未找到pyinstaller，请先安装: pip install pyinstaller')
            return  # [py=259-261]
        if not self.validate_inputs():  # [py=263]
            return
        self.generate_btn.config(state='disabled')  # [py=265]
        self.progress.pack(fill=tk.X, pady=10)  # [py=266]
        self.progress.start()  # [py=267]
        self.status_label.pack(pady=5)  # [py=268]
        self.status_label.config(text='正在生成EXE文件，请稍候...', foreground='blue')  # [py=269]
        self.exe_location_label.config(text='正在生成...', foreground='blue')  # [py=272]
        thread = threading.Thread(target=self.run_pyinstaller)  # [py=275]
        thread.daemon = True  # [py=275]
        thread.start()  # [py=276]

    # [pyd 0x18000e9b0 py=279-310;co_varnames=(self,success,message,exe_path)]
    def generation_complete(self, success, message, exe_path):
        """生成完成回调"""
        self.progress.stop()  # [py=281]
        self.progress.pack_forget()  # [py=282]
        self.generate_btn.config(state='normal')  # [py=283]
        if success:  # [py=284]
            self.status_label.config(text='生成完成！', foreground='green')  # [py=286]
            if exe_path:  # [py=289]
                self.exe_location_label.config(text='文件位置: ' + exe_path,
                                               foreground='green', cursor='hand2')  # [py=290-291]
                # [py=296-297] 点击标签打开目录
                self.exe_location_label.bind(
                    '<Button-1>', lambda event: self.open_exe_location(exe_path))
            else:
                self.exe_location_label.config(
                    text='无法定位生成的EXE文件，请检查dist目录',
                    foreground='orange')  # [py=301]
                messagebox.showinfo('成功', message)  # [py=303-304]
        else:
            self.status_label.config(text='生成失败', foreground='red')  # [py=306]
            self.exe_location_label.config(text='生成失败，无EXE文件生成',
                                           foreground='red')  # [py=307-308]
            messagebox.showerror('错误', message)  # [py=310]

    # [pyd 0x18000ff20 py=312-326;co_varnames=(self,exe_path,directory,e)]
    def open_exe_location(self, exe_path):
        """打开EXE文件所在目录"""
        try:  # [py=314]
            if os.name == 'nt':  # [py=316]
                os.system('explorer /select,"%s"' % exe_path)  # [py=317/323]
            elif sys.platform == 'darwin':  # [py=320]
                os.system('open -R "%s"' % exe_path)  # [py=321]
            else:
                directory = os.path.dirname(exe_path)  # [py=319]
                os.system('xdg-open "%s"' % directory)  # [py=323]
        except Exception as e:  # [py=324]
            messagebox.showerror('错误', '无法打开文件位置: ' + str(e))


# [pyd 0x180011540 py=328-338;co_varnames=(root,app)、co_firstlineno=328]
def main():
    # [py=331-333] PyInstaller 可用性检查(与 generate_exe 同一命令行)
    try:
        subprocess.run([python_exe, '-m', 'PyInstaller', '--version'],
                       capture_output=True, check=True)
    # [py=334] except (subprocess.CalledProcessError, FileNotFoundError)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # [py=335] print 是内建(缓存常量 DAT_1800202a8 = builtins.print,
        #          参数为 1 元组 ('警告: 未找到pyinstaller，请先安装: pip install pyinstaller',))
        print('警告: 未找到pyinstaller，请先安装: pip install pyinstaller')
        return  # [py=336]
    root = tk.Tk()  # [py=337]
    app = ExeGenerator(root)  # [py=338]
    root.mainloop()  # [py=339]


# [pyd pymod_exec py=340-341:if __name__ == '__main__': main();__test__ = {}]
if __name__ == '__main__':
    main()

__test__ = {}
