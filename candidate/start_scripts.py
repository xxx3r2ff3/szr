# -*- coding: utf-8 -*-
# start_scripts —— 快捷脚本.exe 入口整合源码(U007-w03,S 系列交付物)。
# 注:原 start_scripts.pyc 模块级无 docstring,登记信息以注释承载,
#     以保证整合物 CPython 3.10 编译产物与原 pyc 逐码对象全同(见下)。
#
# 来源:start_scripts.pyc(Python 3.10,PyInstaller 解包自 快捷脚本.exe,
# szr2026_out/exe_extract/快捷脚本/start_scripts.pyc)。S005 口径:该入口为纯
# tkinter 工具窗(「脚本管理中心」1200x1000,scripts_dir=cwd/scripts),不含
# modules/ 业务 pyd 逻辑,反编译物即最终源码的高保真恢复物,在 R 系列外单独
# 交付(reports/S005_exe_entries.md §1/§2.4)。
#
# 整合口径(U007-w03,2026-09-13):
# * 底本 = pycdc 反编译产物 exe_extract/快捷脚本/_decomp/start_scripts.py;
#   逐函数对照 pycdas 反汇编 _decomp/start_scripts.pyc.dis 恢复 pycdc 丢失的
#   常量(事件名/sorted/按钮文本)与不完整函数体(_execute_thread),并还原
#   pycdc 的伪调用形态(**('kw',)/None(...)→正常关键字调用)。
# * 行为等价优先于美化:全部行为数值(尺寸/颜色/字体/主题/路径/subprocess 调用
#   形态)取反汇编原值,不做"改进"。伪影逐项处置清单见
#   reports/U007_window_rebuild_fixes.md §8。
# * 整合后以 CPython 3.10 编译并与原 pyc 逐码对象对比:co_code/co_consts/
#   co_names/co_varnames/co_flags 全同,仅行号表与源文件名不同——见报告 §8 复现。
#
# 行为参数(U002_preinventory §44 实测口径):标题「脚本管理中心」/1200x1000/
# minsize(1000,800)/bg #f4f7f9/ttk 主题 clam/scripts_dir=cwd/scripts。
# U003 补充:正常态 13 个 .bat 按钮+执行日志,scripts_app.exe(schtasks
# RunLevel=HIGHEST)。
#
# 遗留验证项:VM 被 I012 浸泡占用,窗口树对照(win32 530 节点/uia 35/tab=4,
# evidence/ui/U002/windows/win03_script_manager/capture.json)与四状态对照
# (ui/baseline/1920x1080/100/u002_script_manager/)顺延至 VM 空闲后,清单见
# reports/U007_window_rebuild_fixes.md §8 末。
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk
import subprocess
import threading
import sys


def get_resource_path(relative_path):
    '''获取资源绝对路径，兼容 PyInstaller'''
    # [pyc dis 70-94] pycdc 输出的 `None.path.join` 为伪影,反汇编两分支均为 os.path.join。
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


class ProBatLauncher:

    def __init__(self, root):
        self.root = root
        self.root.title('脚本管理中心')
        icon_path = get_resource_path('script.ico')
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        self.root.geometry('1200x1000')
        self.root.minsize(1000, 800)
        self.root.configure(bg='#f4f7f9')
        self.scripts_dir = os.path.join(os.getcwd(), 'scripts')
        self.setup_styles()
        self.setup_ui_structure()
        self.create_modern_buttons()

    def setup_styles(self):
        '''现代化样式配置'''
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Modern.TButton',
                             font=('Microsoft YaHei UI', 10),
                             background='#FFFFFF',
                             foreground='#2c3e50',
                             borderwidth=1,
                             relief='flat',
                             padding=(15, 12),
                             anchor='w')
        self.style.map('Modern.TButton',
                       background=[('pressed', '#ecf0f1'), ('active', '#e8f4fd')],
                       foreground=[('active', '#3498db')])

    def setup_ui_structure(self):
        '''UI 布局结构'''
        header = tk.Frame(self.root, bg='#434af9', height=60)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        tk.Label(header, text='脚本控制台', font=('Microsoft YaHei UI', 14, 'bold'),
                 fg='white', bg='#434af9').pack(pady=15)
        main_container = tk.Frame(self.root, bg='#f4f7f9')
        main_container.pack(fill='both', expand=True, padx=25, pady=20)
        self.left_frame = tk.LabelFrame(main_container, text=' 脚本库 ', bg='#f4f7f9',
                                        font=('微软雅黑', 10, 'bold'), width=320)
        self.left_frame.pack(side='left', fill='y', padx=(0, 20))
        self.left_frame.pack_propagate(False)
        self.canvas = tk.Canvas(self.left_frame, bg='#f4f7f9', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient='vertical',
                                       command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f4f7f9')
        # [pyc dis 910-920] 事件名 '<Configure>' 被 pycdc 丢失(显示为 None 实参),按反汇编恢复。
        self.scrollable_frame.bind('<Configure>',
                                   lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame,
                                                       anchor='nw', width=290)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        def _bind_mousewheel(event):
            self.canvas.bind_all('<MouseWheel>', _on_mousewheel)

        def _unbind_mousewheel(event):
            self.canvas.unbind_all('<MouseWheel>')

        self.left_frame.bind('<Enter>', _bind_mousewheel)
        self.left_frame.bind('<Leave>', _unbind_mousewheel)
        self.scrollable_frame.bind('<Enter>', _bind_mousewheel)
        self.canvas.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.scrollbar.pack(side='right', fill='y')
        right_frame = tk.LabelFrame(main_container, text=' 执行日志 ', bg='#f4f7f9',
                                    font=('微软雅黑', 10, 'bold'))
        right_frame.pack(side='right', fill='both', expand=True)
        log_tools = tk.Frame(right_frame, bg='#f4f7f9')
        log_tools.pack(fill='x', side='top', padx=10, pady=10)
        self.clear_btn = tk.Label(log_tools, text=' 清空日志 ', font=('Microsoft YaHei UI', 9),
                                  bg='#ffffff', fg='#666666', padx=15, pady=5, cursor='hand2',
                                  relief='flat', highlightthickness=1,
                                  highlightbackground='#cccccc')
        self.clear_btn.pack(side='right')
        # [pyc dis 1058-1090] 三个绑定的事件名 pycdc 全部丢失(None 调用),按反汇编恢复。
        self.clear_btn.bind('<Button-1>', lambda e: self.clear_log())
        self.clear_btn.bind('<Enter>',
                            lambda e: self.clear_btn.config(bg='#434af9', fg='white',
                                                            highlightbackground='#434af9'))
        self.clear_btn.bind('<Leave>',
                            lambda e: self.clear_btn.config(bg='#ffffff', fg='#666666',
                                                            highlightbackground='#cccccc'))
        self.log_display = scrolledtext.ScrolledText(right_frame, font=('Consolas', 11),
                                                     bg='#1e1e1e', fg='#dcdcdc',
                                                     state='disabled', borderwidth=0,
                                                     padx=15, pady=15)
        self.log_display.pack(fill='both', expand=True, padx=10, pady=(0, 15))

    def log(self, message):
        self.log_display.config(state='normal')
        self.log_display.insert(tk.END, message + '\n')
        self.log_display.see(tk.END)
        self.log_display.config(state='disabled')

    def clear_log(self):
        self.log_display.config(state='normal')
        # [pyc dis 1220] 原常量是浮点 1.0(pycdc 误显示为整数 1);str(1.0)='1.0'
        # 恰为 Tk 标准「行.列」索引,照原值保留。
        self.log_display.delete(1.0, tk.END)
        self.log_display.config(state='disabled')

    def run_bat(self, script_name):
        script_path = os.path.join(self.scripts_dir, script_name)
        thread = threading.Thread(target=self._execute_thread,
                                  args=(script_path, script_name))
        thread.daemon = True
        thread.start()

    def _execute_thread(self, script_path, script_name):
        # pycdc 对本函数标记 Decompyle incomplete;以下按 start_scripts.pyc.dis
        # [1299-1457] 逐指令恢复,行为面以反汇编为准。
        self.log(f'>>> RUNNING: {script_name}')
        try:
            if getattr(sys, 'frozen', False):
                exe_dir = os.path.dirname(sys.executable)
            else:
                exe_dir = os.path.dirname(os.path.abspath(__file__))
            # [pyc dis 1388-1403] list+shell=True 的原始调用形态照保留(Windows 下
            # list2cmdline 拼接后交 cmd /c),不做"规范化"。
            process = subprocess.Popen([script_path],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       shell=True,
                                       text=True,
                                       encoding='utf-8',
                                       errors='replace',
                                       cwd=exe_dir)
            for line in process.stdout:
                self.log(line.strip())
            process.wait()
            self.log('>>> COMPLETED\n')
        except Exception as e:
            self.log(f'[ERROR] {str(e)}')

    def create_modern_buttons(self):
        if not os.path.exists(self.scripts_dir):
            os.makedirs(self.scripts_dir)
            return None
        # [pyc dis 1585-1597] pycdc 丢失了外层 sorted();按钮按文件名排序排布
        # (与 U002 实测 13 个 .bat 的按钮顺序口径一致)。
        files = sorted([f for f in os.listdir(self.scripts_dir) if f.lower().endswith('.bat')])
        for file_name in files:
            display_name = os.path.splitext(file_name)[0]
            btn = ttk.Button(self.scrollable_frame,
                             text=f'  ▶  {display_name}',
                             style='Modern.TButton',
                             command=lambda f=file_name: self.run_bat(f),
                             cursor='hand2')
            btn.pack(fill='x', pady=6)


if __name__ == '__main__':
    # [pyc dis 1737-1755] bare except(pycdc 误显示为 finally: pass);DPI 感知
    # 失败时静默降级,与反汇编 handler 形态(POP_TOP x3 + POP_EXCEPT)一致。
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    root = tk.Tk()
    app = ProBatLauncher(root)
    root.mainloop()
