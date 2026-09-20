# -*- coding: utf-8 -*-
"""streams —— ADS 清理编排(P003 试点重建,自 modules/core/streams.pyd)。

行为证据:M1.2a(名表 83 项 + 反编译,cleanup_task py_line 95-108)+
E1 oracle 差分修正:锁文件路径为 join(home_dir, "ads.lock")(home_dir
来自 modules.core.config,导入期按 cwd 解析),清理后向 stdout 打印
"清理完成/扫描文件总数: N/删除备用数据流数量: M"。
"""
import os
import threading
import time
import tkinter as tk

from modules.core.util import delete_ads_recursive

# E1 修正(R009 定谳 2026-09-10):五行横幅(client version/gpu trt/local ip/
# allow accounts/allow host)**由 modules.core.config 在首次导入时打印一次**;
# 真实 streams 链路经 util → config 触发,故本模块**不得**自行复述横幅
# (否则 stdout 出现两份,与 oracle 差分失配)。

home_dir = os.path.abspath("human_data")
if not os.path.isdir(home_dir):
    os.makedirs(home_dir)


def show_loading_dialog():
    """按 U003 oracle 实采复刻的加载窗(U007 定谳 u007-fix-w01-1)。

    oracle 第一手证据(ui/baseline/1920x1080/100/u002_startup_ads_loading/**
    + evidence/modules/core__streams/static/constants.txt 串表):
      * 顶层标题 = 「正在初始化」(U002 W01 修正项),350x150 逻辑、
        overrideredirect + topmost + toolwindow;
      * 深色背景 #2C2C2C,三圆点走马灯(dots/current_dot/dots_frame/
        animation_frame 均为串表名):直径 10、圆心 y=46、x=149/174/199(间距 25),
        活动 点 #434AF9,静止 #4A4A4A;
      * 文案序列 = 正在初始化...(扫描期) → 正在启动系统(扫描完成后) →
        初始化完成(约 200ms 自毁)/ 清理失败: <exc>;文本色 #E0E0E0;
      * 旧复刻(灰底 #E0E0E0 +「正在清理 ADS ...」)与 oracle 不符,废弃。
    """
    root = tk.Tk()
    root.title("正在初始化")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-toolwindow", True)
    except tk.TclError:
        pass
    root.geometry("350x150")
    root.configure(background="#2C2C2C")

    dots_frame = tk.Canvas(root, width=350, height=80, background="#2C2C2C",
                           highlightthickness=0)
    dots_frame.place(x=0, y=0)
    dots = [dots_frame.create_oval(x - 5, 41, x + 5, 51,
                                   fill="#4A4A4A", outline="")
            for x in (149, 174, 199)]
    current_dot = {"index": 0}

    def animation_frame():
        try:
            active = current_dot["index"] % len(dots)
            for i, item in enumerate(dots):
                dots_frame.itemconfigure(
                    item, fill="#434AF9" if i == active else "#4A4A4A")
            current_dot["index"] += 1
            root.after(500, animation_frame)
        except tk.TclError:
            pass

    root.after(500, animation_frame)

    label = tk.Label(root, text="正在初始化...",
                     background="#2C2C2C", foreground="#E0E0E0",
                     font=("Arial", 12, "bold"))
    label.place(x=0, y=78, width=350, height=22)
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - 350) // 2
    y = (screen_h - 150) // 2
    root.geometry("350x150+%d+%d" % (x, y))
    return root, label


def perform_ads_cleanup():
    ads_lockfile = os.path.join(home_dir, "ads.lock")
    if os.path.exists(ads_lockfile):
        return

    root, label = show_loading_dialog()

    def cleanup_task():
        try:
            current_dir = os.getcwd()
            # [U007 定谳 u007-fix-w01-1] 扫描期保持「正在初始化...」(U003 scan 轮
            # 全部帧皆此文案);扫描完成后 → 「正在启动系统」(U003 fail 轮
            # PrintWindow 采样);终态 = 「初始化完成」/「清理失败: <exc>」。
            # 旧复刻的「正在清理 ...」「清理完成」在 oracle 中不存在。
            delete_ads_recursive(current_dir)
            label.configure(text="正在启动系统")
            with open(ads_lockfile, "w") as handle:
                handle.close()
            label.configure(text="初始化完成")
        except Exception as exc:
            label.configure(text="清理失败: " + str(exc))
        finally:
            root.after(200, root.destroy)

    thread = threading.Thread(target=cleanup_task, daemon=True)
    thread.start()
    root.mainloop()
