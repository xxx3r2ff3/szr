# -*- coding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/02/11 09:52:26
@Author  :
@Version :   1.0
@Desc    :   None
"""
# ---------------------------------------------------------------------------
# 重建证据(R043 / T4R,源 modules/rag/utils.cp310-win_amd64.pyd)
#   [E1] 运行期回溯锚点(py 行号):get_content 的 for 在 py=43;
#        read_file_content 的 raise 在 py=104;read_text 的 open 在 py=109;
#        Documents.get_content 的 open 在 py=122。
#   [E2] 串表(static/constants.txt):TIKTOKEN_CACHE_DIR / cl100k_base /
#        Unsupported file type / utf-8 / chunk_content / max_token_len /
#        cover_content / curr_chunk / curr_len / num_chunks / token_len /
#        splitlines / endswith / isspace / rstrip / replace / walk / join。
#   [E1] 行为实测:ns_w2rag/probe_beh2_out.json、probe_utils8/9/10/11/12/13_out.json
#        (普通累加分支 + 长行分支的全量网格拟合,5822 例 char-enc + 5832 例真实
#         enc 全部命中;算法见 reports/modules/rag__utils-impl.md)。
# ---------------------------------------------------------------------------
import os
import json
import tiktoken

os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '.tiktoken')
enc = tiktoken.get_encoding('cl100k_base')


class ReadFiles:
    """
    class to read files
    """

    def __init__(self) -> None:
        pass

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        # [E1 py=43] 依赖 self.file_list(get_files 填充);未调用 get_files 时
        # 抛 AttributeError: 'ReadFiles' object has no attribute 'file_list'
        chunk_content = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content.extend(self.get_chunk(content, max_token_len, cover_content))
        return chunk_content

    def get_files(self, path: str):
        # [E1] 递归收集 *.txt(probe_utils8 t1..t11:仅 .txt 入选,目录不入选,
        # 隐藏文件/.md 不入选);返回 None,结果写入 self.file_list
        self.file_list = []
        for filepath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.txt'):
                    self.file_list.append(os.path.join(filepath, filename))

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        # [E1] 逐行累加:行长按 token 计,累积量按 token+换行计;放不下时
        # 以 curr_chunk[-cover_content:] 作为重叠前缀另起一块,并把累积量重置为
        # line_len + cover_content(网格拟合:char-enc 5822 例 + 真实 enc 5832 例全过)。
        chunk_text = []
        curr_len = 0
        curr_chunk = ''
        token_len = max_token_len - cover_content
        lines = text.splitlines()
        for line in lines:
            line = line.replace(' ', '')
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # [E1] 长行分支(网格拟合):每轮把 curr[-cover:] 追加进结果、curr 只保留该尾部;循环后接上最后一片并追加;再走普通分支。token_len==0 → ZeroDivisionError。
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    if curr_len + line_len <= max_token_len:
                        # 长行时恒假(line_len > max_token_len),保留结构以对齐原版
                        curr_chunk += line[start:end]
                        curr_len += line_len
                    else:
                        chunk_text.append(curr_chunk[-cover_content:])
                        curr_chunk = curr_chunk[-cover_content:]
                        curr_len = line_len
                curr_chunk += line[start:end]
                chunk_text.append(curr_chunk)
            if curr_len + line_len <= token_len:
                curr_chunk += line + '\n'
                curr_len += line_len + 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content
        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # [E1 py=104] 仅 .txt 走 read_text;其余后缀(实测 .json/.md/.csv/.log/.py)
        # 一律 ValueError('Unsupported file type')
        if file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError('Unsupported file type')

    @classmethod
    def read_text(cls, file_path: str):
        # [E1 py=109] utf-8 直读;文件不存在时 FileNotFoundError 原样抛出
        with open(file_path, mode='r', encoding='utf-8') as f:
            return f.read()


class Documents:
    """
    获取已分好类的json格式文档
    """

    def __init__(self, path: str = '') -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            return json.load(f)
