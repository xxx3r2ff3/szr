# -*- coding: utf-8 -*-
"""R071 `heygem__wenet__utils__checkpoint` 候选实现(门控 T3 上游恢复件)。

源 pyd : modules/heygem/wenet/utils/checkpoint.pyd
         sha256 = 62793330a8b0a18a1601da99466e706f763984c30857cb4ab12ecdd910b9b3bc
上游锚定: wenet-e2e/wenet @ tag v1.0.0 (commit 7f00996e018839f0e2bb8835ef2452896d74ef1e)
         文件 wenet/utils/checkpoint.py
         sha256(upstream) = d34a8b57ddfcffeca831cd30abf382d8b91e196b7657a29ea27d9841f4249682
         许可证 = Apache-2.0(仓库 LICENSE;文件头为 Mobvoi 版权行)
         本文件与上游逐字符一致,仅追加本注释块(代码段未改)。

证据出处(逐项可回溯):
  [ST 0x…]  evidence/modules/heygem__wenet__utils__checkpoint/static/pe_scan.md
            §Cython 字面量池 —— 由 pyd 的 .rdata `__pyx_k_*` 池精确提取(58 条),
            覆盖本文件全部标识符/属性名/日志格式串(见同目录 pool_coverage.md)。
  [MP]      同目录 pe_scan.md §PyMethodDef —— 仅 load_checkpoint / save_checkpoint
            两条(METH_FASTCALL|METH_KEYWORDS = 0x82),与上游两个模块级 def 对齐。
  [DOC]     同目录 pe_scan.md —— save_checkpoint 的 ml_doc 指针内容 =
            "\\n    Args:\\n        infos (dict or None): any info you want to save.\\n    "
            与上游 save_checkpoint 的 '''…''' 文档串逐字节一致。
  [GATE]    evidence/imports/gated_dll_dependencies.json + probe_gated_dll.py ——
            pyd 导入表为 python38.dll,3.10 运行时不可装载(无 oracle 黄金)。
  [HIST]    wenet 仓库 wenet/utils/checkpoint.py 变更史:本内容不含
            OrderedDict/filter_modules/load_trained_modules(见 384f9356cc25,2022-01-23),
            亦不含 strict=（v2.0.0 起),且含 Loader=yaml.FullLoader(4beb5efd68b1 起)
            → 与 v1.0.0 内容一致;串池对该内容覆盖率 100%。
"""
import logging
import os
import re                                                     # [ST 0x26630 're']

import yaml                                                   # [ST 0x2674c 'yaml']
import torch                                                  # [ST 0x2670c 'torch']


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    # [ST 0x26644 'cuda' + 0x267d8 'is_available'] → torch.cuda.is_available()
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')     # [ST 0x26634 'cpu']
    model.load_state_dict(checkpoint)                        # [ST 0x26818]
    info_path = re.sub('.pt$', '.yaml', path)                # [ST 0x26628 '.pt$' / 0x266e4 '.yaml']
    configs = {}                                             # [ST 0x26758 'configs']
    if os.path.exists(info_path):                            # [ST 0x26724 'exists']
        with open(info_path, 'r') as fin:                    # [ST 0x26638 'fin' / 0x266f0 '__enter__']
            configs = yaml.load(fin, Loader=yaml.FullLoader)  # [ST 0x26788 'FullLoader']
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    # [DOC] 文档串逐字节一致;此处 logging.info 在 isinstance 链之前(v1.0.0 原序)
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):             # [ST 0x267b8]
        state_dict = model.module.state_dict()               # [ST 0x267a8 'state_dict']
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # [ST 0x26768 'parallel' + 0x26878 'DistributedDataParallel']
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)                             # [ST 0x266bc 'save']
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:                                        # [ST 0x266fc 'infos']
        infos = {}
    with open(info_path, 'w') as fout:                       # [ST 0x26674 'fout']
        data = yaml.dump(infos)                              # [ST 0x2665c 'dump' / 0x2664c 'data']
        fout.write(data)                                     # [ST 0x26714 'write']
