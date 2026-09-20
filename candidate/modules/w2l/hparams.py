# -*- coding: utf-8 -*-
"""w2l.hparams —— R018(重建自 modules/w2l/hparams.cp310-win_amd64.pyd)。

源 pyd:modules/w2l/hparams.cp310-win_amd64.pyd
        sha256 49d03f452993d723b937ddad06871b0d28867aae1c9eecbd29e6c79bbc02b444
反编译证据:evidence/modules/w2l__hparams/static/pseudocode/
  w2l__hparams__FUN_180001cf0__180001cf0.c(= hparams_debug_string,0x180001cf0)
  w2l__hparams__FUN_180004240__180004240.c(= 推导式过滤 `name != 'sentences'`,
                                            PyObject_RichCompare op=3 即 !=)
  _stringtab.json(全 60 槽位;含 "'HParams' object has no attribute %s"、
  '  '、': '、'Hyperparameters:\\n'、'\\n'、'sentences')

E1 证据(VM oracle 侧 probe_e1_hparams.py / probe_e1_surface.py + _out.json):
  * 模块命名空间 = {HParams, hparams, hparams_debug_string}(11 符号含 dunder);
  * hparams.data 的键序即源码实参顺序(19 项默认值,E1 逐项 repr 采样);
  * HParams(**kwargs) 只写 self.data;未知属性 → AttributeError
    "'HParams' object has no attribute <key>"(`__getattr__` 内 raise,py=10);
  * set_hparam(key,value) 返回 None,覆盖写;缺参 → TypeError
    "set_hparam() takes exactly 3 positional arguments (2 given)"
    (Cython cyfunction 语系,纯 Python 候选无法逐字复刻,见报告"未定谳项");
  * raw setattr(obj,'x',v) 直接进实例 __dict__(不写 data);
  * hparams_debug_string() 在本构建必然抛 AttributeError(见下)。

覆盖单元:w2l__hparams::hparams_debug_string::B0。

版本/缺口登记:hparams_debug_string 取 `hparams.values()` 而 HParams 并无
values 方法(py=65 实测 AttributeError)→ 该函数在本构建是死代码/坏代码,
不得"顺手修好"。
"""


class HParams:  # [pyd py=1]
    def __init__(self, **kwargs):  # [pyd py=2] co_flags=0x4B(VARARGS)
        self.data = {}  # [pyd py=3]
        for key, value in kwargs.items():  # [pyd py=4] varnames key/value
            self.data[key] = value  # [pyd py=5]

    def __getattr__(self, key):  # [pyd py=8]
        if key not in self.data:  # [pyd py=9]
            # [pyd py=10 + ST 0x18000b6c0] 消息逐字来自串表槽位
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]  # [pyd py=11]

    def set_hparam(self, key, value):  # [pyd py=13]
        self.data[key] = value  # [pyd py=14]


# [pyd pymod_exec:19 个关键字实参按源码顺序 PyDict/关键字调用;
#  E1 实测 hparams.data 键序 = num_mels…fmax(见 probe_e1_hparams_out.json)]
hparams = HParams(
    num_mels=80,
    rescale=True,
    rescaling_max=0.9,
    use_lws=False,
    n_fft=800,
    hop_size=200,
    win_size=800,
    sample_rate=16000,
    frame_shift_ms=None,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,
    preemphasize=True,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,
)


def hparams_debug_string():  # [pyd 0x180001cf0 / co_firstlineno=64]
    # [pyd py=65] getattr(hparams,'values') → __getattr__ 抛 AttributeError
    values = hparams.values()
    # [pyd +0x86e..+0x8a5] 推导式:PyTuple_New(4)=('  ',str(name),': ',
    # str(values[name])),过滤 name != 'sentences'(FUN_180004240
    # 内 PyObject_RichCompare(...,3)),PyList_Append 入 hp
    hp = ['  ' + str(name) + ': ' + str(values[name])
          for name in sorted(values) if name != 'sentences']
    # [pyd +0x8a5/+0x8a7] PyUnicode_Join('\n', hp) → PyUnicode_Concat('Hyperparameters:\n', ...)
    return 'Hyperparameters:\n' + '\n'.join(hp)
