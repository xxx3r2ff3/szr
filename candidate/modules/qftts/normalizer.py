# -*- coding: utf-8 -*-
"""normalizer —— R048(T3:GPT-SoVITS 谱系中文/英文文本正则化)。

源件:modules/qftts/normalizer.cp310-win_amd64.pyd
      sha256 46eb71a96f6a3c10...(见 contracts/modules/qftts__normalizer.json)
对账材料:
  * 上游 GPT-SoVITS text/chinese2.py 谱系(ChineseTextNormalizer /
    EnglishTextNormalizer / TextNormalizer(ABC));
  * E1 探针 probe_e1/e2/e4/e6/e7(VM oracle 实测)。

二进制/E1 出处标注:
  [E1-e2] 模块名表 {ABC, ChineseTextNormalizer, EnglishTextNormalizer,
      TextNormalizer, abstractmethod, cn2an, inflect, re};
      TextNormalizer(ABC) 抽象方法 = {normalize},实例化 → TypeError
      ("Can't instantiate abstract class TextNormalizer with abstract method
      normalize");ChineseTextNormalizer / EnglishTextNormalizer 均无抽象方法。
  [E1-e4] ChineseTextNormalizer 不做任何符号替换:对 "。、，：；！？…％＃＆＠＄＾｀ＭＰ
      嗯 呣" 逐字符实测**原样返回**(上游 chinese2.py 的 REPLACE/REPLACE_SYMBOL
      两张表在本版本被移除,属本地改动)。类实例 vars() 为空(无 attributes 属性)。
  [E1-e6/e7] ChineseTextNormalizer.normalize(text) 与
      `cn2an.transform(text, "an2cn")` **全等**:
        * 41 个定向样本 + 400 条随机语料(字母表含全角数字/标点/中英混排)
          共 441 例,oracle 与 cn2an.transform(text,"an2cn") 逐例一致(mismatch=0);
        * 反例排除:"逐字符累积数字再 an2cn" 形态在年份样本上失败
          ("2024年" 该形态给 "二千零二十四年",oracle 给 "二零二四年");
          "an2cn(mode=smart)" 在 38/41 上失败;唯一 41/41 命中的是整串 transform。
        * 非字符串输入 → TypeError("Argument 'text' has incorrect type
          (expected str, got int)");None → TypeError("expected string or
          bytes-like object")。
  [E1-e4/e5] 依赖面:模块级 `import cn2an`(实测 cn2an 版本 0.5.23)、
      `import inflect`、`import re`、`from abc import ABC, abstractmethod`。
  [E1-e5] EnglishTextNormalizer 为 ESPnet/tacotron_cleaner 谱系:
      _abbreviations 20 条(\bmrs\b→misess … \bbtw\b→by the way,均 re.IGNORECASE)、
      _inflect 为 inflect.engine;normalize_numbers("123") 实测
      " one hundred twenty-three ";expand_abbreviations("Mr. Smith") → "mister. Smith";
      normalize("1,234") → " twelve thirty-four "、normalize("$5") → "  five  dollars "。
"""
import re
from abc import ABC, abstractmethod

import cn2an
import inflect


class TextNormalizer(ABC):
    """Abstract base class for text normalization, defining common interface."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize the input text."""


class ChineseTextNormalizer(TextNormalizer):
    """
    A class to handle preprocessing of Chinese text including normalization.
    """

    # normalizer 与 cn2an 组合的替代实现,单测在 CJK 与标点上应与 oracle 一致。
    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(
                "Argument 'text' has incorrect type (expected str, got %s)"
                % type(text).__name__
            )
        # [E1-e6/e7] 与 cn2an.transform(text, "an2cn") 全等(441 例实测)。
        return cn2an.transform(text, "an2cn")


class EnglishTextNormalizer(TextNormalizer):
    """
    A class to handle preprocessing of English text including normalization. Following:
    https://github.com/espnet/espnet_tts_frontend/blob/master/tacotron_cleaner/cleaners.py
    """

    def __init__(self):
        # [E1-e5] 缩写表逐条实测(20 条,顺序与 oracle vars() 一致)。
        # _whitespace_re 仍按 oracle vars() 建实例属性(符号面一致),但不参与 normalize。
        self._abbreviations = [
            (re.compile(r"\bmrs\b", re.IGNORECASE), "misess"),
            (re.compile(r"\bmr\b", re.IGNORECASE), "mister"),
            (re.compile(r"\bdr\b", re.IGNORECASE), "doctor"),
            (re.compile(r"\bst\b", re.IGNORECASE), "saint"),
            (re.compile(r"\bco\b", re.IGNORECASE), "company"),
            (re.compile(r"\bjr\b", re.IGNORECASE), "junior"),
            (re.compile(r"\bmaj\b", re.IGNORECASE), "major"),
            (re.compile(r"\bgen\b", re.IGNORECASE), "general"),
            (re.compile(r"\bdrs\b", re.IGNORECASE), "doctors"),
            (re.compile(r"\brev\b", re.IGNORECASE), "reverend"),
            (re.compile(r"\blt\b", re.IGNORECASE), "lieutenant"),
            (re.compile(r"\bhon\b", re.IGNORECASE), "honorable"),
            (re.compile(r"\bsgt\b", re.IGNORECASE), "sergeant"),
            (re.compile(r"\bcapt\b", re.IGNORECASE), "captain"),
            (re.compile(r"\besq\b", re.IGNORECASE), "esquire"),
            (re.compile(r"\bltd\b", re.IGNORECASE), "limited"),
            (re.compile(r"\bcol\b", re.IGNORECASE), "colonel"),
            (re.compile(r"\bft\b", re.IGNORECASE), "fort"),
            (re.compile(r"\betc\b", re.IGNORECASE), "et cetera"),
            (re.compile(r"\bbtw\b", re.IGNORECASE), "by the way"),
        ]

        _comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
        _decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
        _pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
        _dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
        _ordinal_re = re.compile(r"([0-9]+)(st|nd|rd|th)")
        _number_re = re.compile(r"[0-9]+")
        _fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
        _percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
        _whitespace_re = re.compile(r"\s+")

        # [E1-e4/e5] 正则与 oracle vars() 实测逐条一致。
        self._comma_number_re = _comma_number_re
        self._decimal_number_re = _decimal_number_re
        self._pounds_re = _pounds_re
        self._dollars_re = _dollars_re
        self._ordinal_re = _ordinal_re
        self._number_re = _number_re
        self._fraction_re = _fraction_re
        self._percent_number_re = _percent_number_re
        self._whitespace_re = _whitespace_re

        self._inflect = inflect.engine()

    def _remove_commas(self, m):
        return m.group(1).replace(",", "")

    def _expand_decimal_point(self, m):
        return m.group(1).replace(".", " point ")

    def _expand_dollars(self, m):
        # [E1-probe_norm] oracle 实测 normalize("$5") == "  five  dollars "
        # (金额前后各两个空格)→ 本地把整个匹配串统一替换为
        # " <数字> dollars ",再由 _number_re 展开数字,得到双侧双空格。
        match = m.group(0).replace("$", "")
        parts = match.split(".")
        if len(parts) > 2:
            return match + " dollars"  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return " %s %s, %s %s " % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return " %s %s " % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return " %s %s " % (cents, cent_unit)
        else:
            return " zero dollars "

    def _expand_fraction(self, m):
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        return " %s " % self.fraction_to_words(numerator, denominator)

    def fraction_to_words(self, numerator, denominator):
        # [E1-e5] 三参数签名(num/den 为位置参数),与 oracle 实测一致。
        if numerator == 1 and denominator == 2:
            return "one half"
        if numerator == 1 and denominator == 3:
            return "one third"
        if numerator == 1 and denominator == 4:
            return "one quarter"
        if numerator == 3 and denominator == 4:
            return "three quarters"
        if numerator == 1:
            return "one " + self._inflect.ordinal(self._number_to_words(denominator))
        return "%s %ss" % (self._number_to_words(numerator),
                           self._inflect.ordinal(self._number_to_words(denominator)))

    def _number_to_words(self, value):
        return self._inflect.number_to_words(value)

    def _expand_ordinal(self, m):
        return " " + self._inflect.number_to_words(m.group(0)) + " "

    def _expand_number(self, m):
        # [E1-probe_num4] 本地形态(0..200000 全量 12190 例与 oracle 逐字一致):
        #   1000 < n < 3000 → 年份式两位分组,且**结果两端各带一个空格**:
        #       n == 2000            → " two thousand "
        #       2000 < n < 2010      → " two thousand " + words(n % 100) + " "
        #       n % 100 == 0         → " " + words(n // 100) + " hundred "
        #       其它                 → " " + inflect(number_to_words, andword="",
        #                                     zero="oh", group=2).replace(", ", " ") + " "
        #   其余 → " " + inflect(number_to_words, andword="") + " "
        # 与上游 chinese2.py 的差异仅在空格包裹(上游年份分支不带空格)。
        num = int(m.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return " two thousand "
            elif 2000 < num < 2010:
                return " two thousand " + self._inflect.number_to_words(num % 100) + " "
            elif num % 100 == 0:
                return " " + self._inflect.number_to_words(num // 100) + " hundred "
            else:
                return " " + self._inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                ).replace(", ", " ") + " "
        else:
            return " " + self._inflect.number_to_words(num, andword="") + " "

    def _expand_percent(self, m):
        return self._expand_number(re.match(r"[0-9]+", m.group(1)))

    def normalize_numbers(self, text):
        # [E1-e4] normalize_numbers("123") 实测 " one hundred twenty-three "。
        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r"\1 pounds", text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._fraction_re, self._expand_fraction, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self._percent_number_re, self._expand_percent, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def expand_abbreviations(self, text):
        # [E1-e4] expand_abbreviations("Mr. Smith") 实测 "mister. Smith"。
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def normalize(self, text: str) -> str:
        # [E1-e4/e5] 本地改动:与上游 chinese2.py 的差异在于**不做空白折叠**——
        # oracle 实测 normalize("No. 123 (ABC)") == "No.  one hundred twenty-three  (ABC)"
        # (保留数字展开产生的双空格),normalize("1,234") == " twelve thirty-four "
        # (保留首尾空格);若末尾做 re.sub(_whitespace_re, " ") 这两项都会变成单空格,
        # 故本地 normalize = expand_abbreviations + normalize_numbers。
        text = self.expand_abbreviations(text)
        text = self.normalize_numbers(text)
        return text
