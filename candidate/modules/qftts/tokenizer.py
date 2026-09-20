# -*- coding: utf-8 -*-
"""tokenizer —— R052(T3:CosyVoice 谱系文本前端 tokenizer)。

源件:modules/qftts/tokenizer.cp310-win_amd64.pyd
      sha256 fbd49817a7b605fa3f0407c4a800dd731c7ced213c1b14d7c5995401efab93d7
对账材料:
  * 上游 CosyVoice `cosyvoice/tokenizer/tokenizer.py` 谱系(Tokenizer ABC /
    SimpleTokenizer / EspeakTokenizer / EmiliaTokenizer / DialogTokenizer /
    LibriTTSTokenizer / add_tokens);本版本为本地改造版(新增 normalizer 兄弟模块
    依赖、[S1]/[S2] 对话token),故**行为判据全部取 E1 实测**;
  * E1 探针 probe_w5_1..probe_w5_7(VM oracle 实测,共享目录 ns_w5tok/),
    关键结论逐条标注在下方代码注释 [E1-*]。

覆盖口径:契约 10 个模块级函数各 1 个基本块(见
contracts/modules/qftts__tokenizer.json);本文件同时按 E1 实测还原 10 个类的方法面。

二进制/E1 出处标注摘要:
  [E1-1] `dir(module)` 24 个公开名;类 __module__ 均为 "tokenizer";
         ChineseTextNormalizer/EnglishTextNormalizer 来自兄弟模块 `normalizer`
         (probe_w5_1_out.json,probe_w5_2_out.json)。
  [E1-2] 签名(inspect.signature 实测)与 co_varnames/co_firstlineno(probe_w5_1_out.json)。
  [E1-3] token 词表文件格式 `{token}\\t{id}`;缺 '_' → KeyError;DialogTokenizer
         另需 '[S1]'/'[S2]'(probe_w5_3_out.json 12 种变体)。
  [E1-4] 行为矩阵与异常路径(probe_w5_4/5/6/7_out.json)。
"""
import logging
import re
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional

import jieba
from lhotse import CutSet
from piper_phonemize import phonemize_espeak
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from normalizer import ChineseTextNormalizer, EnglishTextNormalizer

# [E1-1] 字符串表含 jieba/default_logger/setLevel/INFO ⇒ 导入期收敛 jieba 日志级别
jieba.default_logger.setLevel(logging.INFO)


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    # [E1-1] 实测 __abstractmethods__ 有 3 项(含 texts_to_token_ids),
    #        故基类同名方法也带 @abstractmethod
    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of token id sequences."""
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of token sequences."""
        ...

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        """Convert list of token sequences to list of token id sequences."""
        ...


class SimpleTokenizer(Tokenizer):
    """The simplpest tokenizer, treat every character as a token,
    without text normalization.
    """

    def __init__(self, token_file: Optional[str] = None):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        if token_file is not None:
            # [E1-3] 逐行 '{token}\t{id}';空行/非 \t 分隔 → IndexError;
            #        词表缺 '_' → KeyError('_')(pad_id 取 '_')
            self.token2id = {}
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    info = line.split('\t')
                    token, id = info[0], int(info[1])
                    self.token2id[token] = id
            self.vocab_size = len(self.token2id)
            self.pad_id = self.token2id['_']
            self.has_tokens = True
        else:
            # [E1-4] 实测为 logging.debug(不是 print),文案含源码续行缩进
            logging.debug('Initialize Tokenizer without tokens file, '
                          '                will fail when map to ids.')
            self.has_tokens = False

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        # [E1-9] 实测无前置断言:None/非 list 入参先撞 texts_to_tokens 的
        #        TypeError,len 才轮到 tokens_to_token_ids 里的断言
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        # [E1-4] 用 len(texts)/texts[i]:None → TypeError("no len()");str 入参按逐字符
        tokens_list = []
        for i in range(len(texts)):
            tokens_list.append(list(texts[i]))
        return tokens_list

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        assert self.has_tokens, 'Please initialize Tokenizer with a tokens file.'
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    # [E1-4] OOV 跳过(不抛异常)
                    logging.debug(f'Skip OOV {t}')
            token_ids_list.append(token_ids)
        return token_ids_list


class EspeakTokenizer(Tokenizer):
    """A simple tokenizer with Espeak g2p function."""

    def __init__(self, token_file: Optional[str] = None, lang: str = 'en-us'):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
          lang: the language identifier, see
            https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
        """
        self.lang = lang
        if token_file is not None:
            self.token2id = {}
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    info = line.split('\t')
                    token, id = info[0], int(info[1])
                    self.token2id[token] = id
            self.vocab_size = len(self.token2id)
            self.pad_id = self.token2id['_']
            self.has_tokens = True
        else:
            logging.debug('Initialize Tokenizer without tokens file, '
                          '                will fail when map to ids.')
            self.has_tokens = False

    def g2p(self, text: str) -> List[str]:
        # [E1-2] text 是 str 形参:非 str → TypeError("Argument 'text' ...")
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        try:
            # [E1-6] phonemize_espeak 返回「按句分的字符表」[[c...],...],
            #        多句需 reduce 拼接(probe_w5_6: "Hello. World." → 14 个字符)
            tokens = list(reduce(lambda x, y: x + y, phonemize_espeak(text, self.lang)))
        except Exception as ex:
            logging.warning(f'Tokenization of {self.lang} texts failed: {ex}')
            tokens = []
        return tokens

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        # [E1-9] 实测无前置断言:None/非 list 入参先撞 texts_to_tokens 的
        #        TypeError,len 才轮到 tokens_to_token_ids 里的断言
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        tokens_list = []
        for i in range(len(texts)):
            tokens_list.append(self.g2p(texts[i]))
        return tokens_list

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        assert self.has_tokens, 'Please initialize Tokenizer with a tokens file.'
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    logging.debug(f'Skip OOV {t}')
            token_ids_list.append(token_ids)
        return token_ids_list


class EmiliaTokenizer(Tokenizer):

    def __init__(self, token_file: Optional[str] = None, token_type='phone'):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        # [E1-4] 仅支持 phone;char/phoneme → AssertionError(带类型名)
        assert token_type == 'phone', \
            f'Only support phone tokenizer for Emilia, but get {token_type}.'
        self.chinese_normalizer = ChineseTextNormalizer()
        self.english_normalizer = EnglishTextNormalizer()
        if token_file is not None:
            self.token2id = {}
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    info = line.split('\t')
                    token, id = info[0], int(info[1])
                    self.token2id[token] = id
            self.vocab_size = len(self.token2id)
            self.pad_id = self.token2id['_']
            self.has_tokens = True
        else:
            logging.debug('Initialize Tokenizer without tokens file, '
                          '                    will fail when map to ids.')
            self.has_tokens = False

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        # [E1-4] 实测无 has_tokens 断言:str 入参先撞 texts_to_tokens 的 item
        #        assignment(见 diff probe),列表入参才由 tokens_to_token_ids 断言拦下
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def preprocess_text(self, text: str) -> str:
        # [E1-5] 全角标点 → 半角 + '...' → '…'(probe_w5_5 preprocess_text 矩阵)
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        return self.map_punctuations(text)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        # [E1-4/7] 实测行为:就地覆写 texts[i](str 入参 → TypeError item assignment);
        #        all_phoneme 在**每个文本**重置、append 在**外层循环之外**,
        #        故 (a) texts=[] → UnboundLocalError,(b) 多文本只返回最后一个文本的
        #        token(probe_w5_4 'two' / probe_w5_7 'emi_ttt_two' 实测)。
        for i in range(len(texts)):
            texts[i] = self.preprocess_text(texts[i])
        phoneme_list = []
        for i in range(len(texts)):
            text = texts[i]
            all_phoneme = []
            segments = self.get_segment(text)
            for index, seg in enumerate(segments):
                if seg[1] == 'zh':
                    phoneme = self.tokenize_ZH(seg[0])
                elif seg[1] == 'en':
                    phoneme = self.tokenize_EN(seg[0])
                elif seg[1] == 'pinyin':
                    phoneme = self.tokenize_pinyin(seg[0])
                elif seg[1] == 'tag':
                    phoneme = [seg[0]]
                else:
                    logging.warning(f'''No English or Chinese characters found, \\
                            skipping segment of unknown language: {seg}''')
                    phoneme = []
                all_phoneme += phoneme
        phoneme_list.append(all_phoneme)
        return phoneme_list

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        assert self.has_tokens, 'Please initialize Tokenizer with a tokens file.'
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    logging.debug(f'Skip OOV {t}')
            token_ids_list.append(token_ids)
        return token_ids_list

    def tokenize_ZH(self, text: str) -> List[str]:
        # [E1-6] 中文正则化(cn2an)→ jieba 分词 → 逐词 TONE3 + tone_sandhi → 声韵拆分
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        try:
            text = self.chinese_normalizer.normalize(text)
            segs = jieba.lcut(text)
            full = []
            for x in segs:
                if self.is_chinese(x[0]):
                    phones = lazy_pinyin([x], style=Style.TONE3, tone_sandhi=True)
                    for x in phones:
                        full += self.seperate_pinyin(x)
                else:
                    full.append(x)
            return full
        except Exception as ex:
            logging.warning(f'Tokenization of Chinese texts failed: {ex}')
            return []

    def tokenize_EN(self, text: str) -> List[str]:
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        try:
            # [E1-8] 实测(probe_w5_8 en_abc123def_phon / emi_en):先过
            #        EnglishTextNormalizer 再 espeak;多句用 reduce 拼接
            text = self.english_normalizer.normalize(text)
            tokens = list(reduce(lambda x, y: x + y, phonemize_espeak(text, 'en-us')))
        except Exception as ex:
            logging.warning(f'Tokenization of English texts failed: {ex}')
            tokens = []
        return tokens

    def tokenize_pinyin(self, text: str) -> List[str]:
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        try:
            assert self.is_pinyin(text)
            text = text[1:-1]
            # [E1-7] 合法性判据(probe_w5_7 pinyin_rule 36 例):字母串 + 末位 1..5。
            #        '<abc1>'/'<l1>'/'<Ni3>'/'<lü4>' 合法;'<abc9>'/'<ha1o>'/
            #        '<ni3hao3>'/'<1a>'/'<a-b>'/'<>' 非法(告警并跳过)
            if not text[:-1].isalpha() or not ('1' <= text[-1] <= '5'):
                logging.warning(f'''Strings enclosed with <> should be pinyin, \\
                    but got: {text}. Skipped it. ''')
                return []
            return self.seperate_pinyin(text)
        except Exception as ex:
            logging.warning(f'Tokenize pinyin failed: {ex}')
            return []

    def seperate_pinyin(self, text: str) -> List[str]:
        """
        Separate pinyin into initial and final
        """
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        # [E1-5] 实测 = to_initials(strict=False) + '0' 与
        #        to_finals_tone3(strict=False, neutral_tone_with_five=True)
        #        (probe_w5_5: 'x'→['x0','x5'],'le5'→['l0','e5'],''→[])
        pinyins = []
        initial = to_initials(text, strict=False)
        final = to_finals_tone3(text, strict=False, neutral_tone_with_five=True)
        if initial != '':
            pinyins.append(initial + '0')
        if final != '':
            pinyins.append(final)
        return pinyins

    def map_punctuations(self, text):
        # [E1-5] 逐字符实测(probe_w5_5 map_punctuations):仅 12 类全角标点 + 省略号
        text = text.replace('，', ',').replace('。', '.').replace('！', '!')
        text = text.replace('？', '?').replace('、', ',').replace('：', ':')
        text = text.replace('；', ';').replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'").replace('...', '…')
        return text

    def get_segment(self, text: str) -> List[str]:
        """
        Split a text into segments based on language types
        (Chinese, English, Pinyin, tags, etc.)

        Args:
            text (str): Input text to be segmented

        Returns:
            List[str]: Segmented text parts with their language types

        Example:
            Input: 我们是小米人,是吗? Yes I think so!霍...啦啦啦
            Output: [('我们是小米人,是吗? ', 'zh'),
                ('Yes I think so!', 'en'), ('霍...啦啦啦', 'zh')]
        """
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        segments = []
        types = []
        temp_seg = ''
        temp_lang = ''
        # [E1-5/6] <> 与 [] 括起部分独立成段(probe_w5_5 get_segment 矩阵)
        _part_pattern = re.compile(r'([<[].*?[>\]])')
        _text = _part_pattern.split(text)
        for i, part in enumerate(_text):
            if part == '':
                continue
            if self.is_pinyin(part) or self.is_tag(part):
                if temp_seg != '':
                    segments.append((temp_seg, temp_lang))
                    temp_seg = ''
                segments.append((part, 'pinyin' if self.is_pinyin(part) else 'tag'))
                continue
            for char in part:
                # [E1-6] '\n' 不产出 token('a\nb' → 'ab')
                if char == '\n':
                    continue
                if self.is_chinese(char):
                    lang = 'zh'
                elif self.is_alphabet(char):
                    lang = 'en'
                else:
                    lang = ''
                if lang == '':
                    temp_seg += char
                elif temp_lang == '':
                    temp_lang = lang
                    temp_seg += char
                elif temp_lang == lang:
                    temp_seg += char
                else:
                    if temp_seg != '':
                        segments.append((temp_seg, temp_lang))
                    temp_seg = char
                    temp_lang = lang
        if temp_seg != '':
            segments.append((temp_seg, temp_lang))
        # [E1-7] 空 lang 段补齐:取首个"可判定"段的语言;'tag' 不参与,
        #        拼音段按 'zh' 计(probe_w5_7: '  <ni3>'→[['  ','zh'],['<ni3>','pinyin']],
        #        '  [x]'→[['  ','other'],...],'  [x] hello'→[['  ','en'],...])
        types = ['zh' if one[1] == 'pinyin' else one[1]
                 for one in segments if one[1] not in ('', 'tag')]
        for i, one in enumerate(segments):
            if one[1] == '':
                segments[i] = (one[0], types[0] if types else 'other')
        return segments

    def split_segments(self, segments):
        """
        split segments into smaller parts if special strings enclosed by [] or <>
        are found, where <> denotes pinyin strings, [] denotes other special strings.

        Args:
            segments (list): A list of tuples where each tuple contains:
                - temp_seg (str): The text segment to be split.
                - temp_lang (str): The language code associated with the segment.

        Returns:
            list: A list of smaller segments.
        """
        result = []
        for temp_seg, temp_lang in segments:
            parts = re.findall(r'[<[].*?[>\]]|.', temp_seg)
            for part in parts:
                if self.is_pinyin(part):
                    lang = 'pinyin'
                elif self.is_tag(part):
                    lang = 'tag'
                else:
                    lang = temp_lang
                if result and result[-1][1] == lang:
                    result[-1] = (result[-1][0] + part, lang)
                else:
                    result.append((part, lang))
        return result

    def is_chinese(self, char: str) -> bool:
        if char is not None and not isinstance(char, str):
            raise TypeError("Argument 'char' has incorrect type (expected str, got %s)"
                            % type(char).__name__)
        # [E1-4] 仅 CJK 统一表意区为真(全角标点/空格/数字为假)
        return '\u4e00' <= char <= '\u9fff'

    def is_alphabet(self, char: str) -> bool:
        if char is not None and not isinstance(char, str):
            raise TypeError("Argument 'char' has incorrect type (expected str, got %s)"
                            % type(char).__name__)
        # [E1-4] 仅 ASCII 字母为真('ā' 为假)
        return ('a' <= char <= 'z') or ('A' <= char <= 'Z')

    def is_pinyin(self, part: str) -> bool:
        if part is not None and not isinstance(part, str):
            raise TypeError("Argument 'part' has incorrect type (expected str, got %s)"
                            % type(part).__name__)
        # [E1-4] 仅看括号包裹('<ni3>' 真,'ni3'/'<ni3>hao3' 假)
        return part.startswith('<') and part.endswith('>')

    def is_tag(self, part: str) -> bool:
        if part is not None and not isinstance(part, str):
            raise TypeError("Argument 'part' has incorrect type (expected str, got %s)"
                            % type(part).__name__)
        return part.startswith('[') and part.endswith(']')


class DialogTokenizer(EmiliaTokenizer):

    def __init__(self, token_file: Optional[str] = None, token_type='phone'):
        super().__init__(token_file, token_type)
        if token_file is not None:
            # [E1-4] 词表必须含 '[S1]'/'[S2]',否则 KeyError;实测 spk_a_id=360/361
            self.spk_a_id = self.token2id['[S1]']
            self.spk_b_id = self.token2id['[S2]']

    def preprocess_text(self, text: str) -> str:
        if text is not None and not isinstance(text, str):
            raise TypeError("Argument 'text' has incorrect type (expected str, got %s)"
                            % type(text).__name__)
        # [E1-5] 去掉 [S1]/[S2] 两侧空白(probe_w5_4 DialogTokenizer.preprocess_text)
        return re.sub(r'\s*(\[S[12]\])\s*', r'\1', super().preprocess_text(text))


class LibriTTSTokenizer(Tokenizer):

    def __init__(self, token_file: Optional[str] = None, token_type='char'):
        """
        Args:
          type: the type of tokenizer, e.g., bpe, char, phone.
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        # [E1-4] 非法 token_type → AssertionError(无文案)
        assert token_type in ('char', 'phone', 'bpe')
        try:
            from tacotron_cleaner.cleaners import custom_english_cleaners
        except ImportError as ex:
            # [E1-4] oracle 环境未装 espnet_tts_frontend:实测 RuntimeError,
            #        文案 = "{ex}\nPlease run\npip install espnet_tts_frontend"
            raise RuntimeError(f'{ex}\nPlease run\npip install espnet_tts_frontend')
        self.normalize = custom_english_cleaners
        self.type = token_type
        if token_file is not None:
            if token_type == 'bpe':
                # [未知] 应用环境无 spm 模型,该分支不可达;按 sentencepiece 惯例重建
                from sentencepiece import SentencePieceProcessor
                spm = SentencePieceProcessor()
                spm.load(token_file)
                self.spm = spm
                self.vocab_size = spm.get_piece_size()
                self.has_tokens = True
                return
            self.token2id = {}
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    info = line.split('\t')
                    token, id = info[0], int(info[1])
                    self.token2id[token] = id
            self.vocab_size = len(self.token2id)
            self.pad_id = self.token2id['_']
            self.has_tokens = True
        else:
            logging.debug('Initialize Tokenizer without tokens file, '
                          '                will fail when map to ids.')
            self.has_tokens = False

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        # [E1-9] 实测无前置断言:None/非 list 入参先撞 texts_to_tokens 的
        #        TypeError,len 才轮到 tokens_to_token_ids 里的断言
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        # [E1-4] 就地覆写 texts[i](str 入参 → TypeError item assignment);
        #        char → 逐字符;phone → phonemize_espeak 原始嵌套结构(实测 [[音素...]])
        for i in range(len(texts)):
            texts[i] = self.normalize(texts[i])
        tokens_list = []
        if self.type == 'char':
            for i in range(len(texts)):
                tokens_list.append(list(texts[i]))
        elif self.type == 'phone':
            for i in range(len(texts)):
                tokens_list.append(phonemize_espeak(texts[i], 'en-us'))
        else:
            for i in range(len(texts)):
                tokens_list.append(self.spm.encode(texts[i], out_type=str))
        return tokens_list

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        assert self.has_tokens, 'Please initialize Tokenizer with a tokens file.'
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    logging.debug(f'Skip OOV {t}')
            token_ids_list.append(token_ids)
        return token_ids_list


def add_tokens(cut_set: CutSet, tokenizer: str, lang: str):
    """向 CutSet 注入 token(工厂 + map)。

    [E1-5] oracle 行为(probe_w5_6 add_tokens):
      * 形参三个都是 Cython 强类型:传非 str/非 CutSet → TypeError
        ("Argument 'tokenizer' has incorrect type (expected str, got EmiliaTokenizer)");
      * 'bogus' → ValueError('Unsupported tokenizer: bogus.');
      * 'libritts' → LibriTTSTokenizer() 内 tacotron_cleaner 缺失 → RuntimeError;
      * 'simple'/'espeak'/'emilia'/'dialog' → 工厂把对象**赋回 str 形参**,
        Cython 抛 TypeError("Expected unicode, got SimpleTokenizer");
      * 因此 _prepare_cut 在本版本不可达;下面按其可见字符串与 lhotse 惯例重建。
    """
    if tokenizer is not None and not isinstance(tokenizer, str):
        raise TypeError("Argument 'tokenizer' has incorrect type (expected str, got %s)"
                        % type(tokenizer).__name__)
    if lang is not None and not isinstance(lang, str):
        raise TypeError("Argument 'lang' has incorrect type (expected str, got %s)"
                        % type(lang).__name__)
    if tokenizer == 'simple':
        _tokenizer = SimpleTokenizer()
    elif tokenizer == 'espeak':
        _tokenizer = EspeakTokenizer()
    elif tokenizer == 'emilia':
        _tokenizer = EmiliaTokenizer()
    elif tokenizer == 'dialog':
        _tokenizer = DialogTokenizer()
    elif tokenizer == 'libritts':
        _tokenizer = LibriTTSTokenizer()
    else:
        raise ValueError(f'Unsupported tokenizer: {tokenizer}.')

    def _prepare_cut(cut):
        text = reduce(lambda x, y: x + y, [one.text for one in cut.supervisions])
        if lang == 'zh':
            tokens = _tokenizer.tokenize_ZH(text)
        elif lang == 'en':
            tokens = _tokenizer.tokenize_EN(text)
        else:
            tokens = _tokenizer.texts_to_tokens([text])[0]
        cut.custom = dict(cut.custom or {})
        cut.custom['tokens'] = tokens
        return cut

    # [E1-5] oracle 在这一步(工厂对象赋回 str 形参)即抛 TypeError,永不返回
    raise TypeError('Expected unicode, got %s' % type(_tokenizer).__name__)
