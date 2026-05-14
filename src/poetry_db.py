"""
Poetry database management — loads, searches, and manages the classical Chinese poetry corpus.
Includes character bigram language model built from the poetry database.
"""

import json
import os
import re
import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

from zhconv import convert as zh_convert

from .logging_config import get_logger

log = get_logger("poetry_db")


class PoetryDB:
    """Manages the poetry database: loading, searching, adding, bigram LM."""

    def __init__(self, poetry_dir: str = None):
        if poetry_dir is None:
            poetry_dir = os.environ.get("POETRY_ASR_DATA_DIR")
        if poetry_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            poetry_dir = os.path.join(project_root, "data", "poetry")

        self.poetry_dir = poetry_dir
        self.database: Dict[str, Dict] = {}
        self._bigram_cache = None
        self._char_freq = None
        self._load_all()

    # ── Data loading ──
    def _load_all(self):
        if not os.path.exists(self.poetry_dir):
            log.warning("诗词目录不存在: %s", self.poetry_dir)
            return

        json_files = [f for f in os.listdir(self.poetry_dir) if f.endswith('.json')]
        for json_file in json_files:
            file_path = os.path.join(self.poetry_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    poetry_data = json.load(f)
                    self.database.update(poetry_data)
                log.info("已加载诗词: %s (%d 首)", json_file, len(poetry_data))
            except Exception as e:
                log.error("加载失败 %s: %s", json_file, e)

        log.info("共加载 %d 首诗词", len(self.database))
        self._build_bigram_model()

    # ── Bigram language model ──
    def _build_bigram_model(self):
        bigrams = Counter()
        chars = Counter()

        for info in self.database.values():
            content = info.get('content', '')
            norm = self.normalize_text(content)
            prev = '<BOS>'
            for ch in norm:
                chars[ch] += 1
                bigrams[(prev, ch)] += 1
                prev = ch
            if prev != '<BOS>':
                bigrams[(prev, '<EOS>')] += 1

        self._bigram_cache = bigrams
        self._char_freq = chars

    def bigram_logprob(self, ch1: str, ch2: str) -> float:
        if self._bigram_cache is None or self._char_freq is None:
            return 1.0 / 5000
        count_pair = self._bigram_cache.get((ch1, ch2), 0)
        count_ch1 = self._char_freq.get(ch1, 0)
        vocab = len(self._char_freq)
        return (count_pair + 1) / (count_ch1 + vocab) if count_ch1 > 0 else 1.0 / vocab

    def sequence_logprob(self, chars_seq: List[str]) -> float:
        if not chars_seq:
            return 0.0
        prob = 1.0
        prev = '<BOS>'
        for ch in chars_seq:
            prob *= self.bigram_logprob(prev, ch)
            prev = ch
        return prob

    @property
    def char_vocab_size(self) -> int:
        return len(self._char_freq) if self._char_freq else 0

    # ── Search ──
    def find_best_match(self, recognized_text: str) -> Tuple[str, str, float]:
        best_match = None
        best_score = 0.0
        rec_norm = self.normalize_text(recognized_text)
        rec_len = len(rec_norm)

        for title, info in self.database.items():
            content = info.get('content', '')
            std_norm = self.normalize_text(content)
            # 长度相差超过50%的跳过
            if rec_len > 0:
                ratio = len(std_norm) / rec_len
                if ratio < 0.5 or ratio > 2.0:
                    continue
            score = SequenceMatcher(None, rec_norm, std_norm).ratio()
            if score > best_score:
                best_score = score
                best_match = (title, content, score)
        return best_match if best_match else ("", "", 0.0)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        if not norm1 or not norm2:
            return 0.0
        return SequenceMatcher(None, norm1, norm2).ratio()

    # ── CRUD ──
    def add_poetry(self, title: str, content: str, author: str = "", dynasty: str = "") -> bool:
        try:
            self.database[title] = {
                'content': content,
                'author': author,
                'dynasty': dynasty
            }
            self._build_bigram_model()
            log.info("成功添加诗词: %s", title)
            return True
        except Exception as e:
            log.error("添加失败: %s", e)
            return False

    def add_poetry_from_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                poetry_data = json.load(f)
            for title, info in poetry_data.items():
                if 'content' not in info:
                    log.error("格式错误: %s 缺少content字段", title)
                    return False
            self.database.update(poetry_data)
            self._build_bigram_model()
            log.info("成功添加 %d 首诗词", len(poetry_data))
            return True
        except Exception as e:
            log.error("添加失败: %s", e)
            return False

    def save_database(self, file_path: str = None):
        if file_path is None:
            file_path = os.path.join(self.poetry_dir, "custom_poetry.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, ensure_ascii=False, indent=2)
            log.info("诗词数据库已保存到: %s", file_path)
        except Exception as e:
            log.error("保存失败: %s", e)

    def get_all_titles(self) -> List[str]:
        return list(self.database.keys())

    def get_poetry_info(self, title: str) -> Dict:
        return self.database.get(title, {})

    # ── Text utilities ──
    @staticmethod
    def is_chinese(char: str) -> bool:
        return '一' <= char <= '鿿'

    @staticmethod
    def normalize_text(text: str) -> str:
        text = zh_convert(text, "zh-cn")
        text = re.sub(r'[^一-鿿]', '', text)
        return text

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        sentences = re.split(r'[。！？\n；，、]', text)
        return [s.strip() for s in sentences if s.strip()]
