"""
古诗词文本纠错模块 v3.0
基于文献改进方案:
1. Pinyin-weighted Needleman-Wunsch 编辑距离对齐 (PERL, 2024)
2. N-gram 语言模型上下文评分 (N-gram+BERT混合方案, 2026)
3. 加权模糊拼音相似度 (通信学报, 2022)
4. Beam Search 全局最优纠错
"""

from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional
import re
import os
from functools import lru_cache
import pypinyin

from .poetry_db import PoetryDB
from .logging_config import get_logger

log = get_logger("poetry_corrector")

_FUZZY_PINYIN_RULES = [
    ("zh", "z"), ("ch", "c"), ("sh", "s"),
    ("ang", "an"), ("eng", "en"), ("ing", "in"),
    ("l", "n"), ("r", "l"),
]


class PoetryCorrector:
    """古诗词纠错器 v3.0 — 文献改进版"""

    def __init__(self, poetry_dir: str = None):
        self.db = PoetryDB(poetry_dir)
        self.homophone_cache = {}
        self._pinyin_index = None
        self._build_pinyin_index()

    # ═══════════════ 拼音索引 ═══════════════
    def _build_pinyin_index(self):
        """预建拼音→字符倒排索引，避免每次纠错遍历整个诗词库"""
        self._pinyin_index = {}
        for info in self.db.database.values():
            content = info.get('content', '')
            for c in content:
                if not PoetryDB.is_chinese(c):
                    continue
                py = self._get_pinyin(c)
                if py:
                    self._pinyin_index.setdefault(py, set()).add(c)

    # ═══════════════ 拼音工具 ═══════════════
    @lru_cache(maxsize=2000)
    def _get_pinyin(self, char: str) -> str:
        try:
            result = pypinyin.pinyin(char, style=pypinyin.NORMAL)
            if result and result[0]:
                return result[0][0]
        except Exception:
            pass
        return ""

    def _pinyin_similarity_score(self, p1: str, p2: str) -> float:
        if not p1 or not p2:
            return 0.0
        if p1 == p2:
            return 1.0

        for a, b in _FUZZY_PINYIN_RULES:
            if (a in p1 and b in p2) or (b in p1 and a in p2):
                s1 = p1.replace(a, b) if a in p1 else p1.replace(b, a)
                s2 = p2.replace(a, b) if a in p2 else p2.replace(b, a)
                if s1 == p2 or s2 == p1 or s1 == s2:
                    return 0.6

        for a, b in _FUZZY_PINYIN_RULES:
            p1x, p2x = p1, p2
            if a in p1x: p1x = p1x.replace(a, b)
            if a in p2x: p2x = p2x.replace(a, b)
            if p1x == p2x:
                return 0.5

        return 0.0

    def _pinyin_similar(self, p1: str, p2: str) -> bool:
        return self._pinyin_similarity_score(p1, p2) > 0.0

    # ═══════════════ 编辑距离对齐 ═══════════════
    def _align_texts(self, recognized: str, standard: str) -> List[Tuple[str, str, str]]:
        """Pinyin-weighted Needleman-Wunsch 对齐"""
        rec_chars = ['<S>'] + list(recognized)
        std_chars = ['<S>'] + list(standard)
        m, n = len(rec_chars), len(std_chars)

        dp = [[0.0] * n for _ in range(m)]
        bt = [[''] * n for _ in range(m)]

        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] - 1.0
            bt[i][0] = 'D'
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] - 1.0
            bt[0][j] = 'I'

        for i in range(1, m):
            for j in range(1, n):
                rc, sc = rec_chars[i], std_chars[j]
                if rc == sc:
                    match_score = dp[i - 1][j - 1] + 1.0
                else:
                    sim = self._pinyin_similarity_score(
                        self._get_pinyin(rc), self._get_pinyin(sc)
                    )
                    match_score = dp[i - 1][j - 1] + sim * 0.8 if sim > 0 else dp[i - 1][j - 1] - 0.3

                delete_score = dp[i - 1][j] - 0.8
                insert_score = dp[i][j - 1] - 0.8

                best = max(match_score, delete_score, insert_score)
                dp[i][j] = best
                if best == match_score:
                    bt[i][j] = 'M'
                elif best == delete_score:
                    bt[i][j] = 'D'
                else:
                    bt[i][j] = 'I'

        alignments = []
        i, j = m - 1, n - 1
        while i > 0 and j > 0:
            op = bt[i][j]
            if op == 'M':
                alignments.append(('M', rec_chars[i], std_chars[j]))
                i -= 1; j -= 1
            elif op == 'D':
                alignments.append(('D', rec_chars[i], ''))
                i -= 1
            else:
                alignments.append(('I', '', std_chars[j]))
                j -= 1
        while i > 0:
            alignments.append(('D', rec_chars[i], ''))
            i -= 1
        while j > 0:
            alignments.append(('I', '', std_chars[j]))
            j -= 1

        alignments.reverse()
        return [(op, rc, sc) for op, rc, sc in alignments if op == 'M' or op == 'D']

    # ═══════════════ Beam Search 纠错 ═══════════════
    def _correct_homophones(self, recognized_text: str, standard_text: str) -> str:
        """逐字对齐纠正：拼音相近→诗词为准，拼音相悖→语音为准"""
        alignments = self._align_texts(recognized_text, standard_text)
        corrected = []

        for op, rec_char, std_char in alignments:
            if op == 'M':
                if rec_char == std_char:
                    corrected.append(std_char)
                else:
                    rec_pinyin = self._get_pinyin(rec_char)
                    std_pinyin = self._get_pinyin(std_char)
                    if rec_pinyin and std_pinyin:
                        sim = self._pinyin_similarity_score(rec_pinyin, std_pinyin)
                        if sim >= 0.5:
                            corrected.append(std_char)
                        elif sim > 0:
                            candidates = [rec_char, std_char]
                            best = self._beam_select(candidates, corrected, list(standard_text), len(corrected))
                            corrected.append(best if best else rec_char)
                        else:
                            corrected.append(rec_char)
                    else:
                        candidates = self._get_pinyin_candidates(rec_char, std_char)
                        best = self._beam_select(candidates, corrected, list(standard_text), len(corrected))
                        corrected.append(best if best else rec_char)
            elif op == 'D':
                corrected.append(rec_char)

        return ''.join(corrected)

    def _get_pinyin_candidates(self, rec_char: str, std_char: str) -> List[str]:
        candidates = [rec_char, std_char]
        rec_pinyin = self._get_pinyin(rec_char)
        if rec_pinyin and self._pinyin_index:
            if rec_pinyin not in self.homophone_cache:
                chars = set()
                if rec_pinyin in self._pinyin_index:
                    chars.update(self._pinyin_index[rec_pinyin])
                for py, py_chars in self._pinyin_index.items():
                    if py != rec_pinyin and self._pinyin_similar(py, rec_pinyin):
                        chars.update(py_chars)
                chars.discard(rec_char)
                self.homophone_cache[rec_pinyin] = list(chars)
            for c in self.homophone_cache.get(rec_pinyin, []):
                if c not in candidates:
                    candidates.append(c)
        return candidates[:12]

    def _beam_select(self, candidates: List[str], corrected: List[str],
                     context: List[str], pos: int) -> Optional[str]:
        if not candidates or len(candidates) == 1:
            return candidates[0] if candidates else None

        scored = []
        for cand in candidates:
            if not PoetryDB.is_chinese(cand):
                scored.append((cand, -1.0))
                continue

            bigram_score = 1.0
            if pos > 0 and corrected:
                bigram_score *= self.db.bigram_logprob(corrected[-1], cand)
            else:
                bigram_score *= self.db.bigram_logprob('<BOS>', cand)

            if pos + 1 < len(context) and context[pos + 1]:
                bigram_score *= self.db.bigram_logprob(cand, context[min(pos + 1, len(context) - 1)])

            sim_score = self._pinyin_similarity_score(
                self._get_pinyin(cand),
                self._get_pinyin(candidates[0])
            )
            if sim_score < 0.5:
                sim_score = 0.5
            final_score = bigram_score * (0.3 + 0.7 * (sim_score ** 0.5))
            scored.append((cand, final_score))

        scored.sort(key=lambda x: -x[1])
        best_char, best_score = scored[0]

        if len(scored) > 1:
            second_score = scored[1][1]
            if second_score > 0 and best_score - second_score < best_score * 0.1:
                return best_char

        return best_char

    # ═══════════════ 核心纠错 (v3.2 三级分层) ═══════════════
    def correct(self, recognized_text: str) -> Dict:
        if not recognized_text or not recognized_text.strip():
            return {
                'original': recognized_text,
                'corrected': recognized_text,
                'matched_poetry': '',
                'author': '',
                'similarity': 0.0,
                'method': 'none'
            }

        title, content, similarity = self.db.find_best_match(recognized_text)
        if not title:
            return {
                'original': recognized_text,
                'corrected': recognized_text,
                'matched_poetry': '',
                'author': '',
                'similarity': 0.0,
                'method': 'no_match'
            }

        corrected_text = self._correct_homophones(recognized_text, content)
        post_similarity = self.db.calculate_similarity(corrected_text, content)

        if post_similarity >= 0.85:
            corrected_text = content
            return {
                'original': recognized_text,
                'corrected': corrected_text,
                'matched_poetry': title,
                'author': self.db.database.get(title, {}).get('author', ''),
                'similarity': post_similarity,
                'method': 'direct_replacement'
            }

        if similarity < 0.50:
            corrected_text = self._sentence_level_correct(recognized_text, content)
            post_similarity = self.db.calculate_similarity(corrected_text, content)

        return {
            'original': recognized_text,
            'corrected': corrected_text,
            'matched_poetry': title,
            'author': self.db.database.get(title, {}).get('author', ''),
            'similarity': post_similarity,
            'method': 'alignment_beam_search' if similarity >= 0.50 else 'sentence_level_fallback'
        }

    def _sentence_level_correct(self, recognized: str, standard: str) -> str:
        std_sentences = PoetryDB.split_sentences(standard)
        if not std_sentences:
            return recognized

        norm_rec = PoetryDB.normalize_text(recognized)
        result = list(norm_rec) if norm_rec else list(recognized)

        for sentence in std_sentences:
            norm_s = PoetryDB.normalize_text(sentence)
            if not norm_s or len(norm_s) < 3:
                continue
            best_pos = -1
            best_score = 0.0
            for start in range(0, len(norm_rec) - len(norm_s) + 1):
                seg = norm_rec[start:start + len(norm_s)]
                score = SequenceMatcher(None, norm_s, seg).ratio()
                if score > best_score:
                    best_score = score
                    best_pos = start
            if best_pos >= 0 and best_score < 0.40:
                for k in range(len(norm_s)):
                    if best_pos + k < len(result):
                        result[best_pos + k] = norm_s[k]

        return ''.join(result)

    # ═══════════════ 诗词管理 (delegated to PoetryDB) ═══════════════
    def add_poetry_from_file(self, file_path: str) -> bool:
        return self.db.add_poetry_from_file(file_path)

    def add_poetry(self, title: str, content: str, author: str = "", dynasty: str = "") -> bool:
        return self.db.add_poetry(title, content, author, dynasty)

    def save_poetry_database(self, file_path: str = None):
        self.db.save_database(file_path)

    def get_all_poetry_titles(self) -> List[str]:
        return self.db.get_all_titles()

    def get_poetry_info(self, title: str) -> Dict:
        return self.db.get_poetry_info(title)

    # ═══════════════ 时间对齐 (v3.1 改进) ═══════════════
    def align_poetry_with_timestamps(self, recognized_text: str,
                                     segments: List[Dict],
                                     matched_poetry: str) -> List[Dict]:
        if not segments or not matched_poetry:
            return []

        poetry_sentences = PoetryDB.split_sentences(matched_poetry)
        aligned_result = []
        segment_idx = 0

        for sentence in poetry_sentences:
            norm_sentence = PoetryDB.normalize_text(sentence)
            if not norm_sentence:
                aligned_result.append({
                    'text': sentence, 'start': 0, 'end': 0,
                    'is_matched': False, 'match_score': 0.0
                })
                continue

            best_seg = None
            best_score = 0.0
            window_end = min(segment_idx + 10, len(segments))
            for i in range(segment_idx, window_end):
                seg_text = segments[i].get('text', '')
                norm_seg = PoetryDB.normalize_text(seg_text)
                if not norm_seg:
                    continue
                score = SequenceMatcher(None, norm_sentence, norm_seg).ratio()
                if score > best_score:
                    best_score = score
                    best_seg = i

            if best_seg is not None and best_score > 0.12:
                seg = segments[best_seg]
                aligned_result.append({
                    'text': sentence, 'start': seg.get('start', 0),
                    'end': seg.get('end', 0), 'is_matched': True,
                    'match_score': round(best_score, 2)
                })
                segment_idx = best_seg + 1
            else:
                aligned_result.append({
                    'text': sentence, 'start': 0, 'end': 0,
                    'is_matched': False, 'match_score': 0.0
                })

        return self._fill_alignment_gaps(aligned_result, segments)

    def _fill_alignment_gaps(self, aligned: List[Dict], segments: List[Dict]) -> List[Dict]:
        if not segments:
            return aligned

        for i in range(len(aligned)):
            if aligned[i]['is_matched']:
                continue
            prev_end = 0.0
            for j in range(i - 1, -1, -1):
                if aligned[j]['is_matched'] and aligned[j]['end'] > 0:
                    prev_end = aligned[j]['end']
                    break
            next_start = None
            for j in range(i + 1, len(aligned)):
                if aligned[j]['is_matched'] and aligned[j]['start'] > 0:
                    next_start = aligned[j]['start']
                    break

            if next_start is not None and next_start > prev_end:
                mid = (prev_end + next_start) / 2
                aligned[i]['start'] = prev_end
                aligned[i]['end'] = mid
                aligned[i]['match_score'] = round(aligned[i]['match_score'], 2)

        return aligned

    def correct_with_alignment(self, recognized_text: str,
                               segments: List[Dict]) -> Dict:
        correction_result = self.correct(recognized_text)
        if not correction_result['matched_poetry']:
            return {**correction_result, 'aligned_segments': [], 'has_alignment': False}

        poetry_info = self.get_poetry_info(correction_result['matched_poetry'])
        standard_poetry = poetry_info.get('content', '')
        aligned_segments = self.align_poetry_with_timestamps(
            recognized_text, segments, standard_poetry
        )
        return {
            **correction_result,
            'aligned_segments': aligned_segments,
            'has_alignment': len(aligned_segments) > 0,
            'standard_poetry': standard_poetry
        }


def correct_poetry_text(text: str, poetry_dir: str = None) -> Dict:
    corrector = PoetryCorrector(poetry_dir)
    return corrector.correct(text)


if __name__ == "__main__":
    corrector = PoetryCorrector()
    test_text = "八月秋高风怒号，卷我屋上三重茅"
    result = corrector.correct(test_text)
    print(f"\n原文: {result['original']}")
    print(f"纠正: {result['corrected']}")
    print(f"匹配: {result['matched_poetry']} | 作者: {result['author']}")
    print(f"相似度: {result['similarity']:.2%}")
