"""
古诗词语音识别系统 - 单元测试 v3.1
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.audio_processor import AudioProcessor
from src.poetry_corrector import PoetryCorrector
from src.poetry_db import PoetryDB
from src.pipeline import TranscriptionPipeline, TranscriptionResult
from src.mimo_client import MimoClient, MimoConfig


class TestAudioProcessor:
    def setup_method(self):
        self.processor = AudioProcessor(sample_rate=16000)

    def test_normalize_audio(self):
        audio = np.array([0.5, -0.3, 0.1, -0.8], dtype=np.float32)
        normalized = self.processor.normalize_audio(audio)
        assert np.max(np.abs(normalized)) == 1.0

    def test_normalize_silence(self):
        audio = np.zeros(100, dtype=np.float32)
        normalized = self.processor.normalize_audio(audio)
        assert np.array_equal(normalized, audio)

    def test_add_noise_shape(self):
        audio = np.ones(1000, dtype=np.float32)
        noisy = self.processor.add_noise(audio, noise_level=0.01)
        assert noisy.shape == audio.shape
        assert not np.array_equal(noisy, audio)

    def test_pad_or_truncate_pad(self):
        audio = np.ones(8000, dtype=np.float32)
        result = self.processor.pad_or_truncate(audio, max_duration=1)
        assert len(result) == 16000

    def test_pad_or_truncate_truncate(self):
        audio = np.ones(32000, dtype=np.float32)
        result = self.processor.pad_or_truncate(audio, max_duration=1)
        assert len(result) == 16000

    def test_compute_mel_spectrogram(self):
        audio = np.random.randn(16000).astype(np.float32)
        mel = self.processor.compute_mel_spectrogram(audio)
        assert mel.ndim == 2
        assert mel.shape[0] == 80

    def test_compute_mfcc(self):
        audio = np.random.randn(16000).astype(np.float32)
        mfcc = self.processor.compute_mfcc(audio)
        assert mfcc.ndim == 2
        assert mfcc.shape[0] == 13

    def test_time_stretch(self):
        audio = np.random.randn(16000).astype(np.float32)
        stretched = self.processor.time_stretch(audio, rate=1.2)
        assert len(stretched) < 16000

    def test_pitch_shift(self):
        audio = np.random.randn(16000).astype(np.float32)
        shifted = self.processor.pitch_shift(audio, n_steps=2)
        assert shifted.shape == audio.shape


class TestPoetryDB:
    def setup_method(self):
        self.db = PoetryDB()

    def test_is_chinese(self):
        assert PoetryDB.is_chinese("床")
        assert not PoetryDB.is_chinese("A")

    def test_normalize_text(self):
        text = "床前明月光，疑是地上霜。"
        normalized = PoetryDB.normalize_text(text)
        assert "，" not in normalized
        assert "。" not in normalized
        assert "床前明月光" in normalized

    def test_calculate_similarity_same(self):
        text = "床前明月光疑是地上霜"
        score = self.db.calculate_similarity(text, text)
        assert score == 1.0

    def test_calculate_similarity_different(self):
        score = self.db.calculate_similarity("床前明月光", "举头望明月")
        assert 0 < score < 1.0

    def test_find_best_match_maowu(self):
        text = "八月秋高风怒号卷我屋上三重茅"
        title, content, score = self.db.find_best_match(text)
        assert title != ""
        assert score > 0.1

    def test_get_all_titles(self):
        titles = self.db.get_all_titles()
        assert len(titles) > 0

    def test_get_poetry_info(self):
        titles = self.db.get_all_titles()
        if titles:
            info = self.db.get_poetry_info(titles[0])
            assert "content" in info

    def test_add_poetry(self):
        ok = self.db.add_poetry("测试诗", "测试内容一二三", "测试作者", "当代")
        assert ok

    def test_bigram_after_add(self):
        self.db.add_poetry("测试诗2", "床前明月光疑是地上霜")
        score = self.db.bigram_logprob("床", "前")
        assert score > 0

    def test_bigram_model_built(self):
        assert self.db._bigram_cache is not None
        assert len(self.db._bigram_cache) > 0

    def test_bigram_common_pair(self):
        score1 = self.db.bigram_logprob("明", "月")
        score2 = self.db.bigram_logprob("明", "x")
        assert score1 > score2

    def test_split_sentences(self):
        text = "床前明月光。疑是地上霜。举头望明月，低头思故乡。"
        sentences = PoetryDB.split_sentences(text)
        assert len(sentences) >= 3

    def test_char_vocab_size(self):
        assert self.db.char_vocab_size > 0


class TestPoetryCorrector:
    def setup_method(self):
        self.corrector = PoetryCorrector()

    def test_get_pinyin(self):
        pinyin = self.corrector._get_pinyin("床")
        assert pinyin == "chuang"

    def test_correct_no_match(self):
        result = self.corrector.correct("ZZZWWWQQQ999xyz")
        assert result["method"] in ("no_match", "none") or result["similarity"] < 0.15

    def test_correct_empty(self):
        result = self.corrector.correct("")
        assert result["matched_poetry"] == ""

    def test_correct_jingyesi(self):
        result = self.corrector.correct("床前明月光疑是地上霜")
        assert result["matched_poetry"] != "" or result["similarity"] > 0

    def test_correct_with_alignment_no_match(self):
        result = self.corrector.correct_with_alignment("随机文本", [])
        assert not result.get("has_alignment", True)

    def test_add_poetry(self):
        ok = self.corrector.add_poetry("测试诗", "测试内容一二三", "测试作者", "当代")
        assert ok

    def test_get_all_poetry_titles(self):
        titles = self.corrector.get_all_poetry_titles()
        assert len(titles) > 0

    def test_get_poetry_info(self):
        titles = self.corrector.get_all_poetry_titles()
        if titles:
            info = self.corrector.get_poetry_info(titles[0])
            assert "content" in info

    # ── 拼音相似度 ──
    def test_pinyin_similarity_exact(self):
        assert self.corrector._pinyin_similarity_score("chuang", "chuang") == 1.0

    def test_pinyin_similarity_sh_s(self):
        assert self.corrector._pinyin_similarity_score("shi", "si") > 0.0
        assert self.corrector._pinyin_similarity_score("shan", "san") > 0.0

    def test_pinyin_similarity_zh_z(self):
        assert self.corrector._pinyin_similarity_score("zhi", "zi") > 0.0

    def test_pinyin_similarity_different(self):
        assert self.corrector._pinyin_similarity_score("ma", "niu") == 0.0

    def test_pinyin_similar(self):
        assert self.corrector._pinyin_similar("shi", "si") is True
        assert self.corrector._pinyin_similar("ma", "niu") is False

    # ── Alignment ──
    def test_align_identical(self):
        alignments = self.corrector._align_texts("床前明月光", "床前明月光")
        assert all(op == 'M' for op, _, _ in alignments)

    def test_align_homophone(self):
        alignments = self.corrector._align_texts("床前明月光", "床前名月光")
        m_count = sum(1 for op, _, _ in alignments if op == 'M')
        assert m_count >= 3

    # ── 纠错 v3.0 ──
    def test_correct_method_v3(self):
        result = self.corrector.correct("床前明月光疑是地上霜")
        assert result["method"] in (
            "direct_replacement", "alignment_beam_search",
            "sentence_level_fallback", "no_match", "none"
        )

    def test_not_in_database(self):
        result = self.corrector.correct("黄河远上白云间一片孤城万仞山")
        assert isinstance(result["corrected"], str)

    def test_maowu_correction(self):
        result = self.corrector.correct(
            PoetryDB.normalize_text("八月秋高风怒豪卷我屋上山重毛")
        )
        assert isinstance(result["corrected"], str)

    def test_corrector_uses_db(self):
        assert self.corrector.db is not None
        assert len(self.corrector.db.get_all_titles()) > 0


class TestTranscriptionPipeline:
    def setup_method(self):
        self.pipeline = TranscriptionPipeline(
            model_name="tiny",
            language="zh",
            device="cpu",
            enable_poetry=False,
        )

    def test_pipeline_creation(self):
        assert self.pipeline.model_name == "tiny"
        assert self.pipeline.language == "zh"

    def test_reset_model(self):
        self.pipeline.reset_model(model_name="base", language="en")
        assert self.pipeline.model_name == "base"
        assert self.pipeline.language == "en"
        assert self.pipeline._recognizer is None

    def test_get_poetry_titles(self):
        titles = self.pipeline.get_poetry_titles()
        assert isinstance(titles, list)
        assert len(titles) > 0

    def test_add_poem(self):
        ok = self.pipeline.add_poem("流水诗", "流水落花春去也", "李煜", "五代")
        assert ok

    def test_result_slots(self):
        result = TranscriptionResult()
        d = result.to_dict()
        for slot in TranscriptionResult.__slots__:
            assert slot in d

    def test_pipeline_recognizer_lazy(self):
        assert self.pipeline._recognizer is None

    def test_pipeline_corrector_lazy(self):
        assert self.pipeline._corrector is None


class TestTranscriptionResult:
    def test_defaults(self):
        r = TranscriptionResult()
        assert r.text == ""
        assert r.similarity == 0.0
        assert r.has_alignment is False
        assert r.segments == []
        assert r.aligned_segments == []

    def test_to_dict(self):
        r = TranscriptionResult()
        r.text = "测试"
        r.matched_poetry = "静夜思"
        d = r.to_dict()
        assert d["text"] == "测试"
        assert d["matched_poetry"] == "静夜思"


class TestMimoClient:
    def test_client_without_key(self):
        config = MimoConfig(api_key="")
        client = MimoClient(config)
        assert not client.available

    def test_client_with_key(self):
        config = MimoConfig(api_key="sk-test-key")
        client = MimoClient(config)
        assert client.available

    def test_refine_without_key(self):
        config = MimoConfig(api_key="")
        client = MimoClient(config)
        result = client.refine_transcription("测试文本")
        assert result["refined"] == "测试文本"

    def test_config_defaults(self):
        config = MimoConfig()
        assert config.model == "mimo-v2.5"

    def test_config_custom(self):
        config = MimoConfig(api_key="sk-custom", model="mimo-v2.5-pro", max_tokens=2048)
        assert config.model == "mimo-v2.5-pro"

    def test_is_configured(self):
        assert MimoConfig(api_key="").is_configured() is False
        assert MimoConfig(api_key="sk-xxx").is_configured() is True
