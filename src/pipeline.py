"""
Shared transcription pipeline — single implementation used by CLI and Web UI.
Eliminates the duplicated logic spread across main.py, launcher.py, and webui.py.
"""

import os
import logging
from typing import Dict, Optional, List

from .speech_recognizer import SpeechRecognizer
from .poetry_corrector import PoetryCorrector
from .logging_config import get_logger

log = get_logger("pipeline")


class TranscriptionResult:
    """Structured result from a pipeline run."""

    __slots__ = (
        "text", "segments", "word_timestamps",
        "matched_poetry", "author", "similarity", "method",
        "corrected", "aligned_segments", "has_alignment",
        "standard_poetry", "duration", "model_used",
    )

    def __init__(self):
        self.text = ""
        self.segments: List[Dict] = []
        self.word_timestamps: List[Dict] = []
        self.matched_poetry = ""
        self.author = ""
        self.similarity = 0.0
        self.method = "none"
        self.corrected = ""
        self.aligned_segments: List[Dict] = []
        self.has_alignment = False
        self.standard_poetry = ""
        self.duration = 0.0
        self.model_used = ""

    def to_dict(self) -> Dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class TranscriptionPipeline:
    """End-to-end speech recognition pipeline with poetry correction."""

    def __init__(
        self,
        model_name: str = "medium",
        language: str = "zh",
        device: Optional[str] = None,
        poetry_dir: Optional[str] = None,
        enable_poetry: bool = True,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.enable_poetry = enable_poetry

        self._recognizer: Optional[SpeechRecognizer] = None
        self._corrector: Optional[PoetryCorrector] = None
        self._poetry_dir = poetry_dir

    @property
    def recognizer(self) -> SpeechRecognizer:
        if self._recognizer is None:
            self._recognizer = SpeechRecognizer(
                model_name=self.model_name,
                language=self.language,
                device=self.device,
            )
        return self._recognizer

    @property
    def corrector(self) -> PoetryCorrector:
        if self._corrector is None:
            self._corrector = PoetryCorrector(self._poetry_dir)
        return self._corrector

    def reset_model(self, model_name: str = None, language: str = None):
        """Invalidate cached model so next run picks up new settings."""
        if model_name is not None:
            self.model_name = model_name
        if language is not None:
            self.language = language
        self._recognizer = None
        self._corrector = None

    def run(self, audio_path: str) -> TranscriptionResult:
        """Run full pipeline: transcribe → (optional) poetry correction → alignment."""
        result = TranscriptionResult()
        result.model_used = self.model_name

        rec = self.recognizer
        log.info("转录音频: %s (model=%s)", audio_path, self.model_name)
        whisper_result = rec.transcribe_audio(audio_path, word_timestamps=True)

        result.text = rec.extract_text(whisper_result)
        result.segments = rec.extract_segments(whisper_result)
        result.word_timestamps = rec.get_word_level_timestamps(whisper_result)

        if result.segments:
            result.duration = result.segments[-1].get("end", 0)

        if self.enable_poetry and result.text.strip():
            self._run_poetry_correction(result)

        return result

    def run_from_data(self, audio_data, sample_rate: int = 16000) -> TranscriptionResult:
        """Run pipeline on numpy audio data (e.g. from microphone)."""
        result = TranscriptionResult()
        result.model_used = self.model_name

        rec = self.recognizer
        whisper_result = rec.transcribe_audio_data(audio_data, sample_rate, word_timestamps=True)

        result.text = rec.extract_text(whisper_result)
        result.segments = rec.extract_segments(whisper_result)
        result.word_timestamps = rec.get_word_level_timestamps(whisper_result)

        if result.segments:
            result.duration = result.segments[-1].get("end", 0)

        if self.enable_poetry and result.text.strip():
            self._run_poetry_correction(result)

        return result

    def _run_poetry_correction(self, result: TranscriptionResult):
        corr = self.corrector
        correction = corr.correct_with_alignment(result.text, result.segments)
        result.matched_poetry = correction.get("matched_poetry", "")
        result.author = correction.get("author", "")
        result.similarity = correction.get("similarity", 0.0)
        result.method = correction.get("method", "no_match")
        result.corrected = correction.get("corrected", result.text)
        result.aligned_segments = correction.get("aligned_segments", [])
        result.has_alignment = correction.get("has_alignment", False)
        result.standard_poetry = correction.get("standard_poetry", "")

    def get_poetry_titles(self) -> List[str]:
        return self.corrector.get_all_poetry_titles()

    def add_poem(self, title: str, content: str, author: str = "", dynasty: str = "") -> bool:
        return self.corrector.add_poetry(title, content, author, dynasty)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
