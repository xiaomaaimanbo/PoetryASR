import whisper
import torch
import os
import numpy as np
from typing import Dict, Optional, Tuple, List
from .audio_processor import AudioProcessor
from .audio_processor import estimate_speech_quality
from zhconv import convert as zh_convert

from .logging_config import get_logger

log = get_logger("speech_recognizer")


def _detect_speech_segments(audio: np.ndarray, sr: int,
                            top_db: int = 30, min_silence: float = 0.6,
                            min_dur: float = 1.0, target_dur: float = 25.0) -> List[Tuple[float, float]]:
    """基于能量检测的VAD分段 (参考 WhisperX Cut & Merge)
    返回 (start_seconds, end_seconds) 列表
    """
    import librosa
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return [(0.0, len(audio) / sr)]

    chunks = []
    chunk_start = intervals[0][0] / sr
    chunk_end = intervals[0][1] / sr

    for start_samp, end_samp in intervals[1:]:
        start_sec = start_samp / sr
        end_sec = end_samp / sr

        if start_sec - chunk_end < min_silence and (end_sec - chunk_start) <= target_dur * 1.5:
            chunk_end = end_sec
        else:
            if chunk_end - chunk_start >= min_dur:
                chunks.append((chunk_start, chunk_end))
            chunk_start = start_sec
            chunk_end = end_sec

    if chunk_end - chunk_start >= min_dur:
        chunks.append((chunk_start, chunk_end))

    if not chunks:
        return [(0.0, len(audio) / sr)]

    merged = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        if c[1] - prev[0] > target_dur * 1.5:
            merged.append(c)
        else:
            merged[-1] = (prev[0], c[1])

    return merged


class SpeechRecognizer:
    def __init__(self, model_name: str = "base", language: str = "zh", device: str = "cuda",
                 use_vad: bool = False):
        self.model_name = model_name
        self.language = language
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.audio_processor = AudioProcessor()
        self.use_vad = use_vad
        self._load_model()

    def _load_model(self):
        log.info("Loading Whisper %s model...", self.model_name)
        self.model = whisper.load_model(self.model_name).to(self.device)
        log.info("Model loaded successfully on %s", self.device)

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict:
        if self.use_vad:
            return self._transcribe_with_vad(audio_path, **kwargs)
        kwargs.setdefault("no_speech_threshold", 0.6)
        kwargs.setdefault("logprob_threshold", -1.0)
        kwargs.setdefault("compression_ratio_threshold", 2.4)
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            **kwargs
        )
        return result

    def transcribe_with_preprocess(self, audio_path: str, **kwargs) -> Dict:
        """带智能预处理的转录：自动分析音频质量并应用轻量增强"""
        raw_audio = whisper.load_audio(audio_path)
        raw_audio = raw_audio.astype(np.float32)

        quality = estimate_speech_quality(raw_audio)
        if quality["needs_preprocessing"]:
            from .logging_config import get_logger
            log = get_logger("speech_recognizer")
            log.info(
                "音频需预处理: RMS=%.1fdB SNR~%.1fdB LF=%.3f DR=%.1f",
                quality["rms_db"], quality["snr_est"],
                quality["low_freq_ratio"], quality["dynamic_range"]
            )
            processed = self.audio_processor.preprocess_for_asr(raw_audio)
            processed = processed.astype(np.float32)
        else:
            processed = raw_audio.astype(np.float32)

        result = self.model.transcribe(processed, language=self.language, **kwargs)
        return result

    def _transcribe_with_vad(self, audio_path: str, **kwargs) -> Dict:
        """WhisperX风格VAD分块转录"""
        raw_audio = whisper.load_audio(audio_path)
        duration = len(raw_audio) / 16000

        if duration < 35:
            result = self.model.transcribe(raw_audio, language=self.language, **kwargs)
            return result

        log.info("长音频%.0fs, 启动VAD分块转录...", duration)
        chunks = _detect_speech_segments(raw_audio, 16000,
            top_db=30, min_silence=0.6, min_dur=1.0, target_dur=28.0)

        all_segments = []
        full_text_parts = []

        for ci, (t_start, t_end) in enumerate(chunks):
            si = max(0, int(t_start * 16000))
            ei = min(len(raw_audio), int(t_end * 16000))
            chunk_audio = raw_audio[si:ei]

            chunk_result = self.model.transcribe(chunk_audio, language=self.language, **kwargs)
            chunk_text = chunk_result.get("text", "").strip()

            for seg in chunk_result.get("segments", []):
                seg["start"] = seg.get("start", 0) + t_start
                seg["end"] = seg.get("end", 0) + t_start
                all_segments.append(seg)

            if chunk_text:
                full_text_parts.append(chunk_text)

        return {
            "text": "".join(full_text_parts),
            "segments": all_segments,
            "language": self.language,
        }

    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000, **kwargs) -> Dict:
        audio_data = self.audio_processor.resample_audio(audio_data, sample_rate)
        audio_data = self.audio_processor.normalize_audio(audio_data)

        result = self.model.transcribe(
            audio_data,
            language=self.language,
            **kwargs
        )
        return result

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        
        return detected_lang, confidence

    def extract_text(self, result: Dict) -> str:
        text = result.get("text", "")
        return zh_convert(text, "zh-cn")

    def extract_segments(self, result: Dict) -> list:
        segments = result.get("segments", [])
        for seg in segments:
            if "text" in seg:
                seg["text"] = zh_convert(seg["text"], "zh-cn")
        return segments

    def get_word_level_timestamps(self, result: Dict) -> list:
        word_timestamps = []
        segments = self.extract_segments(result)
        
        for segment in segments:
            if "words" in segment:
                for word in segment["words"]:
                    word_timestamps.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("confidence", 1.0)
                    })
        return word_timestamps

    def save_transcription(self, result: Dict, output_path: str):
        text = self.extract_text(result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    def save_detailed_transcription(self, result: Dict, output_path: str):
        segments = self.extract_segments(result)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                f.write(f"[{start:.2f} - {end:.2f}] {text}\n")

    def fine_tune(self, train_data: str, output_dir: str = "./models", epochs: int = 1, batch_size: int = 8):
        raise NotImplementedError(
            "微调功能需要连接 Hugging Face Hub 下载处理器，当前网络环境下不可用。"
            "建议使用预训练模型进行语音识别。"
        )