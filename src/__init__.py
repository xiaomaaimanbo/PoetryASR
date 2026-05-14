# Lazy imports — only load heavy dependencies when actually accessed

__all__ = [
    "SpeechRecognizer",
    "AudioProcessor",
    "SpeechDataLoader",
    "PoetryCorrector",
    "correct_poetry_text",
    "RealTimeRecognizer",
    "realtime_transcribe",
    "MimoClient",
    "MimoConfig",
    "create_mimo_client",
    "PoetryDB",
    "TranscriptionPipeline",
    "TranscriptionResult",
]


def __getattr__(name):
    if name == "SpeechRecognizer":
        from .speech_recognizer import SpeechRecognizer; return SpeechRecognizer
    if name == "AudioProcessor":
        from .audio_processor import AudioProcessor; return AudioProcessor
    if name == "SpeechDataLoader":
        from .data_loader import SpeechDataLoader; return SpeechDataLoader
    if name == "PoetryCorrector":
        from .poetry_corrector import PoetryCorrector; return PoetryCorrector
    if name == "correct_poetry_text":
        from .poetry_corrector import correct_poetry_text; return correct_poetry_text
    if name == "RealTimeRecognizer":
        from .realtime_recognizer import RealTimeRecognizer; return RealTimeRecognizer
    if name == "realtime_transcribe":
        from .realtime_recognizer import realtime_transcribe; return realtime_transcribe
    if name == "MimoClient":
        from .mimo_client import MimoClient; return MimoClient
    if name == "MimoConfig":
        from .mimo_client import MimoConfig; return MimoConfig
    if name == "create_mimo_client":
        from .mimo_client import create_mimo_client; return create_mimo_client
    if name == "PoetryDB":
        from .poetry_db import PoetryDB; return PoetryDB
    if name == "TranscriptionPipeline":
        from .pipeline import TranscriptionPipeline; return TranscriptionPipeline
    if name == "TranscriptionResult":
        from .pipeline import TranscriptionResult; return TranscriptionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
