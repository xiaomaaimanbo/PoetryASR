import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def estimate_speech_quality(audio: np.ndarray, sr: int = 16000) -> dict:
    """估算音频质量指标，用于决定预处理强度
    参考: DENOASR selective denoising (2024)
    """
    rms_energy = np.sqrt(np.mean(audio ** 2))

    if rms_energy > 1e-6:
        rms_db = 20 * np.log10(rms_energy)
    else:
        rms_db = -60.0

    rms_ratio = rms_db + 30

    non_zero = audio[abs(audio) > rms_energy * 0.1]
    if len(non_zero) > 0:
        dynamic_range = float(np.max(non_zero) / (np.median(np.abs(non_zero)) + 1e-8))
    else:
        dynamic_range = 1.0

    try:
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        below_200 = np.sum(spec[freqs < 200]) / (np.sum(spec) + 1e-8)
        above_4k = np.sum(spec[freqs > 4000]) / (np.sum(spec) + 1e-8)
    except Exception:
        below_200 = 0.3
        above_4k = 0.1

    snr_est = 20 * np.log10((1 - min(below_200, 0.8)) / (max(below_200, 0.05)))

    return {
        "rms_db": round(rms_db, 1),
        "snr_est": round(snr_est, 1),
        "dynamic_range": round(dynamic_range, 1),
        "low_freq_ratio": round(below_200, 3),
        "high_freq_ratio": round(above_4k, 3),
        "needs_preprocessing": (rms_db < -25 or dynamic_range > 8.0 or below_200 > 0.15),
    }


class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr

    def save_audio(self, file_path, audio, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        sf.write(file_path, audio, sample_rate)

    def resample_audio(self, audio, original_sr):
        if original_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
        return audio

    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def trim_silence(self, audio, top_db=20):
        audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio

    def add_noise(self, audio, noise_level=0.001):
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise

    def time_stretch(self, audio, rate=1.0):
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, n_steps=0):
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def compute_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=80,
            n_fft=512,
            hop_length=160
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def compute_mfcc(self, audio, n_mfcc=13):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=160
        )
        return mfcc

    def denoise_audio(self, audio, noise_audio=None):
        if noise_audio is not None:
            noise_spec = np.abs(np.fft.fft(noise_audio))
            audio_spec = np.fft.fft(audio)
            audio_spec_denoised = audio_spec * (np.abs(audio_spec) > noise_spec)
            denoised_audio = np.real(np.fft.ifft(audio_spec_denoised))
            return denoised_audio
        else:
            b, a = signal.butter(10, 4000, 'low', fs=self.sample_rate)
            return signal.filtfilt(b, a, audio)

    def pad_or_truncate(self, audio, max_duration=30):
        max_samples = int(self.sample_rate * max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        return audio

    def audio_to_tensor(self, audio):
        import torch
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    def preprocess_for_asr(self, audio: np.ndarray) -> np.ndarray:
        """智能预处理流水线（为Whisper等ASR模型优化）
        参考:
        - DENOASR (2024): 选择性降噪，保守处理
        - Yerevan (2024): 可控降噪强度，避免过度处理
        - "When Denoising Hurts" (2025): Whisper已噪声鲁棒，过度降噪反有害
        策略: 轻量预处理三件套，不引入降噪伪影
        """
        quality = estimate_speech_quality(audio, self.sample_rate)

        audio = self._highpass_filter(audio, cutoff=70)

        audio = self._rms_normalize(audio, target_db=-20)

        if quality["low_freq_ratio"] > 0.25:
            n_fft = 512
            spec = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4)
            mag = np.abs(spec)
            N = np.median(mag[:, :20], axis=1, keepdims=True) + 1e-8
            gain = np.maximum(1.0 - 0.4 * N / (mag + 1e-8), 0.15)
            spec_clean = spec * gain
            audio = librosa.istft(spec_clean, hop_length=n_fft // 4, length=len(audio))

        audio = self._rms_normalize(audio, target_db=-20)

        return audio.astype(np.float32)

    def _highpass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """保守高通滤波：仅去除极低频嗡声（<80Hz），不伤语音"""
        nyq = self.sample_rate / 2
        if cutoff >= nyq:
            return audio
        try:
            b, a = signal.butter(4, cutoff / nyq, btype='high')
            return signal.filtfilt(b, a, audio)
        except Exception:
            return audio

    def _rms_normalize(self, audio: np.ndarray, target_db: float = -20) -> np.ndarray:
        """RMS归一化：稳定音量，不削峰"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        audio = audio * gain
        peak = np.max(np.abs(audio))
        if peak > 0.98:
            audio = audio * 0.98 / peak
        return audio