"""
实时语音识别模块
支持麦克风实时音频捕获和语音转文字
"""

import numpy as np
import pyaudio
import torch
import whisper
import logging
from typing import Optional, Callable
import threading
import time

from .logging_config import get_logger

log = get_logger("realtime_recognizer")


class RealTimeRecognizer:
    """实时语音识别器"""
    
    def __init__(self, model_name: str = "base", language: str = "zh", device: str = "cuda"):
        self.model_name = model_name
        self.language = language
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.audio_queue = []
        self.is_recording = False
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # 音频参数
        self.SAMPLE_RATE = 16000
        self.CHUNK = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32
        
        self._load_model()
    
    def _load_model(self):
        """加载Whisper模型"""
        log.info("Loading Whisper %s model for real-time recognition...", self.model_name)
        self.model = whisper.load_model(self.model_name).to(self.device)
        log.info("Model loaded successfully on %s", self.device)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer = np.append(self.audio_buffer, audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self, callback: Optional[Callable[[str], None]] = None):
        """
        开始实时录音和识别
        :param callback: 识别结果回调函数，接收识别文本
        """
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)
        
        p = pyaudio.PyAudio()
        
        # 打开音频流
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        log.info("开始录音，按 Ctrl+C 停止...")
        
        try:
            # 定期处理音频缓冲区
            while self.is_recording:
                # 当缓冲区积累了足够的音频（约2秒）
                if len(self.audio_buffer) >= self.SAMPLE_RATE * 2:
                    # 提取并清空缓冲区
                    audio_chunk = self.audio_buffer[:self.SAMPLE_RATE * 3]
                    self.audio_buffer = self.audio_buffer[self.SAMPLE_RATE * 2:]
                    
                    # 在单独线程中进行识别
                    if callback:
                        threading.Thread(
                            target=self._recognize_chunk,
                            args=(audio_chunk, callback)
                        ).start()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("停止录音")
        finally:
            self.is_recording = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 处理剩余的音频
            if len(self.audio_buffer) > 0:
                log.info("处理剩余音频...")
                self._recognize_chunk(self.audio_buffer, callback)
    
    def _recognize_chunk(self, audio_data: np.ndarray, callback: Optional[Callable[[str], None]] = None):
        """识别音频片段"""
        try:
            # 归一化音频数据
            audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
            
            # 使用Whisper进行识别（优化参数提升准确率）
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                fp16=torch.cuda.is_available(),
                verbose=False,
                temperature=0.0,
                best_of=5,
                beam_size=5,
                word_timestamps=True
            )
            
            text = result.get("text", "").strip()
            
            if text and callback:
                callback(text)
            elif text:
                print(f"识别结果: {text}")
            
            return text
        except Exception as e:
            log.error("识别错误: %s", e)
            return None
    
    def recognize_microphone(self, duration: Optional[int] = None):
        """
        录制指定时长的音频并识别
        :param duration: 录制时长（秒），None表示无限录制（按Ctrl+C停止）
        :return: 识别结果文本
        """
        all_results = []
        
        def collect_result(text):
            all_results.append(text)
            print(f"[识别] {text}")
        
        if duration:
            print(f"录音 {duration} 秒...")
            timer = threading.Timer(duration, self.stop_recording)
            timer.start()
        
        self.start_recording(callback=collect_result)
        
        return " ".join(all_results)
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False


def realtime_transcribe(model_name: str = "base", language: str = "zh"):
    """便捷函数：启动实时语音识别"""
    recognizer = RealTimeRecognizer(model_name=model_name, language=language)
    try:
        recognizer.start_recording(callback=lambda text: print(f"[识别] {text}"))
    except KeyboardInterrupt:
        print("实时识别已停止")


if __name__ == "__main__":
    realtime_transcribe(model_name="base", language="zh")
