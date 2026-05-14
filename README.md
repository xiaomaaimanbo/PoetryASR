# 语音识别项目

基于 OpenAI Whisper 的端到端语音识别项目，支持多语言语音转文字、语言检测、模型微调等功能。

## 项目结构

```
speech_recognition/
├── src/                    # 源代码目录
│   ├── __init__.py         # 模块导出
│   ├── speech_recognizer.py # 语音识别核心模块
│   ├── audio_processor.py   # 音频处理模块
│   └── data_loader.py       # 数据加载模块
├── data/                   # 数据目录
│   ├── train/              # 训练数据
│   └── test/               # 测试数据
├── models/                 # 模型保存目录
├── notebooks/              # Jupyter Notebook示例
├── main.py                 # 主入口脚本
├── test_recognition.py     # 测试脚本
├── config.yaml             # 配置文件
└── requirements.txt        # 依赖列表
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐GPU加速)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础语音识别

```bash
python main.py --mode transcribe --audio_path data/test/sample.wav --output_path results/output.txt
```

### 2. 语言检测

```bash
python main.py --mode detect_lang --audio_path data/test/sample.wav
```

### 3. 模型微调

```bash
python main.py --mode fine_tune --train_data data/train
```

## API 使用示例

### 语音识别

```python
from src.speech_recognizer import SpeechRecognizer

recognizer = SpeechRecognizer(model_name="base", language="zh")
result = recognizer.transcribe_audio("audio.wav")
text = recognizer.extract_text(result)
print(text)
```

### 音频处理

```python
from src.audio_processor import AudioProcessor

processor = AudioProcessor()
audio, sr = processor.load_audio("audio.wav")
audio = processor.normalize_audio(audio)
mel_spec = processor.compute_mel_spectrogram(audio)
```

### 数据加载

```python
from src.data_loader import SpeechDataLoader

loader = SpeechDataLoader("data/train")
loader.load_audio_files()
audio, sr = loader.get_audio_data(0)
```

## 配置说明

在 `config.yaml` 中可以配置：

- **模型设置**: 模型名称、语言、设备
- **训练设置**: 批次大小、学习率、训练轮数
- **数据设置**: 数据目录、采样率、最大时长
- **输出设置**: 模型保存目录、结果目录

## 支持的模型

| 模型名称 | 大小 | 性能 | 适用场景 |
|---------|------|------|---------|
| tiny | ~150MB | 快速，准确率较低 | 边缘设备 |
| base | ~1GB | 平衡 | 通用场景 |
| small | ~2GB | 较好 | 推荐 |
| medium | ~5GB | 很好 | 高准确率需求 |
| large | ~15GB | 最佳 | 专业场景 |

## 支持的语言

支持99种语言，包括中文、英语、日语、韩语等。默认语言为中文(zh)。

## 注意事项

1. 首次运行会自动下载模型，可能需要较长时间
2. GPU加速可大幅提升识别速度，建议使用CUDA
3. 支持的音频格式: WAV, MP3, FLAC, M4A等
4. 音频最长支持30秒，超出部分会被截断

## 许可证

MIT License