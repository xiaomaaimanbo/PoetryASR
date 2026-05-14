import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .audio_processor import AudioProcessor

class SpeechDataLoader:
    def __init__(self, data_dir: str, sample_rate: int = 16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.data = []

    def load_audio_files(self, file_extensions: List[str] = None) -> List[Dict]:
        if file_extensions is None:
            file_extensions = [".wav", ".mp3", ".flac", ".m4a"]
        
        self.data = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.data.append({
                        "file_path": file_path,
                        "label": label,
                        "file_name": file
                    })
        return self.data

    def load_from_csv(self, csv_path: str) -> List[Dict]:
        df = pd.read_csv(csv_path)
        self.data = df.to_dict("records")
        return self.data

    def load_from_json(self, json_path: str) -> List[Dict]:
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def get_audio_data(self, index: int) -> Tuple[np.ndarray, int]:
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        
        file_path = self.data[index]["file_path"]
        audio, sr = self.audio_processor.load_audio(file_path)
        return audio, sr

    def get_batch(self, indices: List[int]) -> Tuple[List[np.ndarray], List[str]]:
        audios = []
        labels = []
        
        for idx in indices:
            audio, _ = self.get_audio_data(idx)
            audio = self.audio_processor.normalize_audio(audio)
            audio = self.audio_processor.pad_or_truncate(audio)
            audios.append(audio)
            labels.append(self.data[idx].get("label", ""))
        
        return audios, labels

    def split_dataset(self, train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        import random
        shuffled_data = random.sample(self.data, len(self.data))
        split_idx = int(len(shuffled_data) * train_ratio)
        return shuffled_data[:split_idx], shuffled_data[split_idx:]

    def generate_data_list(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def create_data_info(self) -> Dict:
        info = {
            "total_files": len(self.data),
            "labels": list(set(item["label"] for item in self.data if "label" in item)),
            "sample_rate": self.sample_rate
        }
        return info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]