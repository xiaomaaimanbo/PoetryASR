"""
MiMo API 集成模块
支持 Xiaomi MiMo 大模型作为语音识别后处理后端
提供文本纠错、诗词补全、多模态理解等能力

MiMo API 兼容 OpenAI 格式，Base URL: https://token-plan-cn.xiaomimimo.com/v1
"""

import os
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class MimoConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("MIMO_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"
        )
    )
    model: str = "mimo-v2.5"
    max_tokens: int = 1024
    temperature: float = 0.3

    def is_configured(self) -> bool:
        return bool(self.api_key)


class MimoClient:
    """Xiaomi MiMo API 客户端"""

    def __init__(self, config: Optional[MimoConfig] = None):
        self.config = config or MimoConfig()
        self._client = None

    @property
    def available(self) -> bool:
        return self.config.is_configured()

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError(
                    "请安装 openai 包: pip install openai"
                )
        return self._client

    def _chat(self, messages: List[Dict], **kwargs) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_completion_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        return response.choices[0].message.content

    def refine_transcription(
        self, raw_text: str, context: Optional[str] = None
    ) -> Dict:
        """
        用 MiMo 精炼语音识别文本

        Args:
            raw_text: Whisper 原始识别文本
            context: 可选的上下文（如诗词标题）

        Returns:
            包含原始文本和精炼文本的字典
        """
        if not self.available:
            return {"original": raw_text, "refined": raw_text, "model_used": None}

        system_prompt = (
            "你是一个中文语音识别后处理助手。请修正以下语音识别文本中的错误，"
            "特别注意：同音字替换、标点修正、古诗词专用词汇。只输出修正后的文本，不要解释。"
        )

        user_content = f"请修正以下语音识别文本：\n{raw_text}"
        if context:
            user_content = f"这首诗的标题可能是《{context}》。\n{user_content}"

        try:
            refined = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            )
            return {
                "original": raw_text,
                "refined": refined.strip(),
                "model_used": self.config.model,
            }
        except Exception as e:
            return {
                "original": raw_text,
                "refined": raw_text,
                "model_used": None,
                "error": str(e),
            }

    def identify_poem(self, text: str) -> Dict:
        """
        用 MiMo 识别诗词标题和作者

        Args:
            text: 语音识别文本

        Returns:
            包含标题、作者、可信度的字典
        """
        if not self.available:
            return {"title": "", "author": "", "confidence": 0.0}

        prompt = (
            "请识别以下古诗词文本的标题和作者。"
            "以JSON格式回复: {\"title\": \"诗词标题\", \"author\": \"作者\", \"confidence\": 0.0-1.0}\n\n"
            f"文本:\n{text}"
        )

        try:
            result = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return json.loads(result)
        except Exception:
            return {"title": "", "author": "", "confidence": 0.0}

    def align_sentences(
        self, raw_text: str, standard_text: str
    ) -> List[Dict]:
        """
        用 MiMo 辅助句子对齐

        Args:
            raw_text: 识别文本
            standard_text: 标准诗词文本

        Returns:
            对齐结果列表
        """
        if not self.available:
            return []

        prompt = (
            '请将以下「识别文本」与「标准文本」逐句对齐。'
            '标准文本的每个句子，在识别文本中找到最匹配的片段。'
            '以JSON数组格式回复，每个元素包含: '
            '{"standard_sentence": "...", "matched_segment": "...", "score": 0.0-1.0}\n\n'
            f'识别文本:\n{raw_text}\n\n标准文本:\n{standard_text}'
        )

        try:
            result = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            return json.loads(result)
        except Exception:
            return []

    def analyze_audio_scene(self, text: str) -> Dict:
        """
        用 MiMo-V2.5 多模态能力分析音频场景
        （配合音频描述使用）

        Args:
            text: 识别文本内容

        Returns:
            场景分析结果
        """
        if not self.available:
            return {"scene": "未知"}

        prompt = (
            "根据以下语音识别文本，推断这段音频的场景类型和说话人特征。"
            "以JSON格式回复: "
            '{"scene": "诗词朗读/日常对话/演讲/其他", '
            '"speaker_count": 数字, "mood": "情绪描述"}\n\n'
            f"文本:\n{text}"
        )

        try:
            result = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return json.loads(result)
        except Exception:
            return {"scene": "未知"}


def create_mimo_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "mimo-v2.5",
) -> MimoClient:
    config = MimoConfig(model=model)
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    return MimoClient(config)


if __name__ == "__main__":
    print("MiMo API 集成模块加载成功")
    client = MimoClient()
    if client.available:
        print(f"[OK] MiMo API 已配置 (模型: {client.config.model})")
    else:
        print("[!] 未检测到 MIMO_API_KEY 环境变量，MiMo 功能禁用")
        print("   设置方法: set MIMO_API_KEY=your-api-key")
