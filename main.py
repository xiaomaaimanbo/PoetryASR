import argparse
import yaml
import os
from src.pipeline import TranscriptionPipeline
from src.mimo_client import create_mimo_client
from src.logging_config import setup_logging


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="语音识别项目主入口")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["transcribe", "detect_lang", "fine_tune", "realtime"],
                        help="运行模式")
    parser.add_argument("--audio_path", type=str, help="音频文件路径")
    parser.add_argument("--output_path", type=str, default="results/transcription.txt",
                        help="输出文件路径")
    parser.add_argument("--correct", action="store_true", help="启用古诗词纠错功能")
    parser.add_argument("--mimo", action="store_true", help="启用 MiMo AI 后处理精炼")
    parser.add_argument("--train_data", type=str, help="训练数据目录")

    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("log_level", "INFO"),
                  config.get("logging", {}).get("log_dir"))

    if args.mode == "transcribe":
        if not args.audio_path:
            print("错误：转录模式需要提供 --audio_path 参数")
            return

        enable_poetry = args.correct or config.get("poetry", {}).get("auto_detect", False)

        pipeline = TranscriptionPipeline(
            model_name=config["model"]["name"],
            language=config["model"]["language"],
            device=config["model"]["device"],
            poetry_dir=config.get("poetry", {}).get("directory"),
            enable_poetry=enable_poetry,
        )

        print(f"正在转录音频: {args.audio_path}")
        result = pipeline.run(args.audio_path)
        text = result.text

        print(f"识别结果: {text}")

        # MiMo AI 后处理精炼
        if args.mimo:
            print("\n正在用 MiMo AI 精炼识别结果...")
            mimo = create_mimo_client()
            if mimo.available:
                refined = mimo.refine_transcription(text)
                if refined.get("model_used"):
                    text = refined["refined"]
                    print(f"MiMo 精炼完成 (模型: {refined['model_used']})")
                    print(f"   精炼文本: {text}")
                else:
                    print("MiMo API 未配置，跳过精炼 (设置 MIMO_API_KEY 环境变量)")
            else:
                print("MiMo API 未配置，跳过精炼 (设置 MIMO_API_KEY 环境变量)")

        # Save
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for seg in result.segments:
                f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
        print(f"详细转录结果已保存到: {args.output_path}")

        # Word-level timestamps
        if result.word_timestamps:
            print("\n词级时间戳:")
            for word_info in result.word_timestamps:
                print(f"[{word_info['start']:.2f}s - {word_info['end']:.2f}s] {word_info['word']}")

        # Poetry correction output
        if enable_poetry and result.matched_poetry:
            print(f"\n识别到诗词: 《{result.matched_poetry}》")
            print(f"   作者: {result.author}")
            print(f"   相似度: {result.similarity:.2%}")

            aligned_output = args.output_path.replace('.txt', '_aligned.txt')
            with open(aligned_output, 'w', encoding='utf-8') as f:
                f.write(f"识别文本: {text}\n")
                f.write(f"匹配诗词: 《{result.matched_poetry}》\n")
                f.write(f"作者: {result.author}\n")
                f.write(f"相似度: {result.similarity:.2%}\n")
                f.write("=" * 50 + "\n\n")

                if result.has_alignment:
                    f.write("【时间对齐结果】\n\n")
                    for seg in result.aligned_segments:
                        status = "✓" if seg['is_matched'] else "?"
                        f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {status} {seg['text']}\n")
                        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

            print(f"\n对齐结果已保存到: {aligned_output}")

            corrected_output = args.output_path.replace('.txt', '_corrected.txt')
            with open(corrected_output, 'w', encoding='utf-8') as f:
                f.write(result.corrected)
            print(f"纠正文本已保存到: {corrected_output}")
        elif enable_poetry:
            print("未匹配到已知古诗词，保持原识别结果")

    elif args.mode == "detect_lang":
        if not args.audio_path:
            print("错误：语言检测模式需要提供 --audio_path 参数")
            return

        from src.speech_recognizer import SpeechRecognizer
        recognizer = SpeechRecognizer(
            model_name=config["model"]["name"],
            language=config["model"]["language"],
            device=config["model"]["device"]
        )
        lang, confidence = recognizer.detect_language(args.audio_path)
        print(f"检测到语言: {lang} (置信度: {confidence:.4f})")

    elif args.mode == "fine_tune":
        if not args.train_data:
            print("错误：微调模式需要提供 --train_data 参数")
            return
        raise NotImplementedError(
            "微调功能需要连接 Hugging Face Hub，当前网络环境下不可用。"
            "建议使用预训练模型进行语音识别。"
        )

    elif args.mode == "realtime":
        rt_config = config.get("realtime", {})
        model_name = rt_config.get("model_name", config["model"]["name"])
        language = rt_config.get("language", config["model"]["language"])
        print(f"启动实时语音识别...（模型: {model_name}）")
        from src.realtime_recognizer import realtime_transcribe
        realtime_transcribe(model_name=model_name, language=language)


if __name__ == "__main__":
    main()
