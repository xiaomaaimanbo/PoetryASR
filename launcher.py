"""
古诗词语音识别系统 — 统一命令行入口

用法:
  poetry-asr transcribe <音频文件> [--model medium] [--no-poetry]
  poetry-asr webui [--port 7860]
  poetry-asr test
  poetry-asr info
"""

import argparse
import os
import sys
from datetime import datetime


def cmd_transcribe(args):
    from src.pipeline import TranscriptionPipeline
    from src.mimo_client import create_mimo_client

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        sys.exit(1)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"设备: {device}  |  模型: {args.model}")

    pipeline = TranscriptionPipeline(
        model_name=args.model,
        language=args.lang,
        device=device,
        enable_poetry=not args.no_poetry,
    )

    print(f"正在转录音频: {audio_path}")
    result = pipeline.run(audio_path)
    text = result.text

    if args.mimo:
        print("\nMiMo AI 精炼...")
        mimo = create_mimo_client()
        if mimo.available:
            refined = mimo.refine_transcription(text)
            if refined.get("model_used"):
                text = refined["refined"]
                print(f"   模型: {refined['model_used']}")

    print(f"\n{'=' * 50}")
    print(f"识别结果: {text}")

    if not args.no_poetry and result.matched_poetry:
        print(f"\n{'=' * 50}")
        print(f" 《{result.matched_poetry}》")
        print(f"   作者: {result.author}")
        print(f"   相似度: {result.similarity:.1%}")
        for seg in result.aligned_segments:
            status = "✓" if seg.get("is_matched") else "?"
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {status} {seg['text']}")
    elif not args.no_poetry:
        print("未匹配到诗词")

    if args.output or args.no_poetry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = args.output or f"results/{ts}.txt"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"识别文本: {text}\n\n")
            if result.matched_poetry:
                f.write(f"匹配诗词: 《{result.matched_poetry}》\n")
                for seg in result.aligned_segments:
                    f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n")
        print(f"\n结果已保存: {out}")


def cmd_webui(args):
    print(f"启动 Web 界面: http://localhost:{args.port}")
    print(f"正在加载模型...")

    from src.splash import SplashScreen
    splash = SplashScreen("古诗词语音识别系统", "正在加载 AI 模型，请稍候...")

    try:
        splash.set_status("正在加载语音识别引擎...")
        from webui import demo

        splash.set_status("正在加载诗词纠错模块...")
        import importlib
        importlib.import_module("src.poetry_corrector")

        splash.set_status("启动 Web 服务...")
    finally:
        splash.close()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        inbrowser=True,
        share=False,
    )


def cmd_test(args):
    import subprocess
    test_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, "-m", "pytest", os.path.join(test_dir, "tests"), "-v"])


def cmd_info(args):
    import torch
    import whisper
    from src.poetry_corrector import PoetryCorrector

    print("=" * 50)
    print("  古诗词语音识别系统  v2.0")
    print("=" * 50)
    print(f"  Python      {sys.version.split()[0]}")
    print(f"  PyTorch     {torch.__version__}")
    print(f"  CUDA        {'可用' if torch.cuda.is_available() else '不可用'}")
    if torch.cuda.is_available():
        print(f"  GPU         {torch.cuda.get_device_name(0)}")
    print(f"  Whisper     {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")
    print(f"  项目路径    {os.path.dirname(os.path.abspath(__file__))}")

    c = PoetryCorrector()
    titles = c.get_all_poetry_titles()
    print(f"  诗词库      {len(titles)} 首")
    print(f"  模型文件    models/")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        prog="poetry-asr",
        description="古诗词语音识别系统 - CLI & Web 界面"
    )
    parser.add_argument("--version", action="version", version="poetry-asr 2.0.0")

    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("transcribe", help="转录音频文件")
    p1.add_argument("audio", help="音频文件路径")
    p1.add_argument("--model", default="medium",
                    choices=["tiny", "base", "small", "medium", "large-v3"])
    p1.add_argument("--lang", default="zh", help="语言代码")
    p1.add_argument("--no-poetry", action="store_true", help="跳过诗词纠错")
    p1.add_argument("--mimo", action="store_true", help="启用 MiMo AI 精炼")
    p1.add_argument("--output", "-o", help="输出文件路径")

    p2 = sub.add_parser("webui", help="启动 Gradio Web 界面")
    p2.add_argument("--port", type=int, default=7860)

    sub.add_parser("test", help="运行单元测试")
    sub.add_parser("info", help="显示系统信息")

    args = parser.parse_args()

    if args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "webui":
        cmd_webui(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
