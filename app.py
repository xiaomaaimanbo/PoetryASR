"""
古诗词语音识别系统 — 桌面应用启动器
双击此文件或 poetry-asr.exe 直接打开 Web 界面
"""

import os
import sys
from pathlib import Path

APP_TITLE = "古诗词语音识别系统"
APP_PORT = 7860
APP_URL = f"http://localhost:{APP_PORT}"


def _get_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _show_splash():
    from src.splash import SplashScreen
    return SplashScreen(APP_TITLE, "正在加载 AI 模型，请稍候...")


def main():
    root = _get_root()
    os.chdir(str(root))

    print(f"  {APP_TITLE}  v2.0")
    print(f"  正在加载模型，弹窗显示进度...")
    print()

    splash = _show_splash()

    try:
        splash.set_status("正在加载语音识别引擎...")
        from webui import demo

        splash.set_status("正在加载诗词纠错模块...")
        import importlib
        importlib.import_module("src.poetry_corrector")

        splash.set_status("启动 Web 服务...")

    finally:
        splash.close()

    print(f"  服务器启动中: {APP_URL}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=APP_PORT,
        inbrowser=True,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
