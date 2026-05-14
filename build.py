"""
PyInstaller 打包脚本
运行: python build.py
生成 dist/PoetryASR/ 目录
"""

import os, sys, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST, BUILD = ROOT / "dist", ROOT / "build"
APP_NAME = "PoetryASR"


def clean():
    for d in [DIST, BUILD]:
        if d.exists():
            shutil.rmtree(d)
    for p in ROOT.glob("*.spec"):
        p.unlink()


def run_pyinstaller():
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir", "--name", APP_NAME, "--console",
        "--clean", "--noconfirm",
        "--add-data", f"data{os.pathsep}data",
        "--add-data", f"config.yaml{os.pathsep}.",
        "--add-data", f"src{os.pathsep}src",
        "--add-data", f"webui.py{os.pathsep}.",
        "--hidden-import", "whisper",
        "--hidden-import", "pypinyin",
        "--hidden-import", "zhconv",
        "--hidden-import", "librosa",
        "--hidden-import", "soundfile",
        "--hidden-import", "gradio",
        "--hidden-import", "numpy",
        "--hidden-import", "scipy",
        "--hidden-import", "yaml",
        "--collect-all", "whisper",
        "--collect-all", "gradio",
        "--collect-all", "safehttpx",
        "--collect-all", "groovy",
        "--collect-all", "pypinyin",
        "app.py",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    print("=" * 50)
    print("  PoetryASR 打包构建")
    print("=" * 50)
    print("\n[1/3] 清理旧文件...")
    clean()
    print("\n[2/3] PyInstaller 打包...")
    run_pyinstaller()
    print(f"\n[3/3] 完成! 输出: {DIST / APP_NAME}")
    exe = DIST / APP_NAME / f"{APP_NAME}.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / 1048576
        print(f"  可执行文件: {exe} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
