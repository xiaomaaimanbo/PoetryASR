@echo off
chcp 65001 >nul
title PoetryASR 一键安装

echo.
echo   ============================================
echo     古诗词语音识别系统 PoetryASR v2.0
echo     一键安装脚本
echo   ============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [错误] 未检测到 Python，请先安装 Python 3.10+
    echo   下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo   [1/3] 检测 Python 环境...
python --version
echo.

echo   [2/3] 安装依赖包 (可能需要几分钟)...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
if %errorlevel% neq 0 (
    echo   国内镜像失败，尝试官方源...
    pip install -r requirements.txt --quiet
)
echo.

echo   [3/3] 验证安装...
python -c "import whisper; print('  Whisper  OK')" 2>nul || echo   [警告] Whisper 未安装
python -c "import gradio; print('  Gradio  OK')" 2>nul || echo   [警告] Gradio 未安装
python -c "import zhconv; print('  zhconv  OK')" 2>nul || echo   [警告] zhconv 未安装
python -c "import pypinyin; print('  pypinyin OK')" 2>nul || echo   [警告] pypinyin 未安装
echo.

echo   ============================================
echo     安装完成！
echo.
echo     启动方式:
echo       - 双击运行 "启动.bat"
echo       - 命令行: python launcher.py webui
echo       - 命令行: python launcher.py transcribe 音频文件
echo   ============================================
echo.

pause
