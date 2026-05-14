@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo   ============================================
echo     古诗词语音识别系统 PoetryASR v2.0
echo   ============================================
echo.
echo   正在启动 Web 界面...
echo   浏览器将自动打开 http://localhost:7860
echo   按 Ctrl+C 可停止服务
echo.

start "" http://localhost:7860
python app.py

pause
