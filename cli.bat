@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo   ============================================
echo     PoetryASR CLI 命令行工具
echo   ============================================
echo.

if "%1"=="" (
    echo   用法:
    echo     cli.bat info                   查看系统信息
    echo     cli.bat transcribe 音频文件     转录音频
    echo     cli.bat webui                   启动 Web 界面
    echo     cli.bat test                    运行测试
    echo.
    echo   示例:
    echo     cli.bat transcribe data\test\静夜思.m4a
    echo.
    pause
    exit /b
)

python launcher.py %*
pause
