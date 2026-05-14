# PoetryASR 右键菜单注册脚本
# 以管理员身份运行此脚本

$ErrorActionPreference = "Stop"

$appName = "PoetryASR"
$appPath = Join-Path $PSScriptRoot "..\app.py"
$iconPath = Join-Path $PSScriptRoot "..\assets\icon.ico"
$pythonExe = (Get-Command python -ErrorAction Stop).Source

# 文件的右键菜单
$fileRegPath = "Registry::HKEY_CLASSES_ROOT\*\shell\$appName"
if (-not (Test-Path $fileRegPath)) {
    New-Item -Path $fileRegPath -Force | Out-Null
}
Set-ItemProperty -Path $fileRegPath -Name "(Default)" -Value "使用 PoetryASR 识别"
New-Item -Path "$fileRegPath\command" -Force | Out-Null
Set-ItemProperty -Path "$fileRegPath\command" -Name "(Default)" -Value "`"$pythonExe`" `"$appPath`" --file `"%1`""

# 文件夹的右键菜单
$dirRegPath = "Registry::HKEY_CLASSES_ROOT\Directory\shell\$appName"
if (-not (Test-Path $dirRegPath)) {
    New-Item -Path $dirRegPath -Force | Out-Null
}
Set-ItemProperty -Path $dirRegPath -Name "(Default)" -Value "使用 PoetryASR 识别目录"
New-Item -Path "$dirRegPath\command" -Force | Out-Null
Set-ItemProperty -Path "$dirRegPath\command" -Name "(Default)" -Value "`"$pythonExe`" `"$appPath`" --dir `"%1`""

Write-Host "右键菜单已注册" -ForegroundColor Green

# 卸载右键菜单
function Unregister-ContextMenu {
    Remove-Item -Path $fileRegPath -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $dirRegPath -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "右键菜单已移除" -ForegroundColor Yellow
}
