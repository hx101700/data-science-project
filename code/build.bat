@echo off
cd /d %~dp0
pyinstaller main.spec --clean --noconfirm
pause
