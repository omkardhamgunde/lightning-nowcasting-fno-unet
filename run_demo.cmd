@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo.py %*
) else (
    python demo.py %*
)
exit /b %errorlevel%
