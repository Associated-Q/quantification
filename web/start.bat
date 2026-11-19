@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "VENV_PY=C:\Users\AssociatedQuartz\.venvs\ta-gpu\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    set "PY=python"
    echo [warn] venv python not found, falling back to system python
)

echo ============================================
echo   Technical Analysis Knowledge Base
echo   http://127.0.0.1:%PORT%
echo   Press Ctrl+C to stop
echo ============================================
echo.

if /i not "%NO_BROWSER%"=="1" start "" cmd /c "timeout /t 6 >nul & start "" http://127.0.0.1:%PORT%/?v=%RANDOM%%RANDOM%"

"%PY%" -m uvicorn app:app --host 127.0.0.1 --port %PORT%

echo.
echo [stopped] server exited with code %ERRORLEVEL%
pause
