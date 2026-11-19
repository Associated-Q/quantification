@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Usage: start.bat [port]   (default 8000)
rem        set NO_BROWSER=1 to skip opening the browser

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "VENV_PY=C:\Users\AssociatedQuartz\.venvs\ta-gpu\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    set "PY=python"
    echo [warn] venv python not found, falling back to system python
)

"%PY%" -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [error] uvicorn/fastapi not installed in:
    echo         %PY%
    echo Install with:
    echo         "%PY%" -m pip install -r web\requirements.txt
    pause
    exit /b 1
)

if not exist "web\static\index.html" (
    echo [error] web\static\index.html not found - wrong working directory?
    pause
    exit /b 1
)

echo ============================================
echo   Technical Analysis Workbench
echo   http://127.0.0.1:%PORT%
echo   Press Ctrl+C to stop
echo ============================================
echo.

if /i not "%NO_BROWSER%"=="1" start "" cmd /c "timeout /t 4 >nul & start "" http://127.0.0.1:%PORT%/?v=%RANDOM%%RANDOM%"

pushd web
"%PY%" -m uvicorn app:app --host 127.0.0.1 --port %PORT%
set "RC=%ERRORLEVEL%"
popd

echo.
echo [stopped] server exited with code %RC%
pause
