@echo off
chcp 65001 >nul
cd /d "%~dp0.."
"C:\Users\AssociatedQuartz\.venvs\ta-gpu\Scripts\python.exe" -I "%~dp0daily_job.py" %*
