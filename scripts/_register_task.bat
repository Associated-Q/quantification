@echo off
chcp 65001 >nul
schtasks /create /tn "A股每日复盘" /tr "C:\Users\AssociatedQuartz\Desktop\技术分析\scripts\daily_job.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:35 /f
