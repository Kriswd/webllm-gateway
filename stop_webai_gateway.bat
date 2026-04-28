@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title Stop WebAI Gateway

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8610.*LISTENING"') do (
  echo Stopping WebAI Gateway PID %%a...
  taskkill /F /PID %%a >nul 2>&1
)

echo Done.
pause >nul 2>&1
