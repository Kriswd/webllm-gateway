@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title Stop WebLLM Gateway

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8610.*LISTENING"') do (
  echo Stopping WebLLM Gateway PID %%a...
  taskkill /F /PID %%a >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9331.*LISTENING"') do (
  for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "(Get-Process -Id %%a -ErrorAction SilentlyContinue).ProcessName"`) do (
    if /I "%%p"=="ds2api" (
      echo Stopping DeepSeek ds2api PID %%a...
      taskkill /F /PID %%a >nul 2>&1
    )
  )
)

echo Done.
pause >nul 2>&1
