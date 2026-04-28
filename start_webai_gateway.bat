@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title WebAI Gateway
cd /d "%~dp0"

if not exist config.json (
  copy config.example.json config.json >nul
)

echo.
echo ========================================
echo   WebAI Gateway
echo ========================================
echo.
echo   控制台:   http://127.0.0.1:8610/
echo   OpenAI API: http://127.0.0.1:8610/v1
echo   健康检查:  http://127.0.0.1:8610/health
echo.
echo   按 Ctrl+C 停止服务。
echo.

python -m webai_gateway

if !ERRORLEVEL! NEQ 0 (
  echo.
  echo WebAI Gateway 异常退出，错误码 !ERRORLEVEL!.
  pause
  exit /b !ERRORLEVEL!
)
