@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title WebLLM Gateway
cd /d "%~dp0"

if not exist config.json (
  copy config.example.json config.json >nul
)

echo.
echo ========================================
echo   WebLLM Gateway
echo ========================================
echo.
echo   控制台:     http://127.0.0.1:8610/
echo   OpenAI API: http://127.0.0.1:8610/v1
echo   健康检查:   http://127.0.0.1:8610/health
echo.
echo   启动内部网页登录 runtime...
echo.

rem Runtime supervisor starts WebAI2API and ds2api; ds2api concurrency comes from webai_gateway.ds2api_sidecar_config.
python -m webai_gateway.runtime_supervisor --config config.json --ensure

if !ERRORLEVEL! NEQ 0 (
  echo.
  echo 内部 runtime supervisor 执行失败，错误码 !ERRORLEVEL!。
  pause
  exit /b !ERRORLEVEL!
)

echo.
echo   Gateway 已统一托管内部 runtime。按 Ctrl+C 停止前台服务。
echo.

python -m webai_gateway

if !ERRORLEVEL! NEQ 0 (
  echo.
  echo WebLLM Gateway 异常退出，错误码 !ERRORLEVEL!。
  pause
  exit /b !ERRORLEVEL!
)
