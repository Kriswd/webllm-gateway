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

set "WEBAI2API_DIR=%WEBAI2API_SIDECAR_DIR%"
if "%WEBAI2API_DIR%"=="" set "WEBAI2API_DIR=%~dp0..\WebAI2API-sidecar"
if exist "%WEBAI2API_DIR%\" (
  pushd "%WEBAI2API_DIR%" >nul 2>&1
  set "WEBAI2API_DIR=!CD!"
  popd >nul 2>&1
)

set "WEBAI2API_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8500.*LISTENING"') do set "WEBAI2API_PID=%%a"
if defined WEBAI2API_PID (
  echo WebAI2API sidecar 已在运行，PID !WEBAI2API_PID!，端口 8500。
) else if exist "!WEBAI2API_DIR!\package.json" (
  if not exist ".webai-gateway\logs" mkdir ".webai-gateway\logs" >nul 2>&1
  echo 正在启动 WebAI2API sidecar...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$corepack=(Get-Command corepack.cmd -ErrorAction SilentlyContinue).Source;" ^
    "if (-not $corepack) { $corepack=(Get-Command corepack -ErrorAction SilentlyContinue).Source };" ^
    "if (-not $corepack) { throw '未找到 corepack，请先安装 Node.js/Corepack' };" ^
    "Start-Process -FilePath $corepack -ArgumentList @('pnpm','start') -WorkingDirectory '!WEBAI2API_DIR!' -RedirectStandardOutput '.webai-gateway\\logs\\webai2api-out.log' -RedirectStandardError '.webai-gateway\\logs\\webai2api-err.log' -WindowStyle Hidden"
  timeout /t 5 /nobreak >nul
  set "WEBAI2API_PID="
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8500.*LISTENING"') do set "WEBAI2API_PID=%%a"
  if defined WEBAI2API_PID (
    echo WebAI2API sidecar 已启动，PID !WEBAI2API_PID!。
  ) else (
    echo WebAI2API sidecar 正在启动或未启动成功，请稍后刷新；如仍不可用，请检查 .webai-gateway\logs\webai2api-err.log
  )
) else (
  echo 未找到 WebAI2API sidecar：!WEBAI2API_DIR!
  echo ChatGPT/Gemini/LMArena 等 WebAI2API 站点需要本地 sidecar，默认地址是 http://127.0.0.1:8500/v1
)

set "WEBAI_DS2API_EXE=%WEBAI_DEEPSEEK_DS2API_EXE%"
if "%WEBAI_DS2API_EXE%"=="" set "WEBAI_DS2API_EXE=%~dp0.tmp\ds2api\.tmp-bin\ds2api.exe"

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9331.*LISTENING"') do set "WEBAI_DS2API_PID=%%a"
if defined WEBAI_DS2API_PID (
  echo DeepSeek ds2api sidecar 已在运行，PID !WEBAI_DS2API_PID!，端口 9331。
) else if exist "%WEBAI_DS2API_EXE%" (
  if not exist ".webai-gateway\logs" mkdir ".webai-gateway\logs" >nul 2>&1
  echo 正在启动 DeepSeek ds2api sidecar...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$env:PORT='9331';" ^
    "$env:DS2API_CONFIG_JSON='{\"keys\":[],\"accounts\":[],\"runtime\":{\"global_max_inflight\":1,\"account_max_inflight\":1}}';" ^
    "$env:DS2API_ADMIN_KEY='local-dev-admin';" ^
    "$env:DS2API_CONFIG_PATH=(Join-Path '%~dp0' '.webai-gateway\\ds2api\\config.json');" ^
    "Start-Process -FilePath '%WEBAI_DS2API_EXE%' -WorkingDirectory '%~dp0' -RedirectStandardOutput '.webai-gateway\\logs\\ds2api-out.log' -RedirectStandardError '.webai-gateway\\logs\\ds2api-err.log' -WindowStyle Hidden"
  timeout /t 2 /nobreak >nul
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9331.*LISTENING"') do set "WEBAI_DS2API_PID=%%a"
  if defined WEBAI_DS2API_PID (
    echo DeepSeek ds2api sidecar 已启动，PID !WEBAI_DS2API_PID!。
  ) else (
    echo DeepSeek ds2api sidecar 未启动成功，请检查 .webai-gateway\logs\ds2api-err.log
  )
) else (
  echo 未找到 DeepSeek ds2api sidecar：%WEBAI_DS2API_EXE%
  echo DeepSeek Web 需要本地 ds2api sidecar，默认地址是 http://127.0.0.1:9331/v1
)

python -m webai_gateway

if !ERRORLEVEL! NEQ 0 (
  echo.
  echo WebAI Gateway 异常退出，错误码 !ERRORLEVEL!.
  pause
  exit /b !ERRORLEVEL!
)
