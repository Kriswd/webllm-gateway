@echo off
chcp 65001 >nul 2>&1
setlocal

title DeepSeek Web 授权浏览器
cd /d "%~dp0"

set "AUTH_PROFILE=%CD%\.webai-gateway\chrome-auth-profile"
set "DEBUG_PORT=9222"
set "LOGIN_URL=https://chat.deepseek.com"

if defined CHROME_PATH if exist "%CHROME_PATH%" set "BROWSER=%CHROME_PATH%"
if not defined BROWSER if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "BROWSER=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"

if not defined BROWSER (
  echo 没有找到 Chrome 或 Edge。
  echo 你也可以在管理台点击“一键启动授权浏览器”让网关自动查找。
  pause
  exit /b 1
)

if not exist "%AUTH_PROFILE%" mkdir "%AUTH_PROFILE%"

echo.
echo 正在启动 DeepSeek 授权浏览器...
echo 登录地址: %LOGIN_URL%
echo 调试端口: http://127.0.0.1:%DEBUG_PORT%
echo.

start "" "%BROWSER%" --remote-debugging-port=%DEBUG_PORT% --user-data-dir="%AUTH_PROFILE%" --no-first-run --disable-default-apps "%LOGIN_URL%"

echo 浏览器已启动。登录完成后回到 WebLLM Gateway 控制台点击“重新捕获登录态”。
