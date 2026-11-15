@echo off
rem Starts the OpenAI-compatible server of Gemma 4 31B for several users (scripts/serve.py) on
rem http://127.0.0.1:8001/v1. Options are passed on (--port, --host, --kv-gib, --context, --think; -h lists them).
rem With --supervise among them, or supervise = true in serve.toml (or the file --config names): through
rem scripts/supervise.py, which starts the server again when it ends with an error and holds the port meanwhile
rem (logs\supervise.log; the server gets --crash-only: scripts/serve_ops.py).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set SCLMD_SUPERVISE=
set SCLMD_CONFIGURED=
for %%a in (%*) do if /i "%%~a"=="--supervise" set SCLMD_SUPERVISE=1
if exist serve.toml set SCLMD_CONFIGURED=1
echo.%* | findstr /i /c:"--config" >nul && set SCLMD_CONFIGURED=1
if not defined SCLMD_SUPERVISE if defined SCLMD_CONFIGURED for /f %%s in ('..\.venv\Scripts\python.exe -m scripts.supervise --wanted %*') do if "%%s"=="yes" set SCLMD_SUPERVISE=1
if not defined SCLMD_SUPERVISE ..\.venv\Scripts\python.exe -u -m scripts.serve %*
if defined SCLMD_SUPERVISE ..\.venv\Scripts\python.exe -u -m scripts.supervise %*
