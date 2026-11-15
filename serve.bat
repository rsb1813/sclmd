@echo off
rem Starts the OpenAI-compatible server of Gemma 4 31B for several users (scripts/serve.py) on
rem http://127.0.0.1:8001/v1. Options are passed on (--port, --host, --kv-gib, --context, --think; -h lists them).
rem With --supervise among them: through scripts/supervise.py, which starts the server again when it ends with an
rem error and holds the port meanwhile (logs\supervise.log; the server gets --crash-only: scripts/serve_ops.py).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set SCLMD_SUPERVISE=
for %%a in (%*) do if /i "%%~a"=="--supervise" set SCLMD_SUPERVISE=1
if not defined SCLMD_SUPERVISE ..\.venv\Scripts\python.exe -u -m scripts.serve %*
if defined SCLMD_SUPERVISE ..\.venv\Scripts\python.exe -u -m scripts.supervise %*
