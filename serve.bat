@echo off
rem Starts the OpenAI-compatible server of Gemma 4 31B for several users (scripts/serve.py) on
rem http://127.0.0.1:8001/v1. Options are passed on (--port, --host, --kv-gib, --context, --think; -h lists them).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
..\.venv\Scripts\python.exe -u -m scripts.serve %*
