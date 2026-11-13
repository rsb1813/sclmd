@echo off
rem Opens the terminal chat with Gemma 4 31B on the engine (scripts/chat.py).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
..\.venv\Scripts\python.exe -u -m scripts.chat %*
