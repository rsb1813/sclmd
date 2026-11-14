@echo off
rem The GPU's waiting line (scripts/gpu.sh): who is on the card, who waits, and a hold.
rem   gpu          the job on the card and the waiting line
rem   gpu hold     no new job starts (the running one finishes; the line waits)
rem   gpu go       starts again
cd /d "%~dp0"
if "%~1"=="" (
    "C:\Program Files\Git\bin\sh.exe" scripts/gpu.sh status
) else (
    "C:\Program Files\Git\bin\sh.exe" scripts/gpu.sh %1
)
