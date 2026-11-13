@echo off
rem Builds the probe of wider passes (engine\build\wide_probe.dll).
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
nvcc -O2 -std=c++20 -arch=sm_120 --shared -Xcompiler "/EHsc /O2 /utf-8" wide_probe.cu -o build\wide_probe.dll || exit /b 1
