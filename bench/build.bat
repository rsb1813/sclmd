@echo off
rem Builds the benchmarks into bench\build.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
nvcc -O2 -std=c++20 -arch=sm_120 -Xcompiler "/EHsc /O2 /utf-8" mma_gemm.cu -o build\mma_gemm.exe || exit /b 1
nvcc -O2 -std=c++20 -arch=sm_120 -Xptxas -v -Xcompiler "/EHsc /O2 /utf-8" mma_rate.cu -o build\mma_rate.exe || exit /b 1
nvcc -O2 -std=c++20 -arch=sm_120 -Xptxas -v -Xcompiler "/EHsc /O2 /utf-8" mma_s8.cu -o build\mma_s8.exe || exit /b 1
