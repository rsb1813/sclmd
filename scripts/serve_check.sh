#!/bin/sh
# One locked GPU call's worth: start scripts/serve.py on port 8012, wait for /health, run scripts/serve_bench.py
# --check against it, stop the server by the PID started here (and its interpreter child, found by its command line).
#   sh scripts/serve_check.sh [serve_bench options ...]      (server options: SERVE_ARGS; default --clients 1,4,8)
PY=/h/projects/sclm/.venv/Scripts/python.exe
LOG=logs/serve_check_server.log
PYTHONIOENCODING=utf-8 $PY -m scripts.serve --port 8012 $SERVE_ARGS > "$LOG" 2>&1 &
pid=$!
ok=0
for i in $(seq 1 60); do
    sleep 3
    if ! kill -0 $pid 2>/dev/null; then echo "[serve_check] the server ended"; break; fi
    if $PY -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8012/health', timeout=2).status == 200 else 1)" 2>/dev/null; then ok=1; break; fi
done
if [ $ok -eq 1 ]; then
    echo "[serve_check] server up after about $((i * 3)) s"
    if [ $# -eq 0 ]; then set -- --clients 1,4,8; fi
    PYTHONIOENCODING=utf-8 $PY scripts/serve_bench.py --url http://127.0.0.1:8012/v1 --check "$@"
    code=$?
else
    code=1
fi
kill $pid 2>/dev/null
sleep 3
kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null
wait $pid 2>/dev/null
for p in $(powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -like '*scripts.serve --port 8012*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r'); do
    echo "[serve_check] stopping the server's process $p"
    taskkill //PID "$p" //T //F
done
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
tail -3 "$LOG"
exit $code
