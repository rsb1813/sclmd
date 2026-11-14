#!/bin/sh
# B8: one locked GPU call: start scripts/serve.py, wait for /health, run scripts/task_load_bench.py, stop the server by
# the PID started here.   sh scripts/b8_load.sh <tag> [task_load_bench options ...]   (server options: SERVE_ARGS)
PY=/h/projects/sclm/.venv/Scripts/python.exe
tag=$1
shift
PYTHONIOENCODING=utf-8 $PY -m scripts.serve --port 8011 $SERVE_ARGS > "data/task_load_bench_serve_$tag.log" 2>&1 &
pid=$!
echo "[b8] server pid $pid"
ok=0
for i in $(seq 1 60); do
    sleep 3
    if ! kill -0 $pid 2>/dev/null; then echo "[b8] the server ended"; break; fi
    if $PY -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=2).status == 200 else 1)" 2>/dev/null; then ok=1; break; fi
done
if [ $ok -eq 1 ]; then
    echo "[b8] server up after about $((i * 3)) s"
    PYTHONIOENCODING=utf-8 $PY scripts/task_load_bench.py --url http://127.0.0.1:8011/v1 --tag "$tag" "$@"
    code=$?
else
    code=1
fi
kill $pid 2>/dev/null
sleep 3
kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null
wait $pid 2>/dev/null
# the venv's python.exe may be a launcher with the real interpreter as its child: stop that child too, found by the
# command line this script gave it (port 8011), never by name or path
for p in $(powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -like '*scripts.serve --port 8011*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r'); do
    echo "[b8] stopping the server's process $p"
    taskkill //PID "$p" //T //F
done
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
tail -4 "data/task_load_bench_serve_$tag.log"
exit $code
