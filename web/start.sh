#!/usr/bin/env bash
# Start MARGO backend (FastAPI) and frontend (Next.js) as detached daemons.
#
#   bash web/start.sh           # both
#   bash web/start.sh backend   # backend only
#   bash web/start.sh frontend  # frontend only
#
# The processes survive an SSH disconnect or terminal close (nohup + disown +
# stdin redirected to /dev/null). PIDs are recorded in logs/*.pid so they
# can be cleanly stopped via `bash web/stop.sh`.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

HOST="${MARGO_HOST:-0.0.0.0}"
FRONT_PORT="${MARGO_FRONT_PORT:-3000}"
BACK_PORT="${MARGO_BACK_PORT:-8001}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Prefer the project venv if it exists; otherwise fall back to system python.
if [[ -x "${ROOT}/margo/bin/python" ]]; then
  PY="${ROOT}/margo/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

target="${1:-all}"

is_listening() {
  local port="$1"
  ss -lnt 2>/dev/null | awk -v p=":${port}$" '$4 ~ p {found=1} END { exit !found }'
}

start_backend() {
  if is_listening "${BACK_PORT}"; then
    echo "→ backend already listening on :${BACK_PORT}, skipping"
    return 0
  fi
  echo "→ starting backend on :${BACK_PORT}  (log: logs/web_backend.log)"
  cd "${ROOT}"
  nohup "${PY}" -m uvicorn web.backend.main:app \
    --host "${HOST}" --port "${BACK_PORT}" \
    > "${LOG_DIR}/web_backend.log" 2>&1 < /dev/null &
  echo $! > "${LOG_DIR}/web_backend.pid"
  disown || true
  echo "   pid=$(cat "${LOG_DIR}/web_backend.pid")"
}

start_frontend() {
  if is_listening "${FRONT_PORT}"; then
    echo "→ frontend already listening on :${FRONT_PORT}, skipping"
    return 0
  fi
  echo "→ starting frontend on :${FRONT_PORT}  (log: logs/web_frontend.log)"
  cd "${HERE}/frontend"
  if [[ ! -d node_modules ]]; then npm install; fi
  nohup npm run dev -- -H "${HOST}" -p "${FRONT_PORT}" \
    > "${LOG_DIR}/web_frontend.log" 2>&1 < /dev/null &
  echo $! > "${LOG_DIR}/web_frontend.pid"
  disown || true
  echo "   pid=$(cat "${LOG_DIR}/web_frontend.pid")"
}

case "${target}" in
  all)       start_backend; start_frontend ;;
  backend)   start_backend ;;
  frontend)  start_frontend ;;
  *) echo "usage: bash web/start.sh [all|backend|frontend]"; exit 2 ;;
esac

public_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "  Backend : http://${public_ip:-localhost}:${BACK_PORT}/api/health"
echo "  Frontend: http://${public_ip:-localhost}:${FRONT_PORT}"
echo "  Stop    : bash web/stop.sh   |  Status: bash web/status.sh"
