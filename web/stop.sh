#!/usr/bin/env bash
# Stop MARGO backend/frontend daemons started by `web/start.sh`.
#
#   bash web/stop.sh            # both
#   bash web/stop.sh backend    # backend only
#   bash web/stop.sh frontend   # frontend only

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"
LOG_DIR="${ROOT}/logs"
FRONT_PORT="${MARGO_FRONT_PORT:-3000}"
BACK_PORT="${MARGO_BACK_PORT:-8001}"

kill_pid_file() {
  local name="$1" pid_file="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "→ stopping ${name} (pid=${pid})"
      kill "${pid}" 2>/dev/null || true
      # Give it a moment, then SIGKILL if still up.
      for _ in 1 2 3 4 5; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.5
      done
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  else
    echo "→ ${name}: no pid file"
  fi
}

kill_by_port() {
  local name="$1" port="$2"
  local pid
  pid="$(ss -lntp 2>/dev/null | grep -E ":${port}\b" | head -1 | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
  if [[ -n "${pid:-}" ]]; then
    echo "→ stopping ${name} listening on :${port} (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

target="${1:-all}"
case "${target}" in
  all|backend)
    kill_pid_file backend  "${LOG_DIR}/web_backend.pid"
    kill_by_port  backend  "${BACK_PORT}"
    ;;&
  all|frontend)
    kill_pid_file frontend "${LOG_DIR}/web_frontend.pid"
    kill_by_port  frontend "${FRONT_PORT}"
    # Also reap leftover Next.js child workers.
    pkill -f "next dev -H .* -p ${FRONT_PORT}" 2>/dev/null || true
    ;;
  *) echo "usage: bash web/stop.sh [all|backend|frontend]"; exit 2 ;;
esac

echo "done."
