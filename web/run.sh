#!/usr/bin/env bash
# Convenience launcher for the MARGO web demo.
# Starts the FastAPI backend (mock mode by default) and the Next.js dev server.
# Press Ctrl+C once to bring both down cleanly.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

pids=()
cleanup() {
  echo
  echo "→ stopping demo (pids=${pids[*]})"
  for p in "${pids[@]}"; do
    kill "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Bind to 0.0.0.0 by default so the demo is reachable from other machines
# (POSTECH lab servers, demo stations, etc.). Override with MARGO_HOST=127.0.0.1
# if you want strict localhost-only.
HOST="${MARGO_HOST:-0.0.0.0}"
FRONT_PORT="${MARGO_FRONT_PORT:-3000}"
BACK_PORT="${MARGO_BACK_PORT:-8001}"

echo "→ starting FastAPI on ${HOST}:${BACK_PORT} (mock fallback unless MARGO_PROCESSED_DIR is set)"
(
  cd "${ROOT}"
  python -m uvicorn web.backend.main:app --host "${HOST}" --port "${BACK_PORT}" --reload
) &
pids+=("$!")

sleep 1
echo "→ starting Next.js on ${HOST}:${FRONT_PORT}"
(
  cd "${HERE}/frontend"
  if [ ! -d node_modules ]; then npm install; fi
  # `npm run dev` already binds to 0.0.0.0 via the package.json script.
  HOST="${HOST}" npm run dev -- -H "${HOST}" -p "${FRONT_PORT}"
) &
pids+=("$!")

# Public-facing URL hint (works for the common case where users are looking
# at this from a different machine on the same network).
public_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "  Backend : http://${public_ip:-localhost}:${BACK_PORT}/api/health"
echo "  Frontend: http://${public_ip:-localhost}:${FRONT_PORT}"
echo "           (also http://localhost:${FRONT_PORT} on this machine)"
echo
wait
