#!/usr/bin/env bash
# Show the status of MARGO services (vLLM, backend, frontend) at a glance.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"
FRONT_PORT="${MARGO_FRONT_PORT:-3000}"
BACK_PORT="${MARGO_BACK_PORT:-8001}"
VLLM_PORT="${MARGO_VLLM_PORT:-8000}"

row() {
  local name="$1" port="$2" health_url="$3"
  local pid status
  pid="$(ss -lntp 2>/dev/null | grep -E ":${port}\b" | head -1 | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
  if [[ -z "${pid}" ]]; then
    printf "  %-10s  %-7s  %s\n" "${name}" "DOWN" "(:${port} not listening)"
    return
  fi
  if [[ -n "${health_url}" ]]; then
    status="$(curl -sf -m 3 "${health_url}" 2>/dev/null || true)"
    if [[ -n "${status}" ]]; then
      printf "  %-10s  %-7s  pid=%s  :${port}  %s\n" "${name}" "UP" "${pid}" "${status:0:60}"
    else
      printf "  %-10s  %-7s  pid=%s  :${port}  (port up, health failed)\n" "${name}" "PARTIAL" "${pid}"
    fi
  else
    printf "  %-10s  %-7s  pid=%s  :${port}\n" "${name}" "UP" "${pid}"
  fi
}

public_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "MARGO services:"
row "vLLM"     "${VLLM_PORT}"   "http://localhost:${VLLM_PORT}/v1/models"
row "backend"  "${BACK_PORT}"   "http://localhost:${BACK_PORT}/api/health"
row "frontend" "${FRONT_PORT}"  "http://localhost:${FRONT_PORT}"
echo
echo "  → http://${public_ip:-localhost}:${FRONT_PORT}"
