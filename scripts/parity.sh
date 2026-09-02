#!/usr/bin/env bash
# Run the nova-bf parity harness (tests/parity) with a live Qdrant.
#
# Starts a throwaway Qdrant container if one isn't already reachable, runs the
# suite on every device this machine has, and tears the container down again.
# On a GPU box this is the same command — the suite picks the GPU up on its
# own; see python/nova-bf/tests/parity/README.md.
#
#   scripts/parity.sh                  # everything
#   scripts/parity.sh -k sparse        # extra args go straight to pytest
#
# Env:
#   QDRANT_URL              use an existing server instead of starting one
#   NOVA_BF_PARITY_DEVICES  cpu / cuda / cpu,cuda  (default: what's available)
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${PARITY_QDRANT_PORT:-6333}"
CONTAINER="nova-bf-parity-qdrant"
started=""

reachable() { curl -fsS -m 2 "$1/collections" >/dev/null 2>&1; }

if [[ -z "${QDRANT_URL:-}" ]]; then
  QDRANT_URL="http://localhost:${PORT}"
  if ! reachable "$QDRANT_URL"; then
    if ! command -v docker >/dev/null; then
      echo "no Qdrant at $QDRANT_URL and no docker to start one — the" >&2
      echo "live-engine tests will skip; the naive half still runs." >&2
    else
      echo "starting $CONTAINER on :$PORT"
      docker run -d --rm --name "$CONTAINER" -p "${PORT}:6333" qdrant/qdrant >/dev/null
      started=1
      for _ in $(seq 1 30); do reachable "$QDRANT_URL" && break; sleep 1; done
    fi
  fi
fi
export QDRANT_URL
if [[ -n "$started" ]]; then
  # NOT `exec` below: exec would replace this shell and the trap would never
  # run, leaving the container behind after every invocation.
  # shellcheck disable=SC2064
  trap "docker stop $CONTAINER >/dev/null 2>&1 || true" EXIT
fi

echo "QDRANT_URL=$QDRANT_URL  devices=${NOVA_BF_PARITY_DEVICES:-auto}"
cd python/nova-bf
status=0
python -m pytest tests/parity -q "$@" || status=$?
exit "$status"
