#!/usr/bin/env bash
set -euo pipefail

# Launch a local (single-host, non-SkyPilot) N-way `nova load` fleet against
# one loader config: prepare once, run WORKER_COUNT workers in parallel (each
# a distinct --job-rank), then finalize once.
#
# Ported from qdrant-fineweb-gte-10b's scripts/run_ten_billion_upsert_10way.sh:
# that script partitioned a sorted catalog by row range (UPLOAD_OFFSET/MAX_POINTS)
# per worker. nova-load partitions by deterministic file stride instead
# (--num-jobs/--job-rank; see crates/nova-load/src/plan.rs), so there's no row
# math to compute here.
#
# Usage:
#   LOAD_CONFIG=configs/loader/fineweb_10b_full.yaml \
#   ./scripts/run_fineweb_10b_local_fleet.sh
#
# Optional:
#   WORKER_COUNT=10          (default 10)
#   RESUME=false             (default false; pass true to add --resume to each worker)
#   RUN_IN_FOREGROUND=false  (default: background workers + wait)

: "${LOAD_CONFIG:?LOAD_CONFIG is required (path to a nova-load YAML config)}"
: "${WORKER_COUNT:=10}"
: "${RESUME:=false}"
: "${RUN_IN_FOREGROUND:=false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

resume_flag=()
if [[ "${RESUME}" == "true" ]]; then
  resume_flag=(--resume)
fi

echo "Local fleet:"
echo "  LOAD_CONFIG=${LOAD_CONFIG}"
echo "  WORKER_COUNT=${WORKER_COUNT}"
echo "  RESUME=${RESUME}"
echo "  RUN_IN_FOREGROUND=${RUN_IN_FOREGROUND}"

echo "Preparing collection (create + defer indexing)..."
nova load prepare "${LOAD_CONFIG}"

mkdir -p "${ROOT_DIR}/runs"

pids=()
logs=()
for ((worker_idx = 0; worker_idx < WORKER_COUNT; worker_idx++)); do
  log_file="${ROOT_DIR}/runs/load_worker_${worker_idx}.log"
  logs+=("${log_file}")

  echo "Worker ${worker_idx}: --num-jobs ${WORKER_COUNT} --job-rank ${worker_idx} -> ${log_file}"

  (
    cd "${ROOT_DIR}"
    exec nova load load "${LOAD_CONFIG}" \
      --num-jobs "${WORKER_COUNT}" \
      --job-rank "${worker_idx}" \
      "${resume_flag[@]}"
  ) >"${log_file}" 2>&1 &

  pids+=("$!")
  if [[ "${RUN_IN_FOREGROUND}" == "true" ]]; then
    wait "${pids[-1]}"
  fi
done

if [[ "${RUN_IN_FOREGROUND}" != "true" ]]; then
  echo "Launched ${WORKER_COUNT} workers (PIDs: ${pids[*]})."
  echo "Tail logs: tail -f ${logs[*]}"
  wait "${pids[@]}"
fi

echo "All ${WORKER_COUNT} workers finished."
echo "Finalizing (re-enable indexing, wait for it to settle)..."
nova load finalize "${LOAD_CONFIG}"
echo "Fleet load complete."
