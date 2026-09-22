#!/usr/bin/env bash
set -euo pipefail

# Staggered closed-loop storm fleet for max-RPS discovery on FineWeb 10B.
# Workers join one at a time (default 60s apart) so offered load ramps up
# instead of hitting Qdrant with full fleet concurrency immediately.
#
# Usage:
#   export QDRANT_URL=http://<host>:6334
#   export QDRANT_API_KEY=...
#   ./scripts/run_fineweb_storm_ramped_fleet.sh
#
# GCS access for gs:// query parquet is handled by the SkyPilot server / GCP
# worker identity — local AWS HMAC keys are not required when using GCP workers.
#
# Optional:
#   STORM_CONFIG=configs/storm/fineweb_bf_k1000_fleet_ramped.yaml
#   RESOURCES=configs/skypilot/storm-gcp-me-west1.yaml
#   POOL_NAME=nova-storm-fineweb-bf-k1000
#   WORKER_COUNT=10
#   STAGGER_S=60              seconds between worker launches
#   FULL_FLEET_MIN=20         minutes all workers run together at steady state
#   STAGE_CONFIG=true         blue-green pool apply before launch (new config on workers)
#   DRY_RUN=true              print plan only
#   COLLECT=true              wait for jobs and print per-worker + aggregate RPS
#   SKIP_LAUNCH=true          only stage pool / collect (no new launches)

: "${STORM_CONFIG:=configs/storm/fineweb_bf_k1000_fleet_ramped.yaml}"
: "${RESOURCES:=configs/skypilot/storm-gcp-me-west1.yaml}"
RESOURCES_EFFECTIVE="${RESOURCES}"
: "${POOL_NAME:=nova-storm-fineweb-bf-k1000}"
: "${WORKER_COUNT:=10}"
: "${STAGGER_S:=60}"
: "${FULL_FLEET_MIN:=20}"
: "${STAGE_CONFIG:=false}"
: "${DRY_RUN:=false}"
: "${COLLECT:=false}"
: "${SKIP_LAUNCH:=false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: ${name} must be set" >&2
    exit 1
  fi
}

require_env QDRANT_URL

using_gcp_workers() {
  rg -q 'cloud:[[:space:]]*gcp' "${RESOURCES}" 2>/dev/null
}

# Full-fleet plateau = duration_s - (WORKER_COUNT - 1) * STAGGER_S
DURATION_S=$((FULL_FLEET_MIN * 60 + (WORKER_COUNT - 1) * STAGGER_S))
RUN_CONFIG=""

prepare_run_config() {
  local base out_dir
  base="$(basename "${STORM_CONFIG}")"
  out_dir="${ROOT_DIR}/configs/storm/.generated"
  mkdir -p "${out_dir}"
  RUN_CONFIG="${out_dir}/${base}"
  sed "s/^  duration_s:.*/  duration_s: ${DURATION_S}/" "${STORM_CONFIG}" > "${RUN_CONFIG}"
}

prepare_resources() {
  local bin="${ROOT_DIR}/target/release/nova-storm"
  if [[ ! -f "${bin}" ]]; then
    RESOURCES_EFFECTIVE="${RESOURCES}"
    return
  fi
  local out="${ROOT_DIR}/configs/storm/.generated/storm-gcp-resources.yaml"
  python3 - "${RESOURCES}" "${bin}" "${out}" <<'PY'
import sys, yaml
base_path, bin_path, out_path = sys.argv[1:4]
with open(base_path) as f:
    spec = yaml.safe_load(f) or {}
spec["file_mounts"] = {"/cfg/nova-storm-bin": bin_path}
with open(out_path, "w") as f:
    yaml.dump(spec, f, sort_keys=False)
PY
  RESOURCES_EFFECTIVE="${out}"
}

dist_cmd() {
  if command -v nova >/dev/null 2>&1; then
    nova dist "$@"
  elif command -v nova-dist >/dev/null 2>&1; then
    nova-dist "$@"
  else
    echo "error: nova or nova-dist not found on PATH" >&2
    exit 1
  fi
}

parse_staged_dir() {
  awk '/^staged: / { print $2; exit }'
}

parse_job_yaml() {
  awk '/^  job: / { print $2; exit }'
}

wait_pool_ready() {
  local want="$1"
  echo "Waiting for pool ${POOL_NAME} to have at least ${want} READY workers..."
  for _ in $(seq 1 90); do
    local ready
    ready="$(sky jobs pool status "${POOL_NAME}" 2>/dev/null | awk -v pool="${POOL_NAME}" '
      $1 == pool { split($NF, a, "/"); print a[1]; exit }
    ')"
    if [[ -n "${ready}" && "${ready}" -ge "${want}" ]]; then
      echo "Pool ready (${ready} workers)."
      return 0
    fi
    sleep 10
  done
  echo "error: pool ${POOL_NAME} did not reach ${want} READY workers within timeout" >&2
  sky jobs pool status "${POOL_NAME}" || true
  exit 1
}

wait_pool_jobs_done() {
  echo "Waiting for pool ${POOL_NAME} jobs to finish..."
  for _ in $(seq 1 360); do
    local running
    running="$(sky jobs queue 2>/dev/null | awk -v pool="${POOL_NAME}" '
      NR > 3 && index($0, pool) && $(NF-1) == "RUNNING" { c++ }
      END { print c + 0 }
    ')"
    if [[ "${running}" -eq 0 ]]; then
      echo "All pool jobs finished."
      return 0
    fi
    sleep 10
  done
  echo "error: timed out waiting for pool ${POOL_NAME} jobs" >&2
  exit 1
}

collect_rps() {
  local -a job_ids=("$@")
  echo
  echo "=== Per-worker RPS ==="
  local total=0
  local count=0
  for job_id in "${job_ids[@]}"; do
    local rps
    rps="$(sky jobs logs "${job_id}" 2>/dev/null | awk '/requests_per_sec:/ { print $2; exit }')"
    if [[ -n "${rps}" ]]; then
      printf "  job %-4s  %s RPS\n" "${job_id}" "${rps}"
      total="$(awk -v a="${total}" -v b="${rps}" 'BEGIN { printf "%.1f", a + b }')"
      count=$((count + 1))
    else
      printf "  job %-4s  (no summary in logs)\n" "${job_id}"
    fi
  done
  if [[ "${count}" -gt 0 ]]; then
    echo "  ─────────────────────────"
    printf "  aggregate (%d workers)  %.1f RPS\n" "${count}" "${total}"
  fi
}

launch_one_job() {
  local job_yaml="$1"
  local env_args=()
  for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
    AWS_REGION AWS_DEFAULT_REGION GOOGLE_APPLICATION_CREDENTIALS \
    QDRANT_URL QDRANT_API_KEY QDRANT_COLLECTION; do
    if [[ -n "${!var:-}" ]]; then
      env_args+=(--env "${var}=${!var}")
    fi
  done
  sky jobs launch -y -p "${POOL_NAME}" --num-jobs 1 "${env_args[@]}" "${job_yaml}"
}

prepare_run_config
prepare_resources

echo "Staggered storm max-RPS run:"
echo "  STORM_CONFIG=${STORM_CONFIG}"
echo "  RUN_CONFIG=${RUN_CONFIG} (duration_s=${DURATION_S})"
echo "  RESOURCES=${RESOURCES_EFFECTIVE}"
echo "  POOL_NAME=${POOL_NAME}"
echo "  WORKER_COUNT=${WORKER_COUNT}"
echo "  STAGGER_S=${STAGGER_S}"
echo "  FULL_FLEET_MIN=${FULL_FLEET_MIN} (plateau ${FULL_FLEET_MIN}m after last worker joins)"
echo "  STAGE_CONFIG=${STAGE_CONFIG}"
echo "  DRY_RUN=${DRY_RUN}"
echo "  COLLECT=${COLLECT}"

JOB_IDS=()

if [[ "${STAGE_CONFIG}" == "true" ]]; then
  echo
  echo "Staging config on pool (blue-green apply)..."
  stage_out="$(dist_cmd storm "${RUN_CONFIG}" \
    --resources "${RESOURCES_EFFECTIVE}" \
    --num-jobs "${WORKER_COUNT}" \
    --pool-name "${POOL_NAME}" \
    --dry-run)"
  echo "${stage_out}"
  run_dir="$(printf '%s\n' "${stage_out}" | parse_staged_dir)"
  pool_yaml="${run_dir}/pool.yaml"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry run] would run: sky jobs pool apply -p ${POOL_NAME} --mode blue_green ${pool_yaml}"
  else
    sky jobs pool apply -y -p "${POOL_NAME}" --mode blue_green "${pool_yaml}"
    wait_pool_ready "${WORKER_COUNT}"
  fi
fi

if [[ "${SKIP_LAUNCH}" != "true" ]]; then
  echo
  echo "Generating job YAML..."
  launch_plan="$(dist_cmd storm "${RUN_CONFIG}" \
    --resources "${RESOURCES_EFFECTIVE}" \
    --num-jobs 1 \
    --pool-name "${POOL_NAME}" \
    --jobs-only \
    --dry-run)"
  echo "${launch_plan}"
  run_dir="$(printf '%s\n' "${launch_plan}" | parse_staged_dir)"
  job_yaml="$(printf '%s\n' "${launch_plan}" | parse_job_yaml)"
  batch_name="$(date +%Y-%m-%dT%H-%M)_nova-storm-ramped"
  batch_job_yaml="${run_dir}/job_ramped.yaml"
  sed "s/^name:.*/name: ${batch_name}/" "${job_yaml}" > "${batch_job_yaml}"

  echo
  echo "Launching ${WORKER_COUNT} workers (${STAGGER_S}s stagger, closed-loop max RPS)..."
  echo "  job: ${batch_job_yaml}"
  for rank in $(seq 1 "${WORKER_COUNT}"); do
    echo
    echo "--- worker ${rank}/${WORKER_COUNT} ---"
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[dry run] would run: sky jobs launch -p ${POOL_NAME} --num-jobs 1 ${batch_job_yaml}"
    else
      launch_out="$(launch_one_job "${batch_job_yaml}" 2>&1 | tee /dev/stderr)"
      job_id="$(printf '%s\n' "${launch_out}" | awk '/Managed job submitted|job ID|Job ID/ { print $NF; exit }')"
      if [[ -z "${job_id}" ]]; then
        job_id="$(sky jobs queue 2>/dev/null | awk -v pool="${POOL_NAME}" '
          NR > 3 && index($0, pool) && $(NF-1) == "RUNNING" { print $1; exit }
        ')"
      fi
      if [[ -n "${job_id}" ]]; then
        JOB_IDS+=("${job_id}")
        echo "  tracked job id: ${job_id}"
      fi
    fi
    if [[ "${rank}" -lt "${WORKER_COUNT}" && "${DRY_RUN}" != "true" ]]; then
      sleep "${STAGGER_S}"
    fi
  done
fi

if [[ "${COLLECT}" == "true" && "${DRY_RUN}" != "true" ]]; then
  wait_pool_jobs_done
  if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
    mapfile -t JOB_IDS < <(
      sky jobs queue -a 2>/dev/null | awk -v pool="${POOL_NAME}" -v n="${WORKER_COUNT}" '
        NR > 3 && index($0, pool) && $(NF-1) == "SUCCEEDED" { ids[++c] = $1 }
        END { for (i = c - n + 1; i <= c; i++) if (i > 0) print ids[i] }
      '
    )
  fi
  collect_rps "${JOB_IDS[@]}"
fi

if [[ "${DRY_RUN}" != "true" && "${SKIP_LAUNCH}" != "true" ]]; then
  echo
  echo "Monitor:  sky jobs queue"
  echo "Logs:     sky jobs logs <job-id>"
  echo "Collect:  COLLECT=true SKIP_LAUNCH=true ${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
fi
