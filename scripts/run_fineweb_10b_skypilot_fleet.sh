#!/usr/bin/env bash
set -euo pipefail

# Launch (or finalize) a `nova dist load` fleet on SkyPilot for one loader
# config. Companion to scripts/run_fineweb_10b_local_fleet.sh, which runs the
# same lifecycle as local background processes instead of a SkyPilot pool.
#
# Unlike the local fleet script, this cannot `wait` for workers: spot/fleet
# jobs aren't reliably awaitable from the launch call (see docs/distributed.md),
# so launch and finalize are two separate invocations of this script.
#
# Usage (launch — prepare + fan out workers):
#   LOAD_CONFIG=configs/loader/fineweb_10b_full.yaml \
#   WORKER_COUNT=7 \
#   ./scripts/run_fineweb_10b_skypilot_fleet.sh
#
# Usage (finalize — after `sky jobs queue` shows every worker finished):
#   LOAD_CONFIG=configs/loader/fineweb_10b_full.yaml \
#   FINALIZE=true \
#   ./scripts/run_fineweb_10b_skypilot_fleet.sh
#
# Optional (launch only):
#   RESOURCES=...              path to a SkyPilot YAML overriding resources/setup/envs
#   POOL_NAME=...              SkyPilot pool name (default: nova-load-<config stem>)
#   DRY_RUN=false              generate the pool/job YAMLs and print the plan; don't launch
#   CATALOG=...                local parquet catalog to stage to workers (sets FINEWEB_S3_CATALOG)
#   CATALOG_REMOTE_DIR=/catalog  remote mount dir for the staged catalog
#   BUILD_CATALOG_INPUT=...    build the catalog locally from this parquet root before launching
#   BUILD_CATALOG_OUTPUT=...   output path for BUILD_CATALOG_INPUT (required if that's set)
#   BUILD_CATALOG_RESUME=false resume the catalog build (only with BUILD_CATALOG_INPUT)

: "${LOAD_CONFIG:?LOAD_CONFIG is required (path to a nova-load YAML config)}"
: "${WORKER_COUNT:=10}"
: "${FINALIZE:=false}"
: "${DRY_RUN:=false}"
: "${RESOURCES:=}"
: "${POOL_NAME:=}"
: "${CATALOG:=}"
: "${CATALOG_REMOTE_DIR:=/catalog}"
: "${BUILD_CATALOG_INPUT:=}"
: "${BUILD_CATALOG_OUTPUT:=}"
: "${BUILD_CATALOG_RESUME:=false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${FINALIZE}" == "true" ]]; then
  echo "SkyPilot fleet finalize:"
  echo "  LOAD_CONFIG=${LOAD_CONFIG}"
  exec nova dist load "${LOAD_CONFIG}" --finalize
fi

if [[ -n "${BUILD_CATALOG_INPUT}" && -z "${BUILD_CATALOG_OUTPUT}" ]]; then
  echo "error: BUILD_CATALOG_OUTPUT is required when BUILD_CATALOG_INPUT is set" >&2
  exit 1
fi

args=(load "${LOAD_CONFIG}" --num-jobs "${WORKER_COUNT}")

[[ -n "${RESOURCES}" ]] && args+=(--resources "${RESOURCES}")
[[ -n "${POOL_NAME}" ]] && args+=(--pool-name "${POOL_NAME}")
[[ "${DRY_RUN}" == "true" ]] && args+=(--dry-run)
[[ -n "${CATALOG}" ]] && args+=(--catalog "${CATALOG}")
[[ -n "${CATALOG_REMOTE_DIR}" ]] && args+=(--catalog-remote-dir "${CATALOG_REMOTE_DIR}")
if [[ -n "${BUILD_CATALOG_INPUT}" ]]; then
  args+=(--build-catalog-input "${BUILD_CATALOG_INPUT}" --build-catalog-output "${BUILD_CATALOG_OUTPUT}")
  [[ "${BUILD_CATALOG_RESUME}" == "true" ]] && args+=(--build-catalog-resume)
fi

echo "SkyPilot fleet launch:"
echo "  LOAD_CONFIG=${LOAD_CONFIG}"
echo "  WORKER_COUNT=${WORKER_COUNT}"
echo "  RESOURCES=${RESOURCES:-<default>}"
echo "  POOL_NAME=${POOL_NAME:-<default>}"
echo "  DRY_RUN=${DRY_RUN}"
echo "  CATALOG=${CATALOG:-<unset>}"
[[ -n "${BUILD_CATALOG_INPUT}" ]] && echo "  BUILD_CATALOG_INPUT=${BUILD_CATALOG_INPUT} -> ${BUILD_CATALOG_OUTPUT} (resume=${BUILD_CATALOG_RESUME})"

nova dist "${args[@]}"

if [[ "${DRY_RUN}" != "true" ]]; then
  echo
  echo "Monitor: sky jobs queue"
  echo "When every worker finishes, finalize with:"
  echo "  LOAD_CONFIG=${LOAD_CONFIG} FINALIZE=true ${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
fi
