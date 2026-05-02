#!/usr/bin/env bash
# Run the finewiki configs locally for every language under a row-count threshold.
# Reads the manifest written by generate_finewiki_configs.py.
#
# Usage:
#   scripts/embed_small_finewiki_local.sh                    # default threshold: 10000
#   scripts/embed_small_finewiki_local.sh 5000               # override threshold
#   scripts/embed_small_finewiki_local.sh 10000 --dry-run    # list what would run, don't execute
#   scripts/embed_small_finewiki_local.sh --dry-run          # dry-run with default threshold
#
# Assumes AWS creds are already exported into the current shell.

set -u

DRY_RUN=0
THRESHOLD=10000
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) THRESHOLD="${arg}" ;;
  esac
done

CONFIG_DIR="configs/embedder/finewiki_gte_multilingual"
MANIFEST="${CONFIG_DIR}/_manifest.json"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Run: python scripts/generate_finewiki_configs.py" >&2
  exit 1
fi

# parse manifest; emit "<code> <rows>" lines for languages under threshold, smallest first
small_langs=$(python -c "
import json, sys
thr = ${THRESHOLD}
with open('${MANIFEST}') as f:
    m = json.load(f)
for lang in sorted(m['languages'], key=lambda x: x['rows']):
    if lang['rows'] < thr:
        print(f\"{lang['code']}\t{lang['rows']}\")
")

total=$(echo "${small_langs}" | wc -l | tr -d ' ')
total_rows=$(echo "${small_langs}" | awk -F'\t' '{sum+=$2} END {print sum}')

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] ${total} languages under ${THRESHOLD} rows (sum: ${total_rows} rows)"
  echo
  printf '%-12s %12s  %s\n' "code" "rows" "config"
  printf '%-12s %12s  %s\n' "----" "----" "------"
  while IFS=$'\t' read -r code rows; do
    cfg="${CONFIG_DIR}/${code}.yaml"
    tag=""
    [[ ! -f "${cfg}" ]] && tag="  [MISSING]"
    printf '%-12s %12s  %s%s\n' "${code}" "${rows}" "${cfg}" "${tag}"
  done <<< "${small_langs}"
  echo
  echo "[dry-run] No commands executed."
  exit 0
fi

echo "Found ${total} languages under ${THRESHOLD} rows (sum: ${total_rows} rows). Running sequentially..."
echo

fails=()
while IFS=$'\t' read -r code rows; do
  cfg="${CONFIG_DIR}/${code}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "SKIP ${code}: config missing"
    continue
  fi
  echo "=== [${code}] ${rows} rows ==="
  if vf embed "${cfg}"; then
    echo "=== [${code}] OK ==="
  else
    echo "=== [${code}] FAILED ==="
    fails+=("${code}")
  fi
  echo
done <<< "${small_langs}"

echo
echo "==============================="
echo "Done. ${#fails[@]} failures."
if [[ ${#fails[@]} -gt 0 ]]; then
  printf '  - %s\n' "${fails[@]}"
fi
