#!/usr/bin/env bash
# Nuke everything SkyPilot-related to stop AWS costs fast.
#
# Cancels all managed jobs, tears down all pools, tears down all sky clusters,
# optionally stops the jobs controller too.
#
# Usage:
#   scripts/sky-killswitch.sh            # full teardown except jobs controller
#   scripts/sky-killswitch.sh --all      # also stop the jobs controller

set -u  # don't set -e; we want best-effort teardown even if one step fails

STOP_CONTROLLER=0
if [[ "${1:-}" == "--all" ]]; then
  STOP_CONTROLLER=1
fi

echo "=== [1/4] Cancelling all managed jobs ==="
sky jobs cancel --all -y || true

echo
echo "=== [2/4] Listing + tearing down all jobs pools ==="
pools=$(sky jobs pool status --all 2>/dev/null \
  | awk '/^vf-|^sky-/ { print $1 }' \
  | sort -u)
if [[ -n "${pools}" ]]; then
  for pool in ${pools}; do
    echo "  -> sky jobs pool down ${pool}"
    sky jobs pool down "${pool}" -y || true
  done
else
  echo "  (no pools found)"
fi

echo
echo "=== [3/4] Tearing down all non-pool sky clusters ==="
# sky down --all targets everything in `sky status`
sky down --all -y || true

echo
echo "=== [4/4] Jobs controller ==="
if [[ ${STOP_CONTROLLER} -eq 1 ]]; then
  echo "  --all specified; stopping jobs controller"
  sky jobs controller stop -y || true
else
  echo "  left up (autostops on its own in ~10min)"
  echo "  re-run with --all to force-stop it now"
fi

echo
echo "=== Verify ==="
echo "sky status:"
sky status 2>/dev/null || true
echo
echo "sky jobs pool status --all:"
sky jobs pool status --all 2>/dev/null || true