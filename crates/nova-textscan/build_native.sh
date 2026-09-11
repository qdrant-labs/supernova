#!/usr/bin/env bash
# Build `nova_textscan` and drop it where a given interpreter can import it.
#
# maturin is NOT required: the crate is built `abi3-py310`, so one `.so` works
# for every CPython >= 3.10 and the whole install is a copy. That matters on a
# node whose venv is uv-managed and has no pip.
#
#   usage: build_native.sh <python> [dest-dir]
#
# `dest-dir` defaults to that interpreter's site-packages. The crate builds
# standalone (its version is not inherited), so this works both inside the
# supernova workspace and from a bare copy of `crates/nova-textscan`.
set -euo pipefail
PY="${1:?usage: build_native.sh <python> [dest-dir]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${2:-$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')}"
command -v cargo >/dev/null || {
  echo "cargo is not on PATH. Install it with:" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
  echo "  . \$HOME/.cargo/env" >&2
  exit 1
}
cargo build --profile release-textscan --manifest-path "$HERE/Cargo.toml"
# The artifact lands in the workspace target dir when there is one and in the
# crate's own otherwise; ask cargo rather than guessing.
TARGET="$(cargo metadata --format-version 1 --no-deps \
          --manifest-path "$HERE/Cargo.toml" | sed 's/.*"target_directory":"\([^"]*\)".*/\1/')"
SO="$TARGET/release-textscan/libnova_textscan.so"
[ -f "$SO" ] || { echo "built, but $SO is missing" >&2; exit 1; }
cp -f "$SO" "$DEST/nova_textscan.abi3.so"
"$PY" -c "import nova_textscan; print('nova_textscan', nova_textscan.version(), 'installed in $DEST')"
