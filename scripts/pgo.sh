#!/usr/bin/env bash
# Profile-Guided Optimization build.
#
# Measured gain over the default release build is modest (0-5% on
# aarch64 SIMD-bound kernels, ~5% on histogram fractals). The tight
# f64x4 escape loops already optimize well without runtime profile
# data; PGO mainly helps the branch-heavy histogram / search paths.
#
# Workflow:
#   1. Build an instrumented binary.
#   2. Run a representative training workload to emit .profraw files.
#   3. Rebuild with profile-use to produce the final optimized binary.
#
# Final binary lives at target/<triple>/release/fractal-wallpaper.
# Requires cargo-pgo and llvm-tools-preview (provided by the nix devshell).

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== PGO step 1/3: instrument build =="
cargo pgo build

BIN="$(cargo pgo info 2>/dev/null | grep -oE '/[^ ]*/target/[^ ]*/release/fractal-wallpaper' | head -1 \
    || echo "target/$(rustc -vV | awk '/host:/ {print $2}')/release/fractal-wallpaper")"

echo "== PGO step 2/3: training workload =="
TRAIN_DIR="$(mktemp -d)"
trap 'rm -rf "$TRAIN_DIR"' EXIT

for fractal in mandelbrot julia burning-ship tricorn phoenix newton; do
    "$BIN" --seed 42 --width 1280 --height 720 --max-iter 1000 \
        -o "$TRAIN_DIR/$fractal.png" "$fractal" >/dev/null
done
for fractal in flame buddhabrot strange-attractor; do
    "$BIN" --seed 42 --width 1280 --height 720 --max-iter 500 \
        --samples 5000000 -o "$TRAIN_DIR/$fractal.png" "$fractal" >/dev/null
done

echo "== PGO step 3/3: optimize build =="
cargo pgo optimize

echo
echo "Done. PGO binary: $BIN"
