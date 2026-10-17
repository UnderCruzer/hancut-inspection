#!/usr/bin/env bash
# Run with the GPU Python environment activated, inside tmux.
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${1:-$PWD/ml/runs/target_tiles_20261017}"
mkdir "$out"
exec > "$out/pipeline.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$out/pipeline.exit"; if [ "$code" -ne 0 ]; then printf "failed\n" > "$out/stage.txt"; fi' EXIT
printf 'inference\n' > "$out/stage.txt"
python -u ml/scripts/eval_tiled.py \
  --weights ml/runs/detect/e1_smoke/weights/best.pt \
  --out-dir "$out/inference" \
  --targets Lighter Bullet --fraction 0.6 --imgsz 640 --conf 0.001 \
  --fixed-conf 0.25 --nms-iou 0.7 --batch 8 --seed 0 \
  --splits calib test_hidden test_easy test_hard
printf 'e2\n' > "$out/stage.txt"
for arm in baseline tiled; do
  for cap in 0.01 0.02 0.05; do
    python -u ml/scripts/e2_compare.py --pred-dir "$out/inference/$arm" \
      --out-dir "$out/e2_${arm}_${cap}" --cap "$cap" --seed 0 --fraction 0.5
  done
done
printf 'complete\n' > "$out/stage.txt"
