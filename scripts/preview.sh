#!/usr/bin/env bash
# Watch the newest model play one generated map (game view + its memory map + what it is "thinking").
# Usage: scripts/preview.sh [tier 1-4] [seed] [weights dir]
#   scripts/preview.sh            -> tier 3 (key -> locked door -> exit), random map, newest Mac model
#   scripts/preview.sh 1          -> tier 1 (fight arena)
#   scripts/preview.sh 3 123 submission/weights   -> the model that is currently submitted
set -euo pipefail
cd "$(dirname "$0")/.."
TIER="${1:-3}"
SEED="${2:-$((800000000 + RANDOM * 30 + TIER))}"
SRC="${3:-runs/mac_main/export}"
SNAP="runs/preview_snap"
rm -rf "$SNAP"; cp -r "$SRC" "$SNAP"
[ -f "$SNAP/odometry.json" ] || cp submission/weights/odometry.json "$SNAP/"
mkdir -p runs/videos
OUT="runs/videos/preview_T${TIER}_$(date +%H%M%S).mp4"
echo "recording tier $TIER map (seed $SEED) with $SRC ... (takes about 1 minute)"
PYTHONPATH=. uv run python -m training.viz.glassbox --weights "$SNAP" --tier "$TIER" --seed "$SEED" \
  --temperature 0.7 --out runs/videos/_raw.mp4 2>&1 | grep -v Warning || true
if command -v ffmpeg >/dev/null; then
  ffmpeg -y -loglevel error -i runs/videos/_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 26 "$OUT" && rm runs/videos/_raw.mp4
else
  mv runs/videos/_raw.mp4 "$OUT"
fi
echo "saved $OUT"
open "$OUT" 2>/dev/null || true
