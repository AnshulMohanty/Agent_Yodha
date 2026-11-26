#!/usr/bin/env bash
# Upload a checkpoint as a GitHub release asset so Kaggle/Codespaces can download it.
# Usage: scripts/publish_ckpt.sh <tag> <path/to/latest.pt>     e.g. scripts/publish_ckpt.sh ckpt-A runs/mac_main/latest.pt
set -euo pipefail
TAG="$1"; FILE="$2"
TMP="$(mktemp -d)"; cp "$FILE" "$TMP/latest.pt"
gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --prerelease --title "$TAG" --notes "training checkpoint (not the submission)"
gh release upload "$TAG" "$TMP/latest.pt" --clobber
echo "published $FILE as release $TAG"
