#!/bin/bash
# Rebuild ann_bundle.js whenever any .ts file changes
cd "$(dirname "$0")"

build() {
  echo "[$(date +%H:%M:%S)] Building ann_bundle.js..."
  npx esbuild ann_web.ts --bundle --format=esm --outfile=ann_bundle.js \
    --tree-shaking=true --external:./train.json --external:./test.json 2>&1
}

build

echo "Watching .ts files for changes (polling every 1s)..."
LAST=""
while true; do
  CURRENT=$(stat -f '%m' *.ts 2>/dev/null | sort | tail -1)
  if [[ "$CURRENT" != "$LAST" && -n "$LAST" ]]; then
    build
  fi
  LAST="$CURRENT"
  sleep 1
done
