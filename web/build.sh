#!/bin/bash
# Build the browser viewer into web/dist/ (wasm + JS glue + page).
#   ./web/build.sh                      # build only
#   ./web/build.sh ../data/masa4        # build + publish that match's bundle
set -e
cd "$(dirname "$0")/.."

cargo build --release --target wasm32-unknown-unknown -p billiards-viewer
wasm-bindgen --target web --no-typescript \
    --out-dir web/dist \
    target/wasm32-unknown-unknown/release/billiards_viewer.wasm
V=$(shasum target/wasm32-unknown-unknown/release/billiards_viewer.wasm | cut -c1-10)
sed "s/__V__/$V/g" web/index.html > web/dist/index.html

if [ -n "$1" ]; then
    (cd python && python3 publish_bundle.py "$1" --out ../web/dist/bundle)
fi
echo "web/dist ready — serve:  python3 -m http.server -d web/dist 8080"
echo "deploy:  npx wrangler pages deploy web/dist --project-name billiards-coach"
