#!/usr/bin/env bash
set -e

MODEL_DIR="models/kokoro-multi-lang-v1_0"
ARCHIVE="kokoro-multi-lang-v1_0.tar.bz2"
URL="https://github.com/K-Jadeja/tts-models/releases/download/v1.1/$ARCHIVE"

if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "✅ Model already exists, skipping download."
    exit 0
fi

echo "📥 Downloading Kokoro model (~333MB)..."
mkdir -p models
curl -L -o "models/$ARCHIVE" "$URL"

echo "📦 Extracting (preserving custom lexicon-us-en.txt)..."
# --keep-old-files: won't overwrite files that already exist (protects our custom lexicon)
tar -xjf "models/$ARCHIVE" -C models/ --keep-old-files 2>/dev/null || true

rm "models/$ARCHIVE"
echo "✅ Model ready at $MODEL_DIR"
echo "✅ Custom lexicon-us-en.txt preserved."
