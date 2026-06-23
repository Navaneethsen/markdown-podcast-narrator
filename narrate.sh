#!/usr/bin/env bash
set -e

read -rp "Enter markdown file path: " INPUT_FILE

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: file not found: $INPUT_FILE"
    exit 1
fi

OUTPUT_FILE="$(basename "${INPUT_FILE%.*}").mp3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/main.py" "$INPUT_FILE" --engine kokoro -o "$OUTPUT_FILE"
