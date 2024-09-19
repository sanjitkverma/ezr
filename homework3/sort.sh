#!/bin/bash

BASE_DIR="/workspaces/ezr"
DATA_DIR="$BASE_DIR/data/optimize"
SORTED_DIR="$BASE_DIR/homework3/sortedFiles"
LOW_DIR="$SORTED_DIR/low"
HIGH_DIR="$SORTED_DIR/high"
SCRIPT_PATH="$BASE_DIR/homework3/extend_hw3.py"

mkdir -p "$LOW_DIR"
mkdir -p "$HIGH_DIR"

for file in "$DATA_DIR"/*/*; do
    echo "Processing $file"

    output=$(python3.13 "$SCRIPT_PATH" -t "$file")

    if echo "$output" | grep -q "low_dimension"; then
        echo "File $file is low dimension. Moving to $LOW_DIR"
        cp "$file" "$LOW_DIR"
    else
        echo "File $file is high dimension. Moving to $HIGH_DIR"
        cp "$file" "$HIGH_DIR"
    fi
done
