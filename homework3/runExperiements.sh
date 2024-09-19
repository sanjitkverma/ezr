#!/bin/bash
BASE_DIR="/workspaces/ezr"
SORTED_DIR="$BASE_DIR/homework3/sortedFiles"
LOW_DIR="$SORTED_DIR/low"
HIGH_DIR="$SORTED_DIR/high"
TMP_DIR="$BASE_DIR/tmp"
LOW_TMP="$TMP_DIR/low"
HIGH_TMP="$TMP_DIR/high"
SCRIPT_PATH="$BASE_DIR/homework3/extend_hw3.py"

rm -rf "$LOW_TMP" "$HIGH_TMP"

mkdir -p "$LOW_TMP" "$HIGH_TMP"

for file in "$LOW_DIR"/*; do
    output_file="$LOW_TMP/$(basename "$file" .csv).csv"
    python3.13 "$SCRIPT_PATH" -t "$file" > "$output_file"
done

for file in "$HIGH_DIR"/*; do
    output_file="$HIGH_TMP/$(basename "$file" .csv).csv"
    python3.13 "$SCRIPT_PATH" -t "$file" > "$output_file"
done
