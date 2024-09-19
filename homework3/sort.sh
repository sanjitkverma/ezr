#!/bin/bash

BASE_DIR="/workspaces/ezr"
DATA_DIR="$BASE_DIR/data/optimize"
SORTED_DIR="$BASE_DIR/homework3/sortedFiles"
LOW_DIR="$SORTED_DIR/low"
HIGH_DIR="$SORTED_DIR/high"
SCRIPT_PATH="$BASE_DIR/homework3/extend_hw3.py"

rm -rf "$LOW_DIR" "$HIGH_DIR"  
mkdir -p "$LOW_DIR" "$HIGH_DIR" 

for dir in "config" "hpo" "misc" "process"; do
    for file in "$DATA_DIR/$dir"/*; do
        
        if [[ "$file" != *.csv ]]; then
            continue
        fi
        
        echo "Processing $file"
        output=$(python3.13 "$SCRIPT_PATH" -t "$file")

        if echo "$output" | grep -q "low_dimension"; then
            echo "File $file is low dim"
            cp "$file" "$LOW_DIR"
        else
            echo "File $file is high dim"
            cp "$file" "$HIGH_DIR"
        fi
    done
done