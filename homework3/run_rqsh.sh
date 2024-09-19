#!/bin/bash

BASE_DIR="/workspaces/ezr"
TMP_DIR="$BASE_DIR/tmp"
LOW_TMP="$TMP_DIR/low"
HIGH_TMP="$TMP_DIR/high"
RQ_SCRIPT="$BASE_DIR/etc/rq.sh"

echo "select the dim of data"
echo "1. low Dim, normal table"
echo "2. high Dim, normal table"
read -p "Enter your choice (1 or 2): " choice
echo ""
echo ""

case $choice in
    1)
        cd "$LOW_TMP" && bash "$RQ_SCRIPT"
        ;;
    2)
        cd "$HIGH_TMP" && bash "$RQ_SCRIPT"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
