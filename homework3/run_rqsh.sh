#!/bin/bash

BASE_DIR="/workspaces/ezr"
TMP_DIR="$BASE_DIR/tmp"
LOW_TMP="$TMP_DIR/low"
HIGH_TMP="$TMP_DIR/high"
RQ_SCRIPT="$BASE_DIR/etc/rq.sh"

cd "$LOW_TMP" && bash "$RQ_SCRIPT"

cd "$HIGH_TMP" && bash "$RQ_SCRIPT"
