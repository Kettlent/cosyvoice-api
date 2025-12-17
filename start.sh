#!/bin/bash

### -------------------------
### CosyVoice Auto-Start Script (Final)
### -------------------------

LOG_FILE="/workspace/cosyvoice.log"
ENV_NAME="cosyvoice"
MODEL_DIR="/workspace/cosyvoice-api/pretrained_models/Fun-CosyVoice3-0.5B"
SERVER_DIR="/workspace/cosyvoice-api/runtime/python/fastapi"
PORT=8888

echo "" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE
echo "Starting CosyVoice Server at $(date)" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE

# Load conda
if [ -f /workspace/miniconda/etc/profile.d/conda.sh ]; then
    echo "[OK] Loading conda..." >> $LOG_FILE
    source /workspace/miniconda/etc/profile.d/conda.sh
else
    echo "[ERROR] conda.sh not found!" >> $LOG_FILE
    exit 1
fi

# Activate environment
echo "[OK] Activating conda env: $ENV_NAME" >> $LOG_FILE
conda activate $ENV_NAME

# Kill old server if exists
OLD_PIDS=$(pgrep -f "server.py")
if [ ! -z "$OLD_PIDS" ]; then
    echo "[INFO] Killing old server processes: $OLD_PIDS" >> $LOG_FILE
    pkill -f "server.py"
fi

# Go to server directory
cd $SERVER_DIR || {
    echo "[ERROR] Could not cd to $SERVER_DIR" >> $LOG_FILE
    exit 1
}

echo "[OK] Starting server on port $PORT..." >> $LOG_FILE

# Start server in the background
nohup python3 server.py \
    --port $PORT \
    --model_dir $MODEL_DIR \
    >> $LOG_FILE 2>&1 &

sleep 2

# Verify server started
NEW_PID=$(pgrep -f "server.py")
if [ -z "$NEW_PID" ]; then
    echo "[ERROR] Server failed to start!" >> $LOG_FILE
else
    echo "[OK] Server running with PID: $NEW_PID" >> $LOG_FILE
fi

echo "-----------------------------------" >> $LOG_FILE
echo "CosyVoice Startup Complete" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE