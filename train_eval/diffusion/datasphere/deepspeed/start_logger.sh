#!/usr/bin/bash
set -e

# Move to script directory
# to store execution data in the same directory
cd $(dirname "$0")

# Get path of the last created log
LOGS_PATH=$(find /tmp/datasphere/ -type d -printf '%T@ %p\n' | sort -k1,1nr | head -1 | awk '{print $2}')
echo "Automatic logs collection was started"
echo "Run VScode tensorboard to visualize them (Ctrl + Shift + P -> Python: Launch TensorBoard)"
nohup python3 -m animator.utils.logs.logger --path $LOGS_PATH/stdout.txt >/tmp/nohup.out 2>&1 &
