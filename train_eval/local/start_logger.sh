#!/usr/bin/bash
set -e

# Move to script directory
# to store execution data in the same directory
cd $(dirname "$0")

nohup docker --context remote-machine logs -f animator 1>tmp/stdout.txt 2>/dev/null &
echo "Automatic logs collection was started"
echo "Run VScode tensorboard to visualize them (Ctrl + Shift + P -> Python: Launch TensorBoard)"
python3 -m animator.utils.logs.logger --path tmp/stdout.txt