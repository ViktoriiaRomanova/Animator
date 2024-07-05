#!/usr/bin/bash
set -e

# Move to script directory
# to store execution data in the same directory
cd $(dirname "$0")

if [ ! -d tmp/ ]; then
mkdir tmp/
fi

nohup docker --context remote-machine logs -f animator 1>tmp/stdout.txt 2>/dev/null &
echo "Automatic logs collection was started"
echo "Run VScode tensorboard to visualize them (Ctrl + Shift + P -> Python: Launch TensorBoard)"
nohup python3 -m animator.utils.logs.logger --path tmp/stdout.txt >tmp/nohup.out 2>&1 &