#!/usr/bin/bash
# Runs training process for any pipeline in a docker container on a local/remote machine
# Before starting this shell script move/rewrite paths for:
# - train.py
# - worker.py
# - hyperparameters.py
# - dataset
# - pre-trained weights (optionally)
# - create from the provided Dockerfile a docker container on the same/remote machine
# - export MY_REMOTE_DIR=path/to/folder - temporarily add the base catalogue to environment variables

set -e
TRANSFORM=$MY_REMOTE_DIR/datasets/transfer_test/
OUTPUT_MODEL=$MY_REMOTE_DIR/train_checkpoints/
IMODEL=$MY_REMOTE_DIR/train_checkpoints/
PARAMS=$MY_REMOTE_DIR/hyperparameters.yaml

docker run \
--context remote_machine
--mount type=bind,source="$MY_REMOTE_DIR",target=/workspace \
-it --rm \
-w /workspace/ \
--shm-size=1g \
--gpus all cuda12.1.0:pytorch2.3.1 \
--name animator \
python3 train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--imodel ${IMODEL}/2024_07_03_06_52_53/129.pt \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}

ssh remote-machine
cd $MY_REMOTE_DIR

# Get the name of the last obtained weights
WNAME=$(ls -R train_checkpoints/ | sort | tail -n 1 | awk '{print $4}')