#!/usr/bin/bash
# Runs training process of any pipeline in a docker container on a local machine
# Before starting this shell script move/rewrite paths for:
# - train.py
# - worker.py
# - hyperparameters.py
# - dataset
# - pre-trained weights (optionally)
# - create from the provided Dockerfile a docker container on the same machine

set -e
TRANSFORM=datasets/transfer_test/
OUTPUT_MODEL=train_checkpoints/
IMODEL=train_checkpoints/
PARAMS=hyperparameters.yaml

exec sudo docker run --mount type=bind,source="$(pwd)",target=/workdir \
-it --rm \
-w /workdir/ \
--gpus all test:2 \
python3 train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--imodel ${IMODEL}/69.pt \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}