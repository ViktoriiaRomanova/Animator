#!/usr/bin/bash

# Runs training process for any pipeline in a docker container on a local-remote machine

# Before starting this shell script:
# - temporarily add absolute path on the remote machine to environment variables: export MY_REMOTE_DIR=path/to/folder
# - move to remote machine:
#   - train.py
#   - worker.py
#   - animator package
#   - hyperparameters.py
#   - dataset
#   - pre-trained weights (optionally)
#   - create from the provided Dockerfile a docker container on the remote machine

# Necessary preliminary configuration setup:
# - ssh-agent (https://code.visualstudio.com/docs/containers/ssh)
# - Docker context (https://docs.docker.com/engine/context/working-with-contexts/)
# - ssh config (https://linuxize.com/post/using-the-ssh-config-file/)
# note: here, the context for docker and ssh have the same name and lead to the same remote machine
set -e

# Move to script directory
cd $(dirname "$0")

TRANSFORM=datasets/transfer_test/
OUTPUT_MODEL=train_checkpoints/
IMODEL=train_checkpoints/
PARAMS=hyperparameters.yaml

# Automatic move of the necessary data
#scp ../stryle_transfer/train.py remote_machine:$MY_REMOTE_DIR # train
#scp ../stryle_transfer/worker.py remote_machine:$MY_REMOTE_DIR # worker
#scp -r ../../animator remote_machine:$MY_REMOTE_DIR # animator package
#scp ../stryle_transfer/hyperparameters.py remote_machine:$MY_REMOTE_DIR # hyperparameters
#scp -r ../../datasets/transform/ remote_machine:$MY_REMOTE_DIR/$TRANSFORM # dataset
#scp ../stryle_transfer/train_checkpoints/129.py remote_machine:$MY_REMOTE_DIR/$IMODEL # initial weights (optional)

docker --context remote-machine run --name animator \
--mount type=bind,source="$MY_REMOTE_DIR",target=/workspace \
--rm \
-w /workspace/ \
--shm-size=1g \
--gpus all cuda12.1.0:pytorch2.3.1 \
python3 train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--imodel ${IMODEL}/2024_07_03_06_52_53/129.pt \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}

# Get the name of the last obtained weights
WNAME=$(ssh remote-machine "find viktoriia/Animator/train_checkpoints/ -type f -printf '%T@ %p\n' | sort -k1,1nr | head -1" | awk '{print $2}')

# Create a directory if it does not exist 
if [ ! -d train_checkpoints/ ]; then
mkdir train_checkpoints/
fi
# Copy that weight to the local directory
scp remote-machine:$WNAME train_checkpoints/
