#!/usr/bin/bash

# Runs training process for any pipeline in a docker container on a remote machine

# Before starting this shell script:
# - move to remote machine:
#   - train.py
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

TRANSFORM=diffusion/
OUTPUT_MODEL=train_checkpoints/
IMODEL=/train_checkpoints/2025_04_11_16_41_33/restart_from_epoch:1
PARAMS=hyperparameters.yaml

CUR_HOST=name@PI
MY_REMOTE_DIR=/home/name/animator/

#docker context update --docker host=ssh://$CUR_HOST compute-vm

# Automatic move of the necessary data
#scp train.py $CUR_HOST:$MY_REMOTE_DIR # train
#scp ds_config.json $CUR_HOST:$MY_REMOTE_DIR
#scp ds_config_disc.json $CUR_HOST:$MY_REMOTE_DIR # train
#scp -r ../../../../animator $CUR_HOST:$MY_REMOTE_DIR # animator package
#scp hyperparameters.yaml $CUR_HOST:$MY_REMOTE_DIR # hyperparameters
#scp -r ../../../../datasets/diffusion/ $CUR_HOST:$MY_REMOTE_DIR/$TRANSFORM # dataset
#scp /train_checkpoints/129.pt $CUR_HOST:$MY_REMOTE_DIR/$IMODEL # initial weights (optional)

docker --context compute-vm run --name animator \
--mount type=bind,source="$MY_REMOTE_DIR",target=/workspace \
--mount type=bind,source=/home/name/.cache/,target=/root/.cache/ \
--rm \
-w /workspace/ \
--shm-size=1g \
--gpus all cuda12.8.0 \
deepspeed train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--imodel ${IMODEL} \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}

# Get the name of the last obtained weights
#WNAME=$(ssh remote-machine "find viktoriia/Animator/train_checkpoints/ -type f -printf '%T@ %p\n' | sort -k1,1nr | head -1" | awk '{print $2}')

# Create a directory if it does not exist 
#if [ ! -d train_checkpoints/ ]; then
#mkdir train_checkpoints/
#fi
# Copy that weight to the local directory
#scp remote-machine:$WNAME train_checkpoints/
