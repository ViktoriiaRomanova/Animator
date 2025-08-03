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

TRANSFORM=datasets/diffusion/
OUTPUT_MODEL=diffusion/train_checkpoints/2025_07_31_18_00/
#IMODEL=diffusion/train_checkpoints/2025_02_12_16_04_07/1.pt
PARAMS=diffusion/hyperparameters.yaml

# Automatic move of the necessary data
#scp train.py remote-machine:$MY_REMOTE_DIR/diffusion # train
scp -r ../../../../animator remote-machine:$MY_REMOTE_DIR # animator package
scp hyperparameters.yaml remote-machine:$MY_REMOTE_DIR/diffusion # hyperparameters
scp ds_config.json remote-machine:$MY_REMOTE_DIR
scp ds_config_disc.json remote-machine:$MY_REMOTE_DIR # train
#scp -r ../../../datasets/diffusion/ remote-machine:$MY_REMOTE_DIR/$TRANSFORM # dataset
#scp ../diffusion/train_checkpoints/129.pt remote-machine:$MY_REMOTE_DIR/$IMODEL # initial weights (optional)

# !REMINDER do not use deepspeed activation checkpointing!
# as, for now (v0.17.2), it is not supporting 
# "Jointly Training Models With Shared Loss" pipline


# REMINDER2 to make deepspeed activation checkpointing work call:
# deepspeed.checkpointing.configure(mpu_=None, deepspeed_config=ds_config) MANDATORY
# setting ds_confi_dummy_discg with nesessary configurations into:
# deepspeed.initialize(model=model1, config=ds_config) 
# does NOTHING with activations checkpointing configurations

docker --context remote-machine run --name animator \
--mount type=bind,source="$MY_REMOTE_DIR",target=/workspace \
--mount type=bind,source=/root/.cache/,target=/root/.cache/ \
--rm \
-w /workspace/ \
--shm-size=1g \
--gpus all cuda_new:deepspeed17.2 \
deepspeed diffusion/train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}

#deepspeed diffusion/train.py \

# Get the name of the last obtained weights
#WNAME=$(ssh remote-machine "find viktoriia/Animator/diffusion/train_checkpoints/ -type f -printf '%T@ %p\n' | sort -k1,1nr | head -1" | awk '{print $2}')

# Create a directory if it does not exist 
#if [ ! -d train_checkpoints/ ]; then
#mkdir train_checkpoints/
#fi
# Copy that weight to the local directory
#scp remote-machine:$WNAME train_checkpoints/