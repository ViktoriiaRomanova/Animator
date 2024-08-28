#!/usr/bin/bash
set -e

VIDEO_PATH=/home/viktoriia/Downloads/videotest.MOV
IMODEL=train_eval/local/train_checkpoints/199_old.pt
PARAMS=train_eval/style_transfer/hyperparameters.yaml
RES_FOLDER=~/Downloads/

python3 -m animator.post_processing.video_processing --pvideo ${VIDEO_PATH} \
--pmodel ${IMODEL} \
--pres ${RES_FOLDER} \
--hyperp ${PARAMS}
