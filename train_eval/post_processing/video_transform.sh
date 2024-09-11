#!/usr/bin/bash
set -e

VIDEO_PATH=/home/viktoriia/Downloads/waterfall.MOV
IMODEL=train_eval/style_transfer/train_checkpoints/ukiyoe/199.pt
PARAMS=train_eval/style_transfer/hyperparameters.yaml
RES_FOLDER=~/Downloads/

python3 -m animator.post_processing.video_processing --pvideo ${VIDEO_PATH} \
--pmodel ${IMODEL} \
--pres ${RES_FOLDER} \
--hyperp ${PARAMS}
#--start 0.0
#--length 6.0
