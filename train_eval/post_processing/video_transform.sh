#!/usr/bin/bash
set -e

VIDEO_PATH=/home/viktoriia/Downloads/videotest.MOV
IMODEL=train_eval/local/train_checkpoints/199_old.pt
PARAMS=train_eval/style_transfer/hyperparameters.yaml
RES_PATH=train_eval/post_processing/resvideo_tmp.MOV

python3 -m animator.post_processing.video_processing --pvideo ${VIDEO_PATH} \
--pmodel ${IMODEL} \
--pres ${RES_PATH} \
--hyperp ${PARAMS}
