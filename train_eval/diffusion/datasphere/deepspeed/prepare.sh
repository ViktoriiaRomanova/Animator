set -e

TRANSFORM=datasets/diffusion/
OUTPUT_MODEL=diffusion/train_checkpoints/2025_07_31_18_00/
#IMODEL=diffusion/train_checkpoints/2025_02_12_16_04_07/1.pt
PARAMS=hyperparameters.yaml

cp test_config_files/parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py

deepspeed train.py \
--dataset ${TRANSFORM} \
--omodel ${OUTPUT_MODEL} \
--params ${PARAMS} \
--st ${OUTPUT_MODEL}