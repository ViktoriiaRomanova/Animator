#!/usr/bin/bash
set -e
ch_dir=train_checkpoints/

# Move to script directory
cd $(dirname "$0")

# Create a directory if it does not exist 
if [ ! -d $ch_dir ]; then
mkdir $ch_dir
fi

datasphere project job execute -p $1 -c config.yaml

# Get the name of the last obtained weights
#WNAME=$(aws s3 ls s3://diffusion-based-model --recursive --output text | sort | tail -n 1 | awk '{print $4}')

# Copy to the local folder
#aws s3 cp s3://diffusion-based-model/$WNAME $ch_dir

# Move checkpoints to that directory
#mv ./output_model*/* $ch_dir
#rm -r output_model*