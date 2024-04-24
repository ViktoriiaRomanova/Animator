#!/usr/bin/bash
set -e
ch_dir=train_checkpoints/
ch_name=train_checkpoints.zip

# Create a directory if it does not exist 
if [ ! -d $ch_dir ]; then
mkdir $ch_dir
fi

datasphere project job execute -p $1 -c config.yaml

# Move checkpoints to that directory 
if [ -f $ch_name ]; then
mv $ch_name $ch_dir
fi
#tar -xvf $ch_dir$ch_name -C $ch_dir
unzip $ch_dir$ch_name -d $ch_dir
rm $ch_dir$ch_name