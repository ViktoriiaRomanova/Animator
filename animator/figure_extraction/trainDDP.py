'''
    Main file to start distributed training on a remote machine by DataSphere Jobs CLI

    Reminder: 
        To start execute: 'datasphere project job execute -p PROJECT_ID -c config.yaml'
'''

import argparse
import torch.multiprocessing as mp

from processingDataSet import PreprocessingData
from distLearningFunc import worker

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True) # path to dataset
parser.add_argument('--prhome', required=True) # path to project home directory READONLY
parser.add_argument('--omodel', required=True) # output checkpoints of model weights 
parser.add_argument('--imodel') # input model weights 

if __name__ == '__main__':
    args = parser.parse_args()
    random_state = 10
    world_size = 2  # Number of GPU
    batch_size = 64
    seed = random_state
    epochs = 2

    prData = PreprocessingData(0.9)
    train_data, val_data = prData.get_data(args.dataset, random_state, 0.1)

    mp.spawn(worker, args = (args, world_size, train_data, val_data, batch_size, seed, epochs),
         nprocs = world_size)