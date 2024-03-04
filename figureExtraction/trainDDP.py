import argparse
import torch.multiprocessing as mp

from processingDataSet import PreprocessingData
from distLearningFunc import worker

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--prhome', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    random_state = 10
    world_size = 2  # Number of GPU
    batch_size = 64
    seed = random_state
    epochs = 1

    prData = PreprocessingData(0.9)
    train_data, val_data = prData.get_data(args.dataset, random_state)

    mp.spawn(worker, args = (args, world_size, train_data, val_data, batch_size, seed, epochs),
         nprocs = world_size)