'''
    Creats tensorboard writer to store training log.  

    Data taken from remote machine stdout which stored by datasphere at stdout.txt directory.

    To start execute in separate command line: 
        python3 createSummayWrighter.py --path /tmp_path_to_file_returned_by_datasphere_script/stdout.txt
    To stop: 'Ctrl + C'

    Args:
        --path - required path to stdout.txt file
        --sleep - optional argument sets a time for new data arrival check, deafult 30 sec
        --wait - optional argument sets a time to wait(min) after last update, deafult 20 min
'''
import argparse
import os
import sys
from datetime import datetime
from json import loads as jloads
import time
from torch.utils.tensorboard.writer import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True, help = 'required path to stdout.txt file')
parser.add_argument('--sleep', type = int, default = 30, help = 'optional argument sets a time for new data arrival check')
parser.add_argument('--wait', type = int, default = 20, help = 'optional argument sets a time to wait(min) after last update')

def fill_writer():
    # Pars script args
    args = parser.parse_args()

    # Create path to store log 
    working_directory = os.getcwd()
    log_dir = os.path.join(working_directory, 'runs/',
                           datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    wait = args.wait
    last_update_time = time.time()

    writer = SummaryWriter(log_dir, flush_secs = 1)
    with open(args.path) as stdout_file:
        while time.time() - last_update_time < wait * 60:
            line = stdout_file.readline()
            if not line:
                time.sleep(args.sleep)
            else:
                metrics = jloads(line)
                writer.add_scalars('Loss', {'train': metrics['train_loss'], 'val': metrics['val_loss']}, metrics['epoch'])
                writer.add_scalars('IoU', {'train': metrics['train_IoU'], 'val': metrics['val_IoU']}, metrics['epoch'])

                last_update_time = time.time()

if __name__ == '__main__':
    fill_writer()
