'''
    Creats tensorboard writer to store training log.  

    Data taken from remote machine stdout which stored by datasphere at stdout.txt.

    To start execute in separate command line: 
        python3 logger.py --path /tmp_path_to_file_returned_by_datasphere_script/stdout.txt

    Args:
        --path - required path to stdout.txt file

'''
import argparse
from .create_summary_wrighter import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True, help = 'required path to stdout.txt file')
parser.add_argument('--mode', default='epoch', help = 'use as x-axis step "epoch", or data "ind"')

if __name__ == '__main__':
    params = parser.parse_args()
    logger = Logger(params.path, params.mode)
    logger.start()
