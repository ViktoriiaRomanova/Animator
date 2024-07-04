import os
from datetime import datetime
from json import loads as jloads
import time
from torch.utils.tensorboard.writer import SummaryWriter

class Logger:
    """
        Creats and fill tensorboard writer to store training log.  

        Data taken from remote machine stdout which stored by datasphere at stdout.txt.
    """
    def __init__(self, data_path: str, log_path: str | None = None,
                 sleep: int = 30, wait: int = 20):
        """
            Creats tensorboard writer to store training log.
            Args:
                * data_path - required path to stdout.txt file
                * log_path - optional argument path to the log storage,
                             by default will be created catalog in format '%Y_%m_%d_%H_%M_%S'
                * sleep - optional argument sets a time for new data arrival check,
                          deafult 30 sec
                * wait - optional argument sets a time to wait before closing(minutes) after last update,
                         deafult 20 min
        """
        
        self.data_path = data_path
        self.sleep = sleep
        self.wait = wait

        self.log_path = log_path
        if log_path is None:
            # Create path to store log 
            working_directory = os.getcwd()
            self.log_path = os.path.join(working_directory, 'runs/',
                                   datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        self.writer =  SummaryWriter(self.log_path, flush_secs = 1)
        
    def start(self, ) -> None:
        """Starts metrics collection for SammaryWrighter."""
        # Time last collected data 
        last_update_time = time.time()

        with open(self.data_path) as stdout_file:
            while time.time() - last_update_time < self.wait * 60:
                line = stdout_file.readline()
                if not line:
                    time.sleep(self.sleep)
                else:
                    metrics = jloads(line)
                    for main_tag in metrics:
                        if main_tag == 'epoch': continue
                        self.writer.add_scalars(main_tag, metrics[main_tag], metrics['epoch'])

                    last_update_time = time.time()
