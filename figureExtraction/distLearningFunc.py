import os
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from datetime import datetime

from processingDataSet import MaskDataset
from SegNetModel import SegNet
from UNetModel import UNet
from tqdm import tqdm


@torch.compile()
def IoU(pred: torch.tensor, real: torch.tensor,
        border: int = 0, smooth: float = 1e-8) -> float:
    """ Calculate Intersection over Union metric. """
    pred = pred.detach() > border # Checking wich prediction greater then threshold
    pred = pred.squeeze(dim = 1).byte() # Removing channel dimension  (there is only 1 channel)
    real = real.squeeze(dim = 1).byte()
    intersection = (pred & real).float().sum((1,2))
    union_ = (pred | real).float().sum((1,2))
    iou = (intersection / (union_ + smooth)).mean().item()
    return iou

def fit_eval_epoch(model: DDP,
                   loss_func: nn.Module, metric_func: Any, device: torch.device,
                   data: DataLoader, optim: Optional[torch.optim.Optimizer] = None,
                   scaler: Optional[torch.cuda.amp.GradScaler] = None,
                   prof: Optional[torch.profiler.profile] = None) -> Tuple[torch.tensor, torch.tensor]:
    """
        Make train/eval operations per epoch.

        Returns: loss and quality metric.
    """
    
    avg_loss, metric = 0, 0
    
    for x_batch, y_batch in tqdm(data):
        if prof: prof.step()
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = True):
            if optim:
                for param in model.parameters():
                    param.grad = None

            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
        if optim:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

        # Calculate average train loss and metric
        avg_loss += (loss/len(data)).detach()

        # !it is not final result, to get real metric need to divide it into num_batches
        metric += metric_func(y_pred, y_batch)

        del x_batch, y_batch, y_pred, loss
    if prof: prof.stop()
    metric /= len(data)
    return avg_loss, metric

def setup(rank: int, world_size: int) -> None:
    """Setup the process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    # 'nccl' -- for GPU
    dist.init_process_group('nccl', rank = rank, world_size = world_size)


def prepare_dataloader(data: MaskDataset, rank: int,
                    world_size: int, batch_size: int,
                    seed: int) -> DataLoader:
    """
        Split dataset into N parts.

        Returns: DataLoader instance for current part.
    """
    sampler = DistributedSampler(data, num_replicas = world_size,
                                 rank = rank, shuffle = True,
                                 seed = seed, drop_last = True)
    data_loader = DataLoader(data, batch_size = batch_size,
                             shuffle = False, drop_last = True,
                             sampler = sampler, pin_memory = True,
                             num_workers = 8,
                             prefetch_factor = 16,
                             multiprocessing_context = 'spawn',
                             persistent_workers = True,
                             pin_memory_device = str(torch.device(rank)))
    return data_loader

def prepare_strorage_folders() -> Tuple[str, str]:
    """Create/check directories for log and model weights storage."""
    working_directory = os.getcwd()

    #log_dir = os.path.join(working_directory, 'runs/',
                           #datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    log_dir = os.path.join(working_directory, 'runs/2024_02_27_12_38_52')

    model_weights_dir = os.path.join(working_directory, 'train_checkpoints/')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    return log_dir, model_weights_dir

def start_profiler() -> torch.profiler.profile:
    """Create and return class object to collect performance metrics."""
    prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=4, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/SegNet'),
    record_shapes=False,
    with_stack=False)
    prof.start()
    return prof


def worker(rank: int, world_size: int, train_data: List[str],
           val_data: List[str], batch_size: int,
           seed: int, epochs: int, pretrained: Optional[str] = None) -> None:
    """Describe training process which will be implemented for each worker."""
    # Setup process group, for each worker
    setup(rank, world_size)
    
    device = torch.device(rank)
    start_epoch = 0
    
    data_path = '/home/jupyter/mnt/datasets/Segmentation/Training' # Path to dataset
    traind_weights_dir = 'train_checkpoints/' # Directory to store trained model weights
    pretraind_weights_path = 'weights/pretrained_encoder_weights_DEFAULT.pt' # Diectory with loaded encoder weights from pytorch
    
    
    # prepare data
    transform = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(p = 0.1),
    transforms.RandomVerticalFlip(p = 0.1),
    transforms.RandomPerspective(p = 0.1)])
    
    train_set = MaskDataset(data_path, train_data, transform)
    val_set = MaskDataset(data_path, val_data, transform)

    train_loader = prepare_dataloader(train_set, rank, world_size, batch_size, seed)
    val_loader = prepare_dataloader(val_set, rank, world_size, batch_size, seed)      
    
    # prepare model
    model = UNet()  
    model.to(device)
    model = DDP(model, device_ids = [rank],
                output_device = rank,
                find_unused_parameters = False)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    loss_func = torch.compile(nn.BCEWithLogitsLoss())
    
    # Load weights
    if pretrained is None:
        pass
    elif pretrained != 'Transfer':
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, traind_weights_dir, pretrained)
        state = torch.load(weights_dir, map_location = device)
        model.load_state_dict({''.join(['module.', key]): val for key, val in state['model_state_dict'].items()},strict = False)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        epochs += start_epoch
        scaler.load_state_dict(state['scaler'])
    else:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, pretraind_weights_path)
        state = torch.load(weights_dir, map_location = device)
        model.module.encoder.load_state_dict(state, strict = False)
    
    model = torch.compile(model)

    if rank == 0:        
        #Create/check directories for log and model weights storage
        LOG_DIR, MODEL_WEIGHTS_DIR = prepare_strorage_folders()
        
        # Logging entity
        writer = SummaryWriter(LOG_DIR, flush_secs = 1)
    
    prof = None
    #Eable to collect model performance metrics
    #prof = start_profiler()

    for epoch in range(start_epoch, epochs):

        train_loader.sampler.set_epoch(epoch - start_epoch)
        val_loader.sampler.set_epoch(epoch - start_epoch)

        model.train()
        train_loss, train_metric = fit_eval_epoch(model, loss_func, IoU, device, train_loader, optimizer, scaler, prof)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = fit_eval_epoch(model, loss_func, IoU, device, val_loader)

        # Share metrics
        metrics = torch.tensor([train_loss / world_size, train_metric / world_size,
                                val_loss / world_size, val_metric / world_size], device = device)
        dist.all_reduce(metrics, op = dist.ReduceOp.SUM)

        # Making sheduler step if validation metric is not rising for some time
        #scheduler.step(metrics[3])

        if rank == 0:
            '''print('train_loss: ', metrics[0].item(), metrics[0].dtype, 'val_loss: ', metrics[2].item(), 'train_IoU: ',
                  metrics[1].item(), 'val_IoU: ', metrics[3].item(), 'epoch: ', epoch + 1, '/', epochs)'''
            if (epoch + 1) % 5 == 0: 
                torch.save({'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'scaler': scaler.state_dict()}, 
                           os.path.join(MODEL_WEIGHTS_DIR,
                                                                   datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pt'))
            writer.add_scalars('Loss', {'train': metrics[0].item(), 'val': metrics[2].item()}, epoch)
            writer.add_scalars('IoU', {'train': metrics[1].item(), 'val': metrics[3].item()}, epoch)
            if epoch == epochs - 1:
                writer.close()
                pass
            

    dist.destroy_process_group()
