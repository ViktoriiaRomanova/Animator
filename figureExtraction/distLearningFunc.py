import os
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime

from processingDataSet import MaskDataset

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
                   data: DataLoader, optim: Optional[torch.optim.Optimizer] = None
                   ) -> Tuple[torch.tensor, torch.tensor]:
    """
        Make train/eval operations per epoch.

        Returns: loss and quality metric.
    """
    avg_loss, metric = 0, 0
    for x_batch, y_batch in tqdm(data):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if optim is not None: optim.zero_grad()

        y_pred = model(x_batch)
        loss = loss_func(y_pred, y_batch)
        if optim is not None:
            loss.backward()
            optim.step()

        # Calculate average train loss and metric
        avg_loss += (loss/len(data)).detach().cpu()

        # !it is not final result, to get real metric need to divide it into num_batches
        metric += metric_func(y_pred, y_batch)

        del x_batch, y_batch, y_pred, loss

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
                             sampler = sampler, pin_memory = False)
    return data_loader

def prepare_strorage_folders() -> Tuple[str, str]:
    """Create/check directories for log and model weights storage."""
    working_directory = os.getcwd()

    log_dir = os.path.join(working_directory, 'runs/',
                           datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    model_weights_dir = os.path.join(working_directory, 'train_checkpoints/')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    return log_dir, model_weights_dir


def worker(rank: int, model: nn.Module, world_size: int, train_data: MaskDataset,
           val_data: MaskDataset, batch_size: int,
           seed: int, epochs: int, pretrained: Optional[str] = None) -> None:
    """Describe training process which will be implemented for each worker."""
    # Setup process group, for each worker
    setup(rank, world_size)
    
    device = torch.device(rank)
    # Set device GPU to make transformations on GPU
    train_data.device = device
    val_data.device = device

    # prepare data
    train_loader = prepare_dataloader(train_data, rank, world_size, batch_size, seed)
    val_loader = prepare_dataloader(val_data, rank, world_size, batch_size, seed)        

    model.to(device)

    model = DDP(model, device_ids = [rank],
                output_device = rank,
                find_unused_parameters = False)

    optimizer = torch.optim.Adam(model.parameters())
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           #mode = 'max', factor = 0.5,
                                                          #patience = 3, cooldown = 5)
            
    start_epoch = 0      
    if pretrained is not None:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, 'train_checkpoints/', pretrained)
        state = torch.load(weights_dir, map_location = device)
        model.load_state_dict({''.join(['module.', key]): val for key, val in state['model_state_dict'].items()})
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        epochs += start_epoch
    
    loss_func = nn.BCEWithLogitsLoss()

    if rank == 0:        
        #Create/check directories for log and model weights storage
        LOG_DIR, MODEL_WEIGHTS_DIR = prepare_strorage_folders()
        
        # Logging entity
        writer = SummaryWriter(LOG_DIR, flush_secs = 1)

    for epoch in range(start_epoch, epochs):

        train_loader.sampler.set_epoch(epoch - start_epoch)
        val_loader.sampler.set_epoch(epoch - start_epoch)

        model.train()
        train_loss, train_metric = fit_eval_epoch(model, loss_func, IoU, device, train_loader, optimizer)
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
            if (epoch + 1) % 1 == 0: 
                torch.save({'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}, 
                           os.path.join(MODEL_WEIGHTS_DIR,
                                                                   datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pt'))
            writer.add_scalars('Loss', {'train': metrics[0].item(), 'val': metrics[2].item()}, epoch)
            writer.add_scalars('IoU', {'train': metrics[1].item(), 'val': metrics[3].item()}, epoch)
            if epoch == epochs - 1:
                writer.close()
            

    dist.destroy_process_group()
