from collections import OrderedDict
from json import dumps as jdumps

import torch
import torch.distributed as dist

class MerticGroup:
    def __init__(self,
                 device: torch.device | None,
                 world_size: int,
                 num_batch: int,
                 dst: int = 0) -> None:
        self.metrics = OrderedDict()
        self.device = device
        self.dst = dst
        self.world_size = world_size
        self.num_batch = num_batch

    def add(self, metric_name: str, val: int | float) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = val
        else:
            self.metrics[metric_name] += val
    
    def share(self,) -> dict[str, float | int]:
        shared_tensor = torch.tensor([val / (self.world_size * self.num_batch) for val in self.metrics.values()], device = self.device)
        dist.reduce(shared_tensor, dst = self.dst, op = dist.ReduceOp.SUM)
        self.metrics.clear()
        return {key: val.item() for key, val in zip(self.metrics.keys(), shared_tensor)}

class MetricStorage:
    def __init__(self, rang: int, device: torch.device, world_size: int, num_batch: int, dst: int) -> None:
        self.groups = OrderedDict()
        self.rang = rang
        self.device = device
        self.world_size = world_size
        self.num_batch = num_batch
        self.dst = dst
        self.epoch = -1

    def update(self, group_name: str, metric_name: str, val: float | int) -> None:
        if group_name not in self.groups:
            self.groups[group_name] = MerticGroup(self.device, self.world_size, self.num_batch, self.dst)
        self.groups[group_name].add(metric_name, val)
    
    def set_epoch(self, val: int) -> None:
        self.epoch = val
    
    def share(self,) -> None:
        result = {}
        for group_name in self.groups.keys():
            metrics = self.groups[group_name].share()
            if self.rang == self.dst:
                result[group_name] = metrics
        result['epoch'] = self.epoch

        if self.rang == self.dst:        
            # Send metrics into stdout. This channel going to be transferred into initial machine.
            # Store metrics in JSON format to simplify parsing and transferring them into tensorboard at initial machine.
            print(jdumps(result))

            
        





