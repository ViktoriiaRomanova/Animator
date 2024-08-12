from collections import OrderedDict

import torch
import torch.distributed as dist

class MerticGroup:
    def __init__(self,
                 device: torch.device,
                 world_size: int,
                 dst: int = 0) -> None:
        self.metrics = OrderedDict()
        self.device = device
        self.dst = dst
        self.world_size = world_size

    def add(self, metric_name: str, val: int | float) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = val
        else:
            self.metrics[metric_name] += val
    
    def share(self,) -> list[tuple[str, float | int]]:
        shared_tensor = torch.tensor([val / self.world_size for val in self.metrics.values()], device = self.device)
        dist.reduce(shared_tensor, dst = self.dst, op = dist.ReduceOp.SUM)
        for key in self.metrics.keys():
            self.metrics[key] = 0
        return [(key, val.item()) for key, val in zip(self.metrics.keys(), shared_tensor)]

class MetricStarage:
    def __init__(self, rang: int, device: torch.device, world_size: int, batch_size: int, dst: int) -> None:
        self.groups = OrderedDict()
        self.rang = rang
        self.device = device
        self.world_size = world_size
        self.batch_size = batch_size
        self.dst = dst

    def update(self, group_name: str, metric_name: str, val: float | int) -> None:
        if group_name not in self.groups:
            self.groups[group_name] = MerticGroup(self.device, self.world_size, self.dst)
        self.groups[group_name].add(metric_name, val)
    
    def share(self,) -> None:
        result = {}
        for group_name in self.groups.keys():
            
        





