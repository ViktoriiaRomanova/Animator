from collections import OrderedDict
from json import dumps as jdumps

import torch
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
import torch.distributed as dist
'''
class MerticGroup:
    def __init__(self,
                 device: torch.device | None,
                 world_size: int,
                 dst: int = 0) -> None:
        self.metrics = OrderedDict()
        self.device = device
        self.dst = dst
        self.world_size = world_size
        self.count = 0

    def add(self, metric_name: str, val: int | float) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = val
        else:
            self.metrics[metric_name] += val
    
    def share(self,) -> dict[str, float | int]:
        shared_tensor = torch.tensor([val / (self.world_size * self.count) for val in self.metrics.values()], device = self.device)
        dist.reduce(shared_tensor, dst = self.dst, op = dist.ReduceOp.SUM)
        self.metrics.clear()
        return {key: val.item() for key, val in zip(self.metrics.keys(), shared_tensor)}'''
    
class ArbitraryGroup(Metric):
    full_state_update = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="mean")
        self.add_state("count_state", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="mean")
        self.names = set()

    def update(self, name: str, value: torch.tensor, **kwargs) -> None:
        if len(kwargs) > 0:
            raise AttributeError('Unknown arguments {}'.format(kwargs))
        if not hasattr(self, name):
            self.add_state(name, default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="mean")
            self.count_state += 1
            self.names.add(name)
        setattr(self, name, value + getattr(self, name))
        self.total += 1
    
    def compute(self,) -> dict[float]:
        if self.total.remainder(self.count_state).item() != 0:
            raise RuntimeError('Unequal amount of each substate update.{}{}'.format(self.total, self.count_state))
        self.total /= self.count_state
        result = {}
        for name in self.names:
            result[name] = getattr(self, name).item() / self.total.item()
        return result

class MetricStorage:
    def __init__(self, rank: int, dst: int) -> None:
        self.rank = rank
        self.dst = dst
        self.groups = torch.nn.ModuleDict()
        self.epoch = -1
    
    def update(self, group_name: str, state_name: str, 
               val: torch.tensor, **kwargs) -> None:
        if group_name not in self.groups:
            if group_name.find('accuracy') != -1:
                self.groups[group_name] = torch.nn.ModuleDict({state_name: Accuracy(task='binary',
                                                                                    threshold=0.5,
                                                                                    multidim_average='global',
                                                                                    sync_on_compute=True)})
            else:
                self.groups[group_name] = ArbitraryGroup()
        if isinstance(self.groups[group_name], torch.nn.ModuleDict) and \
           state_name not in self.groups[group_name]:
            self.groups[group_name][state_name] = Accuracy(task='binary', threshold=0.5,
                                                           multidim_average='global',
                                                           sync_on_compute=True)
        if isinstance(self.groups[group_name], torch.nn.ModuleDict):
            self.groups[group_name][state_name].update(val, **kwargs)
        else:
            self.groups[group_name].update(state_name, val, **kwargs)
    
   
    def compute(self,) -> None:
        result = {}
        for name, sub_group in self.groups.items():
            if isinstance(sub_group, torch.nn.ModuleDict):
                result[name] = {}
                for key, met in sub_group.items():
                    result[name][key] = met.compute().item()
            else:
                result[name] = sub_group.compute()

        result['epoch'] = self.epoch
        if self.rank == self.dst:
            # Store metrics in JSON format to simplify parsing 
            # and transferring them into tensorboard at initial machine.
            print(jdumps(result))
    
    def reset(self,) -> None:
        for sub_group in self.groups.values():
            if isinstance(sub_group, torch.nn.ModuleDict):
                for met in sub_group.values():
                    met.reset()
            else:
                sub_group.reset()


'''

class MetricStorage:
    def __init__(self, rank: int, device: torch.device, world_size: int, num_batch: int, dst: int) -> None:
        self.groups = OrderedDict()
        self.rank = rank
        self.device = device
        self.world_size = world_size
        self.dst = dst
        self.epoch = -1

    def update(self, group_name: str, metric_name: str, val: float | int) -> None:
        if group_name not in self.groups:
            self.groups[group_name] = MerticGroup(self.device, self.world_size, self.dst)
        self.groups[group_name].add(metric_name, val)
    
    def set_epoch(self, val: int) -> None:
        self.epoch = val
    
    def share(self,) -> None:
        result = {}
        for group_name in self.groups.keys():
            metrics = self.groups[group_name].share()
            if self.rank == self.dst:
                result[group_name] = metrics
        result['epoch'] = self.epoch

        if self.rank == self.dst:        
            # Send metrics into stdout. This channel going to be transferred into initial machine.
            # Store metrics in JSON format to simplify parsing and transferring them into tensorboard at initial machine.
            print(jdumps(result))

            
'''