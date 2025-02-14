from json import dumps as jdumps

import torch
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance as FID


class ArbitraryMetric(Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("arb_val", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor) -> None:
        self.arb_val += value
        self.total += 1

    def compute(
        self,
    ) -> float:
        return self.arb_val.item() / self.total.item()


class MetricGroup:
    def __init__(self, type: Metric, **kwargs) -> None:
        self.metrics = {}
        self.type = type
        self.kwargs = kwargs

    def update(self, name: str, val: torch.Tensor, target: torch.Tensor | None) -> None:
        if name not in self.metrics:
            self.metrics[name] = self.type(**self.kwargs).to(val.device)
        if target is not None:
            self.metrics[name].update(val, real=False)
            self.metrics[name].update(target, real=True)
        else:
            self.metrics[name].update(val)

    def compute(
        self,
    ) -> dict[float]:
        result = {}
        for name in self.metrics:
            values = self.metrics[name].compute()
            result[name] = values.cpu().numpy().tolist() if isinstance(values, torch.Tensor) else values
        return result

    def reset(
        self,
    ) -> None:
        for metric in self.metrics.values():
            metric.reset()


class DiffusionMetricStorage:
    def __init__(self, rank: int, dst: int) -> None:
        self.rank = rank
        self.dst = dst
        self.groups = {}
        self.epoch = -1

    def update(
        self, group_name: str, state_name: str, val: torch.Tensor, target: torch.Tensor | None = None
    ) -> None:
        if group_name not in self.groups:
            if group_name.find("FID") != -1:
                self.groups[group_name] = MetricGroup(FID, feature=2048, normalize=True, sync_on_compute=True)
            else:
                self.groups[group_name] = MetricGroup(ArbitraryMetric)
        self.groups[group_name].update(state_name, val, target)

    def compute(
        self,
    ) -> None:
        result = {}
        for name, sub_group in self.groups.items():
            result[name] = sub_group.compute()

        result["epoch"] = self.epoch
        if self.rank == self.dst:
            # Store metrics in JSON format to simplify parsing
            # and transferring them into tensorboard at initial machine.
            print(jdumps(result))

    def reset(
        self,
    ) -> None:
        for sub_group in self.groups.values():
            sub_group.reset()
