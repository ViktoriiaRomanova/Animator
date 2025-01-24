from torchmetrics import Metric

class DiffusionMetricGroup:
    # TODO add metric storage
    def __init__(self, type: Metric, **kwargs) -> None:
        self.metrics = {}
        self.type = type
        self.kwargs = kwargs
