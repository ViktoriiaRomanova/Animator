from dataclasses import dataclass
from ._nested_dataclass import nested_dataclass

@dataclass(kw_only = True)
class DataParams:
    data_part: float
    sub_part_data: float
    size: list[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    def __post_init__(self):
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)

@dataclass(kw_only = True)
class DistParams:
    world_size: int
    backend: str
    port: str
    address: str

@dataclass(kw_only = True)
class OptimParams:
    lr: float
    betas: tuple[float, float]
    def __post_init__(self):
        self.betas = tuple(self.betas)


@dataclass(kw_only = True)
class ModelParams:
    mtype: str
    marchitecture: str

@dataclass(kw_only = True)
class MainParams:
    random_state: int
    batch_size: int
    epochs: int

@dataclass(kw_only = True)
class BaseLossParams:
    ltype: str

@nested_dataclass
class LossParams:
    pass

@dataclass
class Metrics:
    threshhold: float

@nested_dataclass
class ExtTrainingParams:
    main: MainParams    
    data: DataParams
    distributed: DistParams
    optimizers: OptimParams
    model: ModelParams
    metrics: Metrics
