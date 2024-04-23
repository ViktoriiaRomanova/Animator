from typing import TypedDict
from dataclasses import dataclass

def pre_init(func):
    def decorated(args, **kwargs):
        print(func.__name__)
        return func(**args)
    return decorated

@pre_init
@dataclass(kw_only = True)
class DataParams:
    data_part: float
    sub_part_data: float

@pre_init
@dataclass(kw_only = True)
class DistParams:
    world_size: int
    backend: str
    port: str
    address: str

@pre_init
@dataclass(kw_only = True)
class OptimParams:
    lr: float
    betas: tuple[float, float]

@pre_init
@dataclass(kw_only = True)
class AllOptimParams:
    gen: OptimParams
    discA: OptimParams
    discB: OptimParams

@pre_init
@dataclass(kw_only = True)
class ModelParams:
    mean: float
    std: float

@pre_init
@dataclass(kw_only = True)
class MainParams:
    random_state: int
    batch_size: int
    epochs: int
    buffer_size: int

@pre_init
@dataclass(kw_only = True)
class TrainingParams:
    main: MainParams    
    data: DataParams
    distributed: DistParams
    optimizers: AllOptimParams
    models: ModelParams


