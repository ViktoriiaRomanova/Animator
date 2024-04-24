from dataclasses import dataclass, is_dataclass

def nested_dataclass(*args, **kwargs):
    def wrapper(in_cls):
        in_cls = dataclass(in_cls, **kwargs)
        original_init = in_cls.__init__
        def __init__(self, *args, **kwargs):
            for name, val in kwargs.items():
                field_type = in_cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(val, dict):
                    kwargs[name] = field_type(**val)
            original_init(self, *args, **kwargs)
        in_cls.__init__ = __init__
        return in_cls
    return wrapper(args[0]) if args else wrapper

@dataclass(kw_only = True)
class DataParams:
    data_part: float
    sub_part_data: float

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


@nested_dataclass
class AllOptimParams:
    gen: OptimParams
    discA: OptimParams
    discB: OptimParams

@dataclass(kw_only = True)
class ModelParams:
    mean: float
    std: float

@dataclass(kw_only = True)
class MainParams:
    random_state: int
    batch_size: int
    epochs: int
    buffer_size: int


@nested_dataclass
class TrainingParams:
    main: MainParams    
    data: DataParams
    distributed: DistParams
    optimizers: AllOptimParams
    models: ModelParams


