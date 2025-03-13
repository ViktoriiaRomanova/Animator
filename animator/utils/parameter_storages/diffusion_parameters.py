from dataclasses import dataclass
from ._nested_dataclass import nested_dataclass


@dataclass(kw_only=True)
class DataParams:
    data_part: float
    sub_part_data: float
    size: list[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self):
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)


@dataclass(kw_only=True)
class DistParams:
    world_size: int
    shared_gpu: int
    backend: str
    port: str
    address: str


@dataclass(kw_only=True)
class OptimParams:
    lr: float
    betas: tuple[float, float]

    def __post_init__(self):
        self.betas = tuple(self.betas)

    weight_decay: float


@nested_dataclass
class AllOptimParams:
    gen: OptimParams
    discA: OptimParams
    discB: OptimParams


@dataclass(kw_only=True)
class GeneratorParams:
    unet_lora_rank: int
    vae_lora_rank: int
    gamma: float


@dataclass(kw_only=True)
class DiscriminatorParams:
    cv_type: str
    loss_type: str


@dataclass(kw_only=True)
class MainParams:
    random_state: int
    batch_size: int
    epochs: int
    buffer_size: int
    caption_forward: str
    caption_reverse: str
    segmentation_model: str | None = None
    segmentation_model_type: str = "B"
    save_step: int
    warm_up: int


@dataclass(kw_only=True)
class BaseLossParams:
    ltype: str


@dataclass(kw_only=True)
class AdversarialParams:
    adv_alpha: float


@dataclass(kw_only=True)
class CycleParams(BaseLossParams):
    lambda_A: float
    lambda_B: float
    lambda_lpips: float


@dataclass(kw_only=True)
class IdentityParams(BaseLossParams):
    lambda_idn: float
    lambda_lpips: float


@nested_dataclass
class LossParams:
    adversarial: AdversarialParams
    cycle: CycleParams
    identity: IdentityParams


@nested_dataclass
class DiffusionTrainingParams:
    main: MainParams
    data: DataParams
    distributed: DistParams
    optimizers: AllOptimParams
    generator: GeneratorParams
    discriminator: DiscriminatorParams
    loss: LossParams
