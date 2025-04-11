import argparse
import yaml
from abc import ABC

from .transfer_parameters import TrainingParams
from .extraction_parameters import ExtTrainingParams
from .diffusion_parameters import DiffusionTrainingParams
from .ds_diffusion_parameters import DSDiffusionTrainingParams


class BaseParamsHolder(ABC):
    def __init__(
        self,
    ) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", required=True)  # path to dataset
        parser.add_argument("--prhome", required=False)  # path to project home directory READONLY
        parser.add_argument("--omodel", required=False)  # output checkpoints of model weights
        parser.add_argument("--imodel")  # input model weights
        parser.add_argument("--params", required=True)  # path to training parameters
        parser.add_argument("--st", required=False)  # path to S3 storage to store intermediate results

         # DeepSpeed arguments
        parser.add_argument("--local_rank", type=int, required=False)
        parser.add_argument("--ds_config", default="ds_config.json", required=False)
        parser.add_argument("--ds_config_disc", default="ds_config_disc.json", required=False)
        self.datasphere_params = parser.parse_args()


class ParamsHolder(BaseParamsHolder):
    def __init__(self, ptype: str = "Transfer") -> None:
        super().__init__()
        with open(self.datasphere_params.params, "r") as file:
            if ptype == "Transfer":
                self.hyper_params = TrainingParams(**yaml.safe_load(file))
            elif ptype == "Extraction":
                self.hyper_params = ExtTrainingParams(**yaml.safe_load(file))
            elif ptype == "Diffusion":
                self.hyper_params = DiffusionTrainingParams(**yaml.safe_load(file))
            elif ptype == "DS_Diffusion":
                self.hyper_params = DSDiffusionTrainingParams(**yaml.safe_load(file))
            else:
                raise NotImplementedError("unknown type {} for parameter storage".format(ptype))
