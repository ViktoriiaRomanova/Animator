import argparse
import yaml
from abc import ABC

from .transfer_parameters import TrainingParams
from .extraction_parameters import ExtTrainingParams

class BaseParamsHolder(ABC):
    def __init__(self,) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', required = True) # path to dataset
        parser.add_argument('--prhome', required = False) # path to project home directory READONLY
        parser.add_argument('--omodel', required = True) # output checkpoints of model weights 
        parser.add_argument('--imodel') # input model weights

        self.datasphere_params = parser.parse_args()

class ParamsHolder(BaseParamsHolder):
    def __init__(self, additional: str, ptype: str = 'Transfer') -> None:
        super().__init__()
        with open(additional, 'r') as file:
            if ptype == 'Transfer':
                self.hyper_params = TrainingParams(**yaml.safe_load(file))
            elif ptype == 'Extraction':
                self.hyper_params = ExtTrainingParams(**yaml.safe_load(file))
            else:
                raise NotImplementedError('unknown type {} for parameter storage'.format(ptype))
