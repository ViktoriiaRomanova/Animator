import argparse
import yaml

from animator.utils.parameter_storages import TrainingParams

class ParamsHolder:
    def __init__(self, additional: str) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', required = True) # path to dataset
        parser.add_argument('--prhome', required = False) # path to project home directory READONLY
        parser.add_argument('--omodel', required = True) # output checkpoints of model weights 
        parser.add_argument('--imodel') # input model weights

        self.datasphere_params = parser.parse_args()
        with open(additional, 'r') as file:
            self.hyper_params = TrainingParams(**yaml.safe_load(file))
