import unittest
import yaml
from dataclasses import asdict

from animator.utils.parameter_storages.transfer_parameters import TrainingParams

def dict_factory(args) -> dict:
    res = {}
    for key, val in args:
        if isinstance(val, tuple):
            res[key] = list(val)
        else:
            res[key] = val
    return res

class ParameterLoadingTests(unittest.TestCase):
    def test_load_train_param_correct(self):
        path = 'train_eval/style_transfer/hyperparameters.yaml'
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        
        loaded_params = TrainingParams(**params)

        self.assertTrue(params == asdict(loaded_params, dict_factory = dict_factory))
