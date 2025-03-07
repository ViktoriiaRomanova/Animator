from argparse import Namespace

from animator.diffusion.dist_learning_model import DiffusionDistLearning
from animator.utils.parameter_storages.diffusion_parameters import DiffusionTrainingParams

def worker(rank: int, args: Namespace, params: DiffusionTrainingParams,
           train_data: list[list[str], list[str]],
           val_data: list[list[str], list[str]] | None) -> None:

    dist_process = DiffusionDistLearning(rank, args, params, train_data, val_data)
    dist_process.execute()
