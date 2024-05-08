from argparse import Namespace

from animator.style_transfer.dist_learning_model import DistLearning
from animator.utils.parameter_storages import TrainingParams

def worker(rank: int, args: Namespace, params: TrainingParams,
           train_data: list[list[str], list[str]],
           val_data: list[list[str], list[str]] | None) -> None:

    dist_process = DistLearning(rank, args, params, train_data, val_data)
    dist_process.execute()
