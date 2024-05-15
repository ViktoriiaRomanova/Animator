from argparse import Namespace

from animator.figure_extraction.extr_dist_learning_model import ExtractionDistLearning
from animator.utils.parameter_storages.extraction_parameters import ExtTrainingParams

def worker(rank: int, args: Namespace, params: ExtTrainingParams,
           train_data: list[list[str], list[str]],
           val_data: list[list[str], list[str]]) -> None:

    dist_process = ExtractionDistLearning(rank, args, params, train_data, val_data)
    dist_process.execute()
