"""Tests package."""
import os
DATA_PATH = 'tests/style_transfer/test_img'
HYPERPARAMETERS = 'train_eval/style_transfer/hyperparameters.yaml'
DATA_PATH_WITH_GARBAGE = os.path.join(os.getcwd(), 'tests/style_transfer/test_img_to_clean')