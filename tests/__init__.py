"""Tests package."""
import os

HYPERPARAMETERS = 'train_eval/style_transfer/hyperparameters.yaml'
IMG_PATH = os.path.join(os.getcwd(), 'tests/style_transfer/test_img')
IMG_PATH_WITH_GARBAGE = os.path.join(os.getcwd(), 'tests/style_transfer/test_img_to_clean')