
import pytest
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat, NNInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
from ketos.data_handling.data_feeding import BatchGenerator
import os
import shutil
import tables
import json


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_compute_avg_score():
    pass

def test_match_detection_to_time():
    pass

def test_group_detections():
    pass

def test_process_batch():
    pass

def transform_batch():
    pass

def test_process_audio_loader():
    pass

def test_process_batch_generator():
    pass

def test_merge_overlapping_detections():
    pass


def test_save_detections():
    pass
