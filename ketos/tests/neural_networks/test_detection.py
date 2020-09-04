
import pytest
from ketos.neural_networks.dev_utils.detection import *
import numpy as np
import os
import shutil
import tables
import json


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.mark.parametrize("score_vector, win_len,  expected", [(np.array([1.,1.,1.,1.,1.,1.]), 3, np.array([0.,1.,1.,1.,1.,0.])),
                                                                (np.array([0.,0.,0.,0.,0.,0.]), 3, np.array([0.,0.,0.,0.,0.,0.])),])
def test_compute_avg_score(score_vector, win_len, expected):
    avg = compute_avg_score(score_vector, win_len)
    assert len(avg) == len(score_vector)
    assert np.array_equal(avg, expected)


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
