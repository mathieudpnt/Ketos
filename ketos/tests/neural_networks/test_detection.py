
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



@pytest.fixture
def scores_and_support_1():
    scores = np.array([0,0,0,0.1,0.3,0.4,0.52,0.89,0.78,0.6,0.4,0.3,0.4,0.2,0.1,0,0,0,0,0,0.7,0.4,0.5,0.8,0.7,0.4,0.3,0,0,0,0.3,0.4,0.3,0])
    support = np.array([('file_1.wav', 0),
                         ('file_1.wav', 0.5),
                         ('file_1.wav', 1.0),
                         ('file_1.wav', 1.5),
                         ('file_1.wav', 2.0),
                         ('file_1.wav', 2.5),
                         ('file_1.wav', 3.0),
                         ('file_1.wav', 3.5),
                         ('file_1.wav', 4.0),
                         ('file_1.wav', 4.5),
                         ('file_2.wav', 0),
                         ('file_2.wav', 0.5),
                         ('file_2.wav', 1.0),
                         ('file_2.wav', 1.5),
                         ('file_2.wav', 2.0),
                         ('file_2.wav', 2.5),
                         ('file_2.wav', 3.0),
                         ('file_2.wav', 3.5),
                         ('file_2.wav', 4.0),
                         ('file_2.wav', 4.5),
                         ('file_3.wav', 0),
                         ('file_3.wav', 0.5),
                         ('file_3.wav', 1.0),
                         ('file_3.wav', 1.5),
                         ('file_3.wav', 2.0),
                         ('file_3.wav', 2.5),
                         ('file_3.wav', 3.0),
                         ('file_3.wav', 3.5),
                         ('file_3.wav', 4.0),
                         ('file_3.wav', 4.5),
   
    ])
    return scores, support


@pytest.mark.parametrize("score_vector, win_len,  expected", [(np.array([1.,1.,1.,1.,1.,1.]), 3, np.array([0.,1.,1.,1.,1.,0.])),
                                                                (np.array([0.,0.,0.,0.,0.,0.]), 3, np.array([0.,0.,0.,0.,0.,0.])),
                                                                (np.array([1.,1.,1.,1.,1.,1.]), 5, np.array([0.,0.,1.,1.,0.,0.])),
                                                                (np.array([0.,0.,0.,0.,0.,0.]), 3, np.array([0.,0.,0.,0.,0.,0.])),
                                                                (np.array([0.1,0.35,0.7,0.8,0.5,0.3,0.2,0,0.89,0.5]), 3, np.array([0,0.3833,0.6166,0.6666,0.5333,0.3333,0.1666,0.3633,0.4633,0])),
                                                                 
                                                                 ])
def test_compute_avg_score(score_vector, win_len, expected):
    avg = compute_avg_score(score_vector, win_len)
    assert len(avg) == len(score_vector)
    assert np.allclose(avg, expected,  rtol=1e-03)


def test_compute_avg_score_raise_exception():
    with pytest.raises(AssertionError):
        avg = compute_avg_score(np.array([1,1,1,1,1]), win_len=2)


@pytest.mark.parametrize("det_start, det_end, batch_start_timestamp, batch_end_timestamp, step, spec_dur, buffer, expected", 
                        [(0,1,300,600,0.5,3.0, 1.0, (300,2.0)),
                         (0,2,300,600,0.5,3.0, 1.0, (300,2.5)),
                         (0,3,300,600,0.5,3.0, 1.0, (300,3.0)),
                         (1,4,300,600,0.5,3.0, 1.0, (300,3.5)),
                         (2,5,300,600,0.5,3.0, 1.0, (300,4.0)),
                         (2,6,300,600,0.5,3.0, 1.0, (300,4.5)),
                         (3,7,300,600,0.5,3.0, 1.0, (300.5,4.5)),
                         (4,8,300,600,0.5,3.0, 1.0, (301,4.5)),
                         (1,2,300,600,0.5,3.0, 1.0, (300,2.5)),
                         (2,2,300,600,0.5,3.0, 1.0, (300.0,2.5)),
                         (20,25,300,600,0.5,3.0, 1.0, (309.0,5.0)),
                         ])
def test_map_detection_to_time(det_start, det_end, batch_start_timestamp, batch_end_timestamp, step, spec_dur, buffer, expected):
    mapped_detection = map_detection_to_time(det_start, det_end, batch_start_timestamp, batch_end_timestamp, step, spec_dur, buffer)
    assert mapped_detection == expected

def test_map_detection_to_time_det_end_exception():
    with pytest.raises(ValueError):
        map_detection_to_time(det_start=10, det_end=9, batch_start_timestamp=300, batch_end_timestamp=600, step=0.5, spec_dur=3.0, buffer=1)



@pytest.mark.parametrize("buffer, step, spec_dur, threshold, expected", 
                        [(1.0, 0.5, 3.0, 0.5, [('file_1.wav',3.0, 3.5, 0.73),]),
                        ])
def test_group_detections(scores_and_support_1,buffer, step, spec_dur, threshold, expected):
    scores, support = scores_and_support_1
    grp_det = group_detections(scores_vector=scores, batch_support_data=support, buffer=1.0,step=0.5, spec_dur=3.0, threshold=0.5)
    assert grp_det == expected
    

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
