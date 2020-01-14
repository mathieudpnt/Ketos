# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" Unit tests for the 'annotation' module within the ketos library
"""

import pytest
import unittest
import numpy as np
import pandas as pd
from ketos.audio_processing.annotation_new import AnnotationHandler


def test_empty_annotation_handler_has_correct_columns():
    handler = AnnotationHandler()
    a = handler.get()
    unittest.TestCase().assertCountEqual(list(a.columns), ['label', 'time_start', 'time_stop', 'freq_min', 'freq_max'])

def test_add_individual_annotations():
    handler = AnnotationHandler()
    # add annotation without units
    handler.add(time_start=0.0, time_stop=4.1, label=1)
    a = handler.get()
    assert len(a) == 1
    assert a['time_start'][0] == 0.0
    assert a['time_stop'][0] == 4.1
    assert np.isnan(a['freq_min'][0])
    assert np.isnan(a['freq_max'][0])
    assert a['label'][0] == 1
    # add annotation with SI units
    handler.add(time_start='2min', time_stop='5min', label=2)
    a = handler.get()
    assert len(a) == 2
    assert a['time_start'][1] == 120
    assert a['time_stop'][1] == 300
    assert np.isnan(a['freq_min'][1])
    assert np.isnan(a['freq_max'][1])
    assert a['label'][1] == 2
    # add annotation with frequency range
    handler.add(time_start='2min', time_stop='5min', freq_min='200Hz', freq_max=3000, label=3)
    a = handler.get()
    assert len(a) == 3
    assert a['freq_min'][2] == 200
    assert a['freq_max'][2] == 3000

def test_add_annotations_as_dataframe():
    handler = AnnotationHandler()
    df = pd.DataFrame({'time_start':[1,2], 'time_stop':[7,9], 'label':[14,11]})
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 2
    assert a['time_start'][0] == 1
    assert a['time_stop'][0] == 7
    assert a['time_start'][1] == 2
    assert a['time_stop'][1] == 9

def test_add_annotations_as_dict():
    handler = AnnotationHandler()
    df = {'time_start':1, 'time_stop':7, 'label':14}
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 1
    assert a['time_start'][0] == 1
    assert a['time_stop'][0] == 7

def test_crop_annotations_along_time_axis():
    handler = AnnotationHandler()
    handler.add(1, 1, 3, 0, 100)
    handler.add(2, 3, 5.2, 0, 100)
    handler.add(3, 5, 7.3, 0, 100)
    handler.add(4, 8, 10, 0, 100)
    a = handler.get()
    assert len(a) == 4
    # crop from t=4 to t=9
    # 1st annotation is fully removed, 2nd and 4th are partially cropped
    handler.crop(4, 9)
    a = handler.get()
    assert len(a) == 3
    assert np.allclose(a['time_start'], [0, 1, 4], atol=1e-08) 
    assert np.allclose(a['time_stop'], [1.2, 3.3, 5], atol=1e-08) 
    assert np.array_equal(a['label'], [2, 3, 4]) 

def test_segment_annotations():
    handler = AnnotationHandler()
    handler.add(1, 0.2, 1.1, 0, 100)
    handler.add(2, 3.1, 4.7, 0, 100)
    # divided into 1.0-second long segments with 50% overlap
    ann = handler.segment(num_segs=20, window_size=1.0, step_size=0.5)
    # the segments overlapping with the two annotations are:
    # 0) 0.0-1.0, 1) 0.5-1.5, 2) 1.0-2.0, 5) 2.5-3.5, 6) 3.0-4.0, 
    # 7) 3.5-4.5, 8) 4.0-5.0, 9) 4.5-5.5
    assert len(ann) == 8
    # check 1st segment
    a = ann[0].get()
    assert np.allclose(a['time_start'], [0.2])
    assert np.allclose(a['time_stop'], [1.0])
    assert np.array_equal(a['label'], [1]) 
    # check 2nd segment
    a = ann[1].get()
    assert np.allclose(a['time_start'], [0.0])
    assert np.allclose(a['time_stop'], [0.6])
    assert np.array_equal(a['label'], [1]) 
    # check 8th segment
    a = ann[9].get()
    assert np.allclose(a['time_start'], [0.0])
    assert np.allclose(a['time_stop'], [0.2])
    assert np.array_equal(a['label'], [2])

def test_segment_annotations_with_nonzero_start_time():
    handler = AnnotationHandler()
    handler.add(1, 0.2, 1.1, 0, 100)
    # divided into 1.0-second long segments with 50% overlap, and start 
    # time set to -0.9 seconds
    ann = handler.segment(num_segs=20, window_size=1.0, step_size=0.5, time_start='-0.9sec')
    # the segments overlapping with the two annotations are:
    # 1) -0.4-0.6, 2) 0.1-1.1, 3) 0.6-1.6
    assert len(ann) == 3
    # check 1st segment
    a = ann[1].get()
    assert np.allclose(a['time_start'], [0.6])
    assert np.allclose(a['time_stop'], [1.0])
    assert np.array_equal(a['label'], [1]) 
    # check 2nd segment
    a = ann[2].get()
    assert np.allclose(a['time_start'], [0.1])
    assert np.allclose(a['time_stop'], [1.0])
    assert np.array_equal(a['label'], [1]) 
    # check 3rd segment
    a = ann[3].get()
    assert np.allclose(a['time_start'], [0.0])
    assert np.allclose(a['time_stop'], [0.5])
    assert np.array_equal(a['label'], [1])