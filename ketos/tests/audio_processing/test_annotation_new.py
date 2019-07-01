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
from ketos.audio_processing.annotation_new import AnnotationHandler, AudioSourceHandler


def test_empty_annotation_handler_has_correct_columns():
    handler = AnnotationHandler()
    a = handler.get()
    unittest.TestCase().assertCountEqual(list(a.columns), ['start', 'stop', 'low', 'high', 'label'])

def test_add_individual_annotations():
    handler = AnnotationHandler()
    # add simple annotation
    handler.add(start=0.0, stop=4.1, label=1)
    a = handler.get()
    assert len(a) == 1
    assert a['start'][0] == 0.0
    assert a['stop'][0] == 4.1
    assert np.isnan(a['low'][0])
    assert np.isnan(a['high'][0])
    assert a['label'][0] == 1
    # add annotation with SI units
    handler.add(start='2min', stop='5min', label=2)
    a = handler.get()
    assert len(a) == 2
    assert a['start'][1] == 120
    assert a['stop'][1] == 300
    assert np.isnan(a['low'][1])
    assert np.isnan(a['high'][1])
    assert a['label'][1] == 2
    # add annotation with frequency range
    handler.add(start='2min', stop='5min', low='200Hz', high=3000, label=3)
    a = handler.get()
    assert len(a) == 3
    assert a['low'][2] == 200
    assert a['high'][2] == 3000

def test_add_annotations_as_dataframe():
    handler = AnnotationHandler()
    df = pd.DataFrame({'start':[1,2], 'stop':[7,9], 'label':[14,11]})
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 2
    assert a['start'][0] == 1
    assert a['stop'][0] == 7
    assert a['start'][1] == 2
    assert a['stop'][1] == 9

def test_add_annotations_as_dict():
    handler = AnnotationHandler()
    df = {'start':1, 'stop':7, 'label':14}
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 1
    assert a['start'][0] == 1
    assert a['stop'][0] == 7

def test_crop_annotations_along_time_axis():
    handler = AnnotationHandler()
    handler.add(1, 3, 0, 100, 1)
    handler.add(3, 5, 0, 100, 2)
    handler.add(5, 7, 0, 100, 3)
    handler.add(8, 10, 0, 100, 4)
    a = handler.get()
    assert len(a) == 4
    handler.crop(4, 9)
    a = handler.get()
    assert len(a) == 3
    assert np.array_equal(a['start'], [0, 1, 4]) 
    assert np.array_equal(a['stop'], [1, 3, 5]) 
    assert np.array_equal(a['label'], [2, 3, 4]) 


    ### CONTINUE HERE ###


def test_segment_annotations():
    handler = AnnotationHandler()
    handler.add(0.2, 1.1, 0, 100, 1)
    handler.add(3.1, 4.7, 0, 100, 2)
    ann = handler.segment(num_segs=20, window_size=1.0, step_size=0.5)
    for x in ann:
        print('-----------------------------')
        print(x)
        print(ann[x].get())

def test_add_audio_tags():
    tracker = AudioSourceHandler()
    tracker.add(start=0.0, stop=4.1, label='test1.wav')
    tracker.add(4.1, 8.3, 'test2.wav', offset=700.)
    print(tracker.get())

def test_crop_audio_tags():
    tracker = AudioSourceHandler()
    tracker.add(start=0.0, stop=4.1, label='test1.wav')
    tracker.add(4.1, 8.3, 'test2.wav', offset=700.)
    tracker.crop(2.0, 10.0)
    print(tracker.get())

def test_segment_audio_tags():
    tracker = AudioSourceHandler()
    tracker.add(start=0.0, stop=4.1, label='test1.wav')
    tracker.add(4.1, 8.3, 'test2.wav', offset=700.)
    ann = tracker.segment(num_segs=20, window_size=1.0, step_size=0.5)
    for x in ann:
        print('-----------------------------')
        print(x)
        print(ann[x].get())
