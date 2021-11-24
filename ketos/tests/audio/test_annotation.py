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

""" Unit tests for the 'audio.annotation' module within the ketos library
"""

import pytest
import unittest
import numpy as np
import pandas as pd
from ketos.audio.annotation import AnnotationHandler, stack_annotations


def test_empty_annotation_handler_has_correct_columns():
    handler = AnnotationHandler()
    a = handler.get()
    unittest.TestCase().assertCountEqual(list(a.columns), ['label', 'start', 'end', 'freq_min', 'freq_max'])

def test_add_individual_annotations():
    handler = AnnotationHandler()
    # add annotation without units
    handler.add(start=0.0, end=4.1, label=1)
    a = handler.get()
    assert len(a) == 1
    assert a['start'][0] == 0.0
    assert a['end'][0] == 4.1
    assert np.isnan(a['freq_min'][0])
    assert np.isnan(a['freq_max'][0])
    assert a['label'][0] == 1
    # add annotation with SI units
    handler.add(start='2min', end='5min', label=2)
    a = handler.get()
    assert len(a) == 2
    assert a['start'][1] == 120
    assert a['end'][1] == 300
    assert np.isnan(a['freq_min'][1])
    assert np.isnan(a['freq_max'][1])
    assert a['label'][1] == 2
    # add annotation with frequency range
    handler.add(start='2min', end='5min', freq_min='200Hz', freq_max=3000, label=3)
    a = handler.get()
    assert len(a) == 3
    assert a['freq_min'][2] == 200
    assert a['freq_max'][2] == 3000

def test_add_annotations_as_dataframe():
    handler = AnnotationHandler()
    df = pd.DataFrame({'start':[1,2], 'end':[7,9], 'label':[14,11]})
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 2
    assert a['start'][0] == 1
    assert a['end'][0] == 7
    assert a['start'][1] == 2
    assert a['end'][1] == 9

def test_add_annotations_as_dict():
    handler = AnnotationHandler()
    df = {'start':1, 'end':7, 'label':14} #single annotation
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 1
    assert a['start'][0] == 1
    assert a['end'][0] == 7
    df={'label':[1,2], 'start':[1,1.5], 'end':[2,2.5], 'freq_min':[3,3.5], 'freq_max':[4,4.5]} #multiple annotations
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 3
    assert a['start'][1] == 1
    assert a['start'][2] == 1.5

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
    assert np.allclose(a['start'], [0, 1, 4], atol=1e-08) 
    assert np.allclose(a['end'], [1.2, 3.3, 5], atol=1e-08) 
    assert np.array_equal(a['label'], [2, 3, 4]) 

def test_crop_annotations_along_freq_axis():
    handler = AnnotationHandler()
    handler.add(1, 1, 3, 20, 100)
    handler.add(2, 3, 5.2, 40, 200)
    handler.add(3, 3, 5.2, 300, 400)
    a = handler.get()
    assert len(a) == 3
    handler.crop(start=None, end=None, freq_min=25, freq_max=150)
    a = handler.get()
    assert len(a) == 2
    assert np.allclose(a['freq_min'], [25, 40], atol=1e-08) 
    assert np.allclose(a['freq_max'], [100, 150], atol=1e-08) 
    assert np.array_equal(a['label'], [1, 2]) 

def test_segment_annotations():
    handler = AnnotationHandler()
    handler.add(1, 0.2, 1.1, 0, 100)
    handler.add(2, 3.3, 4.7, 0, 100)
    handler.add(3, 4.2, 5.1, 0, 100)
    # divided into 1.0-second long segments with 50% overlap
    handler = handler.segment(num_segs=10, window=1.0, step=0.5)
    # 1st segment
    a1 = handler.get(0)
    expected = np.sort([100.0,0.0,1,0.2,1.0])
    result = np.sort(a1.to_numpy())
    assert np.all(expected == result)
    assert handler._df.index.nlevels == 3

def test_stack_annotations():
    h1 = AnnotationHandler()
    h1.add(1, 0.2, 1.1, 50, 200)
    h2 = AnnotationHandler()
    h2.add(1, 0.1, 0.8, 20, 100)
    h2.add(2, 3.1, 4.7, 0, 30)
    h3 = AnnotationHandler()
    h3.add(4, 0.4, 0.9)
    h3.add(12, 2.9, 3.2)
    h3.add(17, 3.3, 4.1)
    h = stack_annotations([h1,h2,h3]) #stacked handler
    assert h.num_sets() == 3
    h0val = np.sort(h.get(id=0).to_numpy())
    expected = np.sort(np.array([1, 0.2, 1.1, 50, 200]))
    assert np.all(h0val == expected) # check that annotations match for handler #0

def test_add_annotations_to_multiple_sets():
    h = AnnotationHandler()
    h.add(1, 0.2, 1.1, 50, 200, id=2)
    h.add(2, 3.1, 4.7, 0, 30, id=0)
    h.add(1, 13.1, 14.7, id=2)
    assert h.num_sets() == 2
    assert h.num_annotations() == 3

def test_add_annotations_to_stacked_handler():
    h1 = AnnotationHandler()
    h1.add(1, 0.2, 1.1, 0, 100)
    h2 = AnnotationHandler()
    h2.add(13, 0.25, 1.15, 0, 200)
    handler = stack_annotations([h1,h2])
    handler.add(14, 3.35, 4.75, 0, 200, id=1)
    a = handler.get()
    val = np.sort(a.loc[1,1].values)
    expected = np.sort(np.array([200.0, 0.0, 14, 3.35, 4.75]))
    assert np.all(val == expected)

def test_segment_stacked_annotations():
    h1 = AnnotationHandler()
    h1.add(1, 0.2, 1.1, 0, 100)
    h1.add(2, 3.3, 4.7, 0, 100)
    h1.add(3, 4.2, 5.1, 0, 100)
    h2 = AnnotationHandler()
    h2.add(13, 0.25, 1.15, 0, 200)
    h2.add(14, 3.35, 4.75, 0, 200)
    h2.add(15, 4.25, 5.15, 0, 200)
    handler = stack_annotations([h1,h2])
    # divided into 1.0-second long segments with 50% overlap
    handler = handler.segment(num_segs=10, window=1.0, step=0.5)
    # 1st annotation set, 1st segment
    a11 = handler.get((0,0))
    expected = np.sort([100.0,0.0,1,0.2,1.0])
    result = np.sort(a11.to_numpy())
    assert np.all(expected == result)
    assert handler._df.index.nlevels == 3

def test_segment_stacked_annotations_with_custom_ids():
    h = AnnotationHandler()
    h.add(1, 0.2, 1.1, 50, 200, id=2)
    h.add(2, 3.1, 4.7, 0, 30, id=0)
    h.add(1, 13.1, 14.7, id=2)
    h = h.segment(num_segs=10, window=1.0, step=0.5)
    assert h._df.index.nlevels == 3

def test_get_single_annotation():
    h = AnnotationHandler()
    h.add(1, 0.2, 1.1, 50, 200)
    ann = h.get(id=0)
    assert len(ann) == 1
    assert ann['label'][0] == 1
    assert ann['start'][0] == 0.2
    assert ann['end'][0] == 1.1
    assert ann['freq_min'][0] == 50.
    assert ann['freq_max'][0] == 200.

def test_add_annotations_with_extra_columns():
    handler = AnnotationHandler()
    df = {'start':1, 'end':7, 'label':14, 'comment':'whale', 'ID':4}
    handler.add(df=df)
    a = handler.get()
    assert len(a) == 1
    assert a['comment'][0] == 'whale'
    assert a['ID'][0] == 4
