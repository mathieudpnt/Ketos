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

""" Unit tests for the 'time_data' module within the ketos library
"""
import pytest
import ketos.audio_processing.time_data as td
import numpy as np


def test_init(time_data_1d):
    """Test if the TimeData object has expected attribute values"""
    o, d = time_data_1d
    assert o.ndim == 1
    assert o.filename == 'x'
    assert o.offset == 2.
    assert o.label == 13

def test_init_stacked(time_data_1d_stacked):
    """Test if a stacked TimeData object has expected attribut values"""
    o, d = time_data_1d_stacked
    assert np.all(o.get_data(1) == d[:,1])
    assert o.ndim == 1
    assert o.filename == ['x','y','z']
    assert np.all(o.offset == 2.)
    assert np.all(o.label == 13)

def test_get_data(time_data_1d):
    """Test that the get_data method works as it should"""
    o, d = time_data_1d
    assert np.all(o.get_data() == d)
    assert np.all(o.get_data(0) == d)

def test_get_data_stacked(time_data_1d_stacked):
    """Test that the get_data method works as it should for stacked objects"""
    o, d = time_data_1d_stacked
    for i in range(3):
        assert np.all(o.get_data(i) == d[:,i])
    
def test_crop(time_data_1d):
    """Test if a cropped TimeData object has the expected content and length"""
    o, d = time_data_1d
    oc = o.crop(start=0.2, end=3.8)
    assert oc.length() == 3.6
    assert np.all(oc.get_data() == d[200:3800])

def test_annotations_returns_none(time_data_1d):
    """Test that no annotations are returned when none are provided"""
    o, d = time_data_1d
    assert o.annotations() == None

def test_time_res(time_data_1d):
    """Test if time resolution is correct"""
    o, d = time_data_1d
    assert o.time_res() == 0.001

def test_deep_copy(time_data_1d):
    """Test that changes to copy to not affect original instance"""
    o, d = time_data_1d
    oc = o.deepcopy()
    oc.filename = 'y'
    oc.time_ax.label = 'new axis'
    assert o.filename == 'x'
    assert o.time_ax.label == 'Time (s)'

def test_normalize_stacked(time_data_1d_stacked):
    """Test that stacked object is properly normalized"""
    N = 10000
    d = np.arange(N)
    d = np.concatenate([d,2*d])
    o = td.TimeData(time_res=0.001, data=d, ndim=1, filename='x', offset=2., label=13)
    o.normalize()
    assert np.all(np.min(o.data, axis=0) == 0)
    assert np.all(np.max(o.data, axis=0) == 1)

def test_segment(time_data_1d):
    """Test segment method on 1d object"""
    o, d = time_data_1d
    s = o.segment(window=2, step=1) #integer number of steps
    assert s.ndim == o.ndim
    assert s.data.shape == (2000,9)
    assert np.all(s.label == 13)
    s = o.segment(window=2, step=1.1) #non-integer number of steps
    assert s.ndim == o.ndim
    assert s.data.shape == (2000,9)
    assert np.all(s.data[1200:,-1] == 0) #last frame was padded with zeros

def test_segment_stacked(time_data_1d_stacked):
    """Test segment method on stacked 1d object"""
    o, d = time_data_1d_stacked
    s = o.segment(window=2, step=1) 
    assert s.ndim == o.ndim
    assert s.data.shape == (2000, 9*3)
    assert np.all(s.data[:,:9] == 1)
    assert np.all(s.data[:,9:18] == 2)
    assert np.all(s.data[:,18:27] == 3)
    assert s.filename[:9] == ['x' for _ in range(9)]
    assert s.filename[9:18] == ['y' for _ in range(9)]
    assert s.filename[18:27] == ['z' for _ in range(9)]

def test_annotate(time_data_1d):
    """Test that we can add annotations"""
    o, d = time_data_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=2, start=1.8, end=2.2) 
    assert o.annot.num_annotations() == 2

def test_label_array(time_data_1d):
    """Check that method label_array returns expected array"""
    o, d = time_data_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=1, start=1.8, end=2.2) 
    res = o.label_array(1)
    ans = np.concatenate([np.zeros(200),np.ones(1100),np.zeros(500),np.ones(400),np.zeros(7800)]) 
    assert np.all(res == ans)

def test_segment_with_annotations(time_data_1d):
    """Test segment method on 1d object with annotations"""
    o, d = time_data_1d
    o.annotate(label=1, start=0.2, end=1.3) #fully contained in first segment, partially contained in second
    o.annotate(label=2, start=1.8, end=2.2) #partially contained in first, second and third segment
    s = o.segment(window=2, step=1) 
    df0 = s.annotations(id=0) #annotations for 1st segment
    assert len(df0) == 2
    assert np.all(df0['start'].values == [0.2, 1.8])
    assert np.all(df0['end'].values == [1.3, 2.0])
    df1 = s.annotations(id=1) #annotations for 2nd segment
    assert len(df1) == 2
    assert np.all(df1['start'].values == [0.0, 0.8])
    assert np.all(np.abs(df1['end'].values - [0.3, 1.2]) < 1e-9)

