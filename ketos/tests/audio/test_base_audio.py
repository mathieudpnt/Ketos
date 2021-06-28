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

""" Unit tests for the 'audio.base_audio' module within the ketos library
"""
import pytest
import ketos.audio.base_audio as aba
import numpy as np


def test_init(base_audio_1d):
    """Test if the BaseAudio object has expected attribute values"""
    o, d = base_audio_1d
    assert o.ndim == 1
    assert o.filename == 'x'
    assert o.offset == 2.
    assert o.label == 13

def test_data(base_audio_1d):
    """Test that the data attribute has expect contents"""
    o, d = base_audio_1d
    assert np.all(o.data == d)

def test_get_data(base_audio_1d):
    """Test that the get_data method returns expected result"""
    o, d = base_audio_1d
    assert np.all(o.get_data() == d)

def test_crop(base_audio_time_1d):
    """Test if a cropped BaseAudio object has the expected content and length"""
    o, d = base_audio_time_1d
    oc = o.crop(start=0.2, end=3.8)
    assert oc.duration() == 3.6
    assert np.all(oc.data == d[200:3800])

def test_annotations_returns_none(base_audio_1d):
    """Test that no annotations are returned when none are provided"""
    o, d = base_audio_1d
    assert o.get_annotations() == None

def test_time_res(base_audio_time_1d):
    """Test if time resolution is correct"""
    o, d = base_audio_time_1d
    assert o.time_res() == 0.001

def test_deep_copy(base_audio_time_1d):
    """Test that changes to copy to not affect original instance"""
    o, d = base_audio_time_1d
    oc = o.deepcopy()
    oc.filename = 'yy'
    oc.time_ax.label = 'new axis'
    assert o.filename == 'x'
    assert o.time_ax.label == 'Time (s)'

def test_normalize(base_audio_1d):
    """Test that object is properly normalized"""
    N = 10000
    d = np.arange(N)
    o = aba.BaseAudio(data=d, filename='x', offset=2., duration=10., label=13)
    o.normalize()
    assert np.all(np.isclose(np.mean(o.data, axis=0), 0, atol=1e-10))
    assert np.all(np.isclose(np.std(o.data, axis=0), 1, atol=1e-10))
    o.normalize(mean=0.2, std=7.02)
    assert np.all(np.isclose(np.mean(o.data, axis=0), 0.2, atol=1e-10))
    assert np.all(np.isclose(np.std(o.data, axis=0), 7.02, atol=1e-10))

def test_adjust_range(base_audio_1d):
    """Test that object isproperly transformed to the specified range"""
    N = 10000
    d = np.arange(N)
    o = aba.BaseAudio(data=d, filename='x', offset=2., duration=10., label=13)
    o.adjust_range()
    assert np.all(np.min(o.data, axis=0) == 0)
    assert np.all(np.max(o.data, axis=0) == 1)
    o.adjust_range(range=(-1,3.5))
    assert np.all(np.min(o.data, axis=0) == -1)
    assert np.all(np.max(o.data, axis=0) == 3.5)

def test_segment(base_audio_time_1d):
    """Test segment method on 1d object"""
    o, d = base_audio_time_1d
    ss = o.segment(window=2, step=1) #integer number of steps
    assert len(ss) == 9
    for i,s in enumerate(ss):
        assert s.ndim == o.ndim
        assert s.data.shape == (2000,)
        assert s.label == 13
        assert s.offset == o.offset + 1.0 * i
    ss = o.segment(window=2, step=1.1) #non-integer number of steps
    assert len(ss) == 9
    for i,s in enumerate(ss):
        assert s.ndim == o.ndim
        assert s.data.shape == (2000,)
        if i==len(ss)-1: assert np.all(s.data[1200:] == 0) #last frame was padded with zeros
        assert s.offset == o.offset + 1.1 * i

def test_annotate(base_audio_1d):
    """Test that we can add annotations"""
    o, d = base_audio_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=2, start=1.8, end=2.2) 
    assert o.annot.num_annotations() == 2

def test_label_array(base_audio_time_1d):
    """Check that method label_array returns expected array"""
    o, d = base_audio_time_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=1, start=1.8, end=2.2) 
    res = o.label_array(1)
    ans = np.concatenate([np.zeros(200),np.ones(1100),np.zeros(500),np.ones(400),np.zeros(7800)]) 
    assert np.all(res == ans)

def test_segment_with_annotations(base_audio_time_1d):
    """Test segment method on 1d object with annotations"""
    o, d = base_audio_time_1d
    o.annotate(label=1, start=0.2, end=1.3) #fully contained in first segment, partially contained in second
    o.annotate(label=2, start=1.8, end=2.2) #partially contained in first, second and third segment
    ss = o.segment(window=2, step=1) 
    df0 = ss[0].get_annotations() #annotations for 1st segment
    assert len(df0) == 2
    assert np.all(df0['start'].values == [0.2, 1.8])
    assert np.all(df0['end'].values == [1.3, 2.0])
    df1 = ss[1].get_annotations() #annotations for 2nd segment
    assert len(df1) == 2
    assert np.all(df1['start'].values == [0.0, 0.8])
    assert np.all(np.abs(df1['end'].values - [0.3, 1.2]) < 1e-9)

def test_get_attrs():
    """Test that base audio returns attrs"""
    N = 10000
    d = np.arange(N)
    o = aba.BaseAudio(data=d, filename='x', offset=2., duration=10., label=13, confidence=0.5)
    x = o.get_instance_attrs()
    assert x == {'filename':'x', 'offset':2., 'duration':10., 'label':13, 'confidence':0.5}
    o = aba.BaseAudioTime(data=d, time_res=0.001, filename='x', offset=2., label=13, confidence=0.5)
    x = o.get_instance_attrs()
    assert x == {'filename':'x', 'offset':2., 'label':13, 'confidence':0.5}
