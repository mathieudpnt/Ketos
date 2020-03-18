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

def test_init_stacked(base_audio_1d_stacked):
    """Test if a stacked BaseAudio object has expected attribute values"""
    o, d = base_audio_1d_stacked
    assert np.all(o.data[:,1] == d[:,1])
    assert o.ndim == 1
    assert np.all(o.filename == ['x','yy','z'])
    assert np.all(o.offset == 2.)
    assert np.all(o.label == 13)

def test_data(base_audio_1d):
    """Test that the data attribute has expect contents"""
    o, d = base_audio_1d
    assert np.all(o.data == d)

def test_data_stacked(base_audio_1d_stacked):
    """Test that the data attribute of a stacked BaseAudio object has expect contents"""
    o, d = base_audio_1d_stacked
    assert np.all(o.data == d)

def test_get_data(base_audio_1d):
    """Test that the get_data method returns expected result"""
    o, d = base_audio_1d
    assert np.all(o.get_data() == d)

def test_get_data_stacked(base_audio_1d_stacked):
    """Test that the get_data method returns expected result for stacked objects"""
    o, d = base_audio_1d_stacked
    assert np.all(o.get_data() == d)
    assert np.all(o.get_data(1) == d[:,1])

def test_get_filename_stacked(base_audio_1d_stacked):
    """Test that the get_filename method returns expected result for stacked objects"""
    o, d = base_audio_1d_stacked
    assert np.all(o.get_filename() == ['x','yy','z'])
    assert np.all(o.get_filename(1) == 'yy')

def test_crop(base_audio_1d):
    """Test if a cropped BaseAudio object has the expected content and length"""
    o, d = base_audio_1d
    oc = o.crop(start=0.2, end=3.8)
    assert oc.duration() == 3.6
    assert np.all(oc.data == d[200:3800])

def test_annotations_returns_none(base_audio_1d):
    """Test that no annotations are returned when none are provided"""
    o, d = base_audio_1d
    assert o.get_annotations() == None

def test_time_res(base_audio_1d):
    """Test if time resolution is correct"""
    o, d = base_audio_1d
    assert o.time_res() == 0.001

def test_deep_copy(base_audio_1d):
    """Test that changes to copy to not affect original instance"""
    o, d = base_audio_1d
    oc = o.deepcopy()
    oc.filename = 'yy'
    oc.time_ax.label = 'new axis'
    assert o.filename == 'x'
    assert o.time_ax.label == 'Time (s)'

def test_normalize_stacked(base_audio_1d_stacked):
    """Test that stacked object is properly normalized"""
    N = 10000
    d = np.arange(N)
    d = np.concatenate([d,2*d])
    o = aba.BaseAudio(time_res=0.001, data=d, ndim=1, filename='x', offset=2., label=13)
    o.normalize()
    assert np.all(np.min(o.data, axis=0) == 0)
    assert np.all(np.max(o.data, axis=0) == 1)

def test_segment(base_audio_1d):
    """Test segment method on 1d object"""
    o, d = base_audio_1d
    s = o.segment(window=2, step=1) #integer number of steps
    assert s.ndim == o.ndim
    assert s.data.shape == (2000,9)
    assert np.all(s.label == 13)
    assert np.all(s.offset == np.linspace(0.,8.,9))
    s = o.segment(window=2, step=1.1) #non-integer number of steps
    assert s.ndim == o.ndim
    assert s.data.shape == (2000,9)
    assert np.all(s.data[1200:,-1] == 0) #last frame was padded with zeros
    assert np.all(s.offset == np.linspace(0.0,8.8,9))

def test_segment_stacked(base_audio_1d_stacked):
    """Test segment method on stacked 1d object"""
    o, d = base_audio_1d_stacked
    s = o.segment(window=2, step=1) 
    assert s.ndim == o.ndim
    assert s.data.shape == (2000, 3, 9)
    assert np.all(s.data[:,0,:] == 1)
    assert np.all(s.data[:,1,:] == 2)
    assert np.all(s.data[:,2,:] == 3)
    assert np.all(s.filename[0] == 'x')
    assert np.all(s.filename[1] == 'yy')
    assert np.all(s.filename[2] == 'z')
    for i in range(3):
        assert np.all(s.offset[i] == np.linspace(0.,8.,9))
        assert np.all(s.label[i] == 13)

def test_annotate(base_audio_1d):
    """Test that we can add annotations"""
    o, d = base_audio_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=2, start=1.8, end=2.2) 
    assert o.annot.num_annotations() == 2

def test_label_array(base_audio_1d):
    """Check that method label_array returns expected array"""
    o, d = base_audio_1d
    o.annotate(label=1, start=0.2, end=1.3) 
    o.annotate(label=1, start=1.8, end=2.2) 
    res = o.label_array(1)
    ans = np.concatenate([np.zeros(200),np.ones(1100),np.zeros(500),np.ones(400),np.zeros(7800)]) 
    assert np.all(res == ans)

def test_segment_with_annotations(base_audio_1d):
    """Test segment method on 1d object with annotations"""
    o, d = base_audio_1d
    o.annotate(label=1, start=0.2, end=1.3) #fully contained in first segment, partially contained in second
    o.annotate(label=2, start=1.8, end=2.2) #partially contained in first, second and third segment
    s = o.segment(window=2, step=1) 
    df0 = s.get_annotations(id=0) #annotations for 1st segment
    assert len(df0) == 2
    assert np.all(df0['start'].values == [0.2, 1.8])
    assert np.all(df0['end'].values == [1.3, 2.0])
    df1 = s.get_annotations(id=1) #annotations for 2nd segment
    assert len(df1) == 2
    assert np.all(df1['start'].values == [0.0, 0.8])
    assert np.all(np.abs(df1['end'].values - [0.3, 1.2]) < 1e-9)

def test_segment_with_annotations_stacked(base_audio_1d_stacked):
    """Test segment method on 1d object with annotations"""
    o, d = base_audio_1d_stacked
    o.annotate(label=1, start=0.2, end=1.3, id=0) #add this annotation to 1st object
    o.annotate(label=2, start=1.8, end=2.2, id=2) #add this annotation to 3rd object
    s = o.segment(window=2, step=1) 
    df00 = s.get_annotations(id=(0,0)) #annotations for 1st segment of 1st object
    assert len(df00) == 1
    assert df00['start'].values[0] == 0.2
    assert df00['end'].values[0] == 1.3
    df01 = s.get_annotations(id=(0,1)) #annotations for 2nd segment of 1st object
    assert len(df01) == 1
    assert df01['start'].values[0] == 0.
    assert pytest.approx(df01['end'].values[0], 0.3, abs=0.00001)
    df02 = s.get_annotations(id=(0,2)) #annotations for 3rd segment of 1st object
    assert df02 == None
    df20 = s.get_annotations(id=(2,0)) #annotations for 1st segment of 3rd object
    assert len(df20) == 1
    assert df20['start'].values[0] == 1.8
    assert df20['end'].values[0] == 2.0

def test_stack(base_audio_1d):
    """Test that we can stack two audio objects"""
    o, d = base_audio_1d
    stacked = aba.BaseAudio.stack([o, o])
    assert stacked.ndim == 1
    assert np.ndim(stacked.get_data()) == 2
    for i in range(2):
        assert np.all(stacked.get_data(i) == d)
        assert stacked.get_filename(i) == o.filename
        assert stacked.get_label(i) == o.label

def test_stack_with_annotations(base_audio_1d):
    """ Test that we can stack two audio objects with annotations"""
    o, d = base_audio_1d
    o.annotate(label=1, start=0.2, end=1.3)
    o.annotate(label=2, start=1.8, end=2.2)
    stacked = aba.BaseAudio.stack([o, o])
    assert stacked.ndim == 1
    assert np.ndim(stacked.get_data()) == 2
    for i in range(2):
        assert np.all(stacked.get_data(i) == d)
        assert stacked.get_filename(i) == o.filename
        assert len(stacked.get_annotations(i)) == 2

