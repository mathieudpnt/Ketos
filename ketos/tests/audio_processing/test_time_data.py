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
    o, d = time_data_id
    assert o.ndim == 1
    assert o.filename == 'x'
    assert o.offset == 2.
    assert o.label == 13

def test_init_stacked(time_data_1d_stacked):
    """Test if a stacked TimeData object has expected attribut values"""
    o, d = time_data_id_stacked
    assert np.all(o.get_data(1) == d[:,1])
    assert o.ndim == 1
    assert o.filename == ['x','x','x']
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