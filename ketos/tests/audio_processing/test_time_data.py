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


def test_init():
    """Test if the TimeData object has expected attribute values"""
    N = 10000
    d = np.ones(N)
    o = td.TimeData(time_res=0.001, data=d, ndim=1, filename='x', offset=2., label=13)
    assert np.all(o.get_data() == d)
    assert o.ndim == 1
    assert o.filename == 'x'
    assert o.offset == 2.
    assert o.label == 13

def test_init_stacked():
    """Test if a stacked TimeData object has expected attribut values"""
    N = 10000
    d = np.ones((N,3))
    o = td.TimeData(time_res=0.001, data=d, ndim=1, filename='x', offset=2., label=13)
    assert np.all(o.get_data(1) == d[:,1])
    assert o.ndim == 1
    assert o.filename == ['x','x','x']
    assert np.all(o.offset == 2.)
    assert np.all(o.label == 13)

def test_crop(sine_audio):
    """Test if a cropped TimeData object has the expected content and length"""
    N = 10000 # 10 seconds at 1kHz
    d = np.ones(N)
    o = td.TimeData(time_res=0.001, data=d, ndim=1, filename='x', offset=2., label=13)
    oc = o.crop(start=0.2, end=3.8)
    assert oc.length() == 3.6
    assert np.all(oc.get_data() == d[200:3800])