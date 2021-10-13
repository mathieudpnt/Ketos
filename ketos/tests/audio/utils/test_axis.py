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

""" Unit tests for the 'audio.utils.axis' module within the ketos library
"""
import os
import pytest
import copy
import numpy as np
from ketos.audio.utils.axis import LinearAxis, Log2Axis

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

def test_linear_axis_get_bin_single_value(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin(0.6)
    assert b == 1

def test_linear_axis_get_low_edge_single_bin(linear_axis_200):
    ax = linear_axis_200
    x = ax.low_edge(0)
    assert x == 0.

def test_linear_axis_get_bin_several_values(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin([0.6,11.1])
    assert np.all(b == [1,22])

def test_linear_axis_get_low_edge_several_bins(linear_axis_200):
    ax = linear_axis_200
    x = ax.low_edge([0, 199])
    assert np.all(x == [0, 99.5])

def test_linear_axis_get_bin_edge(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin([0.0,0.5,1.0,100.])
    assert np.all(b == [0,1,2,199])
    #Note that when the value sits between two bins, 
    #the higher bin number is returned, expect if the 
    #value sits at the upper edge of the last bin, in 
    #which case the lower bin number (i.e. the last bin)
    #is returned.  

def test_linear_axis_get_bin_outside_range(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin([-2.1, 100.1])
    assert np.all(b == [-5,200])

def test_linear_axis_get_bin_truncate(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin([-2.1, 100.1], truncate=True)
    assert np.all(b == [0,199])

def test_linear_axis_get_bin_truncate_and_closed_right(linear_axis_200):
    ax = linear_axis_200
    b = ax.bin([ax.max(), ax.max()+66.], closed_right=True, truncate=True)
    assert np.all(b == [199,199])

def test_linear_axis_resize(linear_axis_200):
    ax = copy.deepcopy(linear_axis_200)
    ax.resize(bins=50)
    assert ax.min() == linear_axis_200.min()
    assert ax.max() == linear_axis_200.max()
    assert ax.bins == 50
    assert ax.bin(98.5) == 49

def test_log2_axis_get_bin_single_value(log2_axis_8_16):
    ax = log2_axis_8_16
    b = ax.bin(10.)
    assert b == 0

def test_log2_axis_get_low_edge_single_bin(log2_axis_8_16):
    ax = log2_axis_8_16
    x = ax.low_edge(0)
    assert x == 10.

def test_log2_axis_get_bin_several_values(log2_axis_8_16):
    ax = log2_axis_8_16
    b = ax.bin([10.,20.])
    assert np.all(b == [0,16])

def test_log2_axis_get_low_edge_several_bins(log2_axis_8_16):
    ax = log2_axis_8_16
    x = ax.low_edge([0,16])
    assert np.all(x == [10.,20.])

def test_log2_axis_resize(log2_axis_8_16):
    ax = copy.deepcopy(log2_axis_8_16)
    ax.resize(bins=64) #reduce no. bins by a factor of 2
    assert ax.min() == log2_axis_8_16.min()
    assert ax.max() == log2_axis_8_16.max()
    assert ax.bins == 64
    assert ax.bins_per_oct == 8
    assert ax.bin(10.5) == 0
    assert ax.bin(ax.max()-0.5) == 63
    ax.resize(bins=62) #reduce no. bins by a non-integer factor
    assert ax.min() == log2_axis_8_16.min()
    assert ax.max() == log2_axis_8_16.max()
    assert ax.bins == 62

def test_mel_axis_get_bin_single_value(mel_axis_40_500):
    ax = mel_axis_40_500
    b = ax.bin(26.0)
    assert b == 2

def test_mel_axis_get_low_edge_single_bin(mel_axis_40_500):
    ax = mel_axis_40_500
    x = ax.low_edge(2)
    assert x == pytest.approx(23.3, abs=0.1)

def test_mel_axis_get_bin_several_values(mel_axis_40_500):
    ax = mel_axis_40_500
    b = ax.bin([28.15876149, 484.32773418])
    assert np.all(b == [2,39])

def test_mel_axis_get_low_edge_several_bins(mel_axis_40_500):
    ax = mel_axis_40_500
    x = ax.low_edge([1,39])
    assert np.all(np.isclose(x, [13.9, 476.5], atol=0.1))

def test_mel_axis_cut(mel_axis_40_500):
    b_min, b_max = mel_axis_40_500.cut(x_min=28.1, x_max=468.)
    assert b_min==2
    assert b_max==38

def test_mel_axis_resize(mel_axis_40_500):
    ax = copy.deepcopy(mel_axis_40_500)
    ax.resize(bins=22)
    assert ax.min() == mel_axis_40_500.min()
    assert ax.max() == mel_axis_40_500.max()
    assert ax.bins == 22
    assert ax.bin(499) == 21

def test_linear_axis_ticks_and_labels(linear_axis_200):
    ticks, labels = linear_axis_200.ticks_and_labels(numeric_format='.2f', num_labels=5)
    assert np.all(ticks == [0,25,50,75,100])
    assert np.all(labels == ['0.00','25.00','50.00','75.00','100.00'])

    ticks, labels = linear_axis_200.ticks_and_labels(step=20)
    assert np.all(ticks == [0,20,40,60,80,100])

    ticks, labels = linear_axis_200.ticks_and_labels(step_bins=40)
    assert np.all(ticks == [0,20,40,60,80,100])

    ticks, labels = linear_axis_200.ticks_and_labels(numeric_format='.0f', ticks=[0,50,100])
    assert np.all(ticks == [0,50,100])
    assert np.all(labels == ['0','50','100'])

def test_mel_axis_ticks_and_labels(mel_axis_40_500):
    ticks, labels = mel_axis_40_500.ticks_and_labels(numeric_format='.2f', num_labels=5)
    assert np.all(np.isclose(ticks, [0,125,250,375,500], atol=1e-9))
    assert np.all(labels == ['0.00', '103.61', '216.52', '345.28', '500.00'])
    ticks, labels = mel_axis_40_500.ticks_and_labels(numeric_format='.1f', num_labels=5, significant_figures=2)
    assert np.all(np.isclose(ticks, [0,120.71712989,253.60852227,379.28287011,500.], atol=1e-6))
    assert np.all(labels == ['0.0', '100.0', '220.0', '350.0', '500.0'])
