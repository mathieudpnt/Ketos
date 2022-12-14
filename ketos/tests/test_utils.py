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

""" Unit tests for the 'utils' module within the ketos library
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from ketos.utils import tostring, morlet_func, octave_bands, random_floats, nearest_values,\
    detect_peaks, str_is_int, signif, ensure_dir, fractional_overlap, floor, ceil, floor_round_up, ceil_round_down


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(current_dir, "assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_fractional_overlap():
    f = fractional_overlap(a=(3,7), b=(5.5,10))
    assert f == 0.375

def test_tostring():
    box = (1,2,3)
    s = tostring(box)
    assert s == '[1,2,3]'
    box = [1,2,3]
    s = tostring(box)
    assert s == '[1,2,3]'
    box = np.array([1,2,3])
    s = tostring(box)
    assert s == '[1,2,3]'
    box = [[1,2,3],[1,2]]
    s = tostring(box)
    assert s == '[[1,2,3],[1,2]]'

def test_octave_bands():
    fc, fmin, fmax = octave_bands(1, 3)
    assert fc[0] == 62.5
    assert fc[1] == 125.
    assert fc[2] == 250.

def test_morlet_func_single_time():
    time = 0.5
    f = morlet_func(time=time, frequency=10, width=3, displacement=0)
    assert f == pytest.approx(0.42768108, abs=1E-5) 

def test_morlet_func_multiple_times():
    time = np.array([-1., 0., 0.5])
    f = morlet_func(time=time, frequency=10, width=3, displacement=0)
    assert f[0] == pytest.approx(0.41022718, abs=1E-5) 
    assert f[1] == pytest.approx(0.43366254, abs=1E-5) 
    assert f[2] == pytest.approx(0.42768108, abs=1E-5) 

def test_morlet_func_with_dfdt_nonzero():
    time = 0.5
    f = morlet_func(time=time, frequency=10, width=3, displacement=0, dfdt=0.5)
    assert f == pytest.approx(0.302416, abs=1E-5) 

def test_random_floats():
    x = random_floats(3, 0.4, 7.2)
    assert x[0] == pytest.approx(3.23574963, abs=1e-5)
    assert x[1] == pytest.approx(5.29820656, abs=1e-5)
    assert x[2] == pytest.approx(0.40077775, abs=1e-5)

def test_nearest_values():
    x = np.array([1.0, 4.0, 5.1, 6.0, 0.2, 0.3, 10.0])
    y = nearest_values(x=x, i=3, n=3)
    assert np.all(y == [5.1, 6.0, 0.2])
    y = nearest_values(x=x, i=0, n=3)
    assert np.all(y == [1.0, 4.0, 5.1])
    y = nearest_values(x=x, i=0, n=4)
    assert np.all(y == [1.0, 4.0, 5.1, 6.0])
    y = nearest_values(x=x, i=6, n=4)
    assert np.all(y == [10.0, 0.3, 0.2, 6.0])

def test_detect_peaks():
    # create a two time series, where only the first contains a peak
    d = {'series1' : pd.Series([1.0, 2.3, 22.0, 2.2, 1.5]), 'series2': pd.Series([1.0, 2.3, 1.8, 2.2, 1.5])}
    df = pd.DataFrame(d)
    # detect peaks with multiplicity 1 and prominence of at least 2.0
    peaks = detect_peaks(df=df, multiplicity=1, prominence=2.0)
    assert np.all(peaks == [0, 0, 1, 0, 0])
    # try again, but this time require multiplicity 2
    peaks = detect_peaks(df=df, multiplicity=2, prominence=2.0)
    assert np.all(peaks == [0, 0, 0, 0, 0])

def test_str_is_int():
    assert str_is_int('5')
    assert str_is_int('-5')
    assert str_is_int('+5')
    assert not str_is_int('-5', signed=False)
    assert not str_is_int('5.')

def test_signif():
    assert signif(1234,2) == 1200
    assert np.all(signif([1236,-4444],3) == [1240, -4440])

def test_ensure_dir():
    ensure_dir("x") #do nothing
    ensure_dir("./x") #do nothing
    folder = os.path.join(path_to_tmp, 'subfolder')
    file_path = os.path.join(folder, 'file.txt') 
    ensure_dir(file_path)
    os.path.isdir(folder)
    shutil.rmtree(folder)

def test_floor():
    assert 2.0 == floor(2, 0)
    assert 2.0 == floor(2.999, 0)
    assert 2.24 == floor(2.2499999, 2)
    assert 2.24 == floor(2.24, 2)
     
def test_ceil():    
    assert 2.0 == ceil(2, 0)
    assert 3.0 == ceil(2.001, 0)
    assert 2.25 == ceil(2.2499999, 2)
    assert 2.25 == ceil(2.24, 2)

def test_floor_round_up(): 
    assert 2.0 == floor_round_up(2.999, 0)
    assert 2.0 == floor_round_up(2.0, 0)
    assert 3.0 == floor_round_up(2.999, 1)
    assert 3.0 == floor_round_up(2.999, 2)
    assert 3.0 == floor_round_up(2.999, 3)
    assert 3.0 == floor_round_up(2.9998888, 3)
    assert 2.0 == floor_round_up(2.9989999, 3)
    
def test_ceil_round_down():
    assert 3.0 == ceil_round_down(2.001, 0)
    assert 3.0 == ceil_round_down(3.0, 0)
    assert 2.0 == ceil_round_down(2.001, 1)
    assert 2.0 == ceil_round_down(2.001, 2)
    assert 2.0 == ceil_round_down(2.001, 3)
    assert 2.0 == ceil_round_down(2.0009999, 3)
    assert 3.0 == ceil_round_down(2.0010001, 3)