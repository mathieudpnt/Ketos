""" Unit tests for the 'utils' module within the ketos library

    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
             Project goal: The ketos library provides functionalities for handling data, processing audio signals and
             creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import pytest
import numpy as np
from ketos.utils import tostring, morlet_func, octave_bands, random_floats


@pytest.mark.test_tostring
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

@pytest.mark.test_octave_bands
def test_octave_bands():
    fc, fmin, fmax = octave_bands(1, 3)
    assert fc[0] == 62.5
    assert fc[1] == 125.
    assert fc[2] == 250.

@pytest.mark.test_morlet_func
def test_morlet_func_single_time():
    time = 0.5
    f = morlet_func(time=time, frequency=10, width=3, displacement=0)
    assert f == pytest.approx(0.42768108, abs=1E-5) 

@pytest.mark.test_morlet_func
def test_morlet_func_multiple_times():
    time = np.array([-1., 0., 0.5])
    f = morlet_func(time=time, frequency=10, width=3, displacement=0)
    assert f[0] == pytest.approx(0.41022718, abs=1E-5) 
    assert f[1] == pytest.approx(0.43366254, abs=1E-5) 
    assert f[2] == pytest.approx(0.42768108, abs=1E-5) 

@pytest.mark.test_morlet_func
def test_morlet_func_with_dfdt_nonzero():
    time = 0.5
    f = morlet_func(time=time, frequency=10, width=3, displacement=0, dfdt=0.5)
    assert f == pytest.approx(0.302416, abs=1E-5) 

@pytest.mark.test_random_floats
def test_random_floats():
    x = random_floats(3, 0.4, 7.2)
    assert x[0] == pytest.approx(3.23574963, abs=1e-5)
    assert x[1] == pytest.approx(5.29820656, abs=1e-5)
    assert x[2] == pytest.approx(0.40077775, abs=1e-5)