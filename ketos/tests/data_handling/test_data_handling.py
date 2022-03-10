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

""" Unit tests for the 'data_handling' module within the ketos library
"""

import pytest
import numpy as np
import pandas as pd
import ketos.data_handling.data_handling as dh
import scipy.io.wavfile as wave
import datetime
import shutil
import os
from glob import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

today = datetime.datetime.today()


@pytest.mark.parametrize("input,depth,expected",[
    (1,2,np.array([0,1])),
    (0,2,np.array([1,0])),
    (1.0,2,np.array([0,1])),
    (0.0,2,np.array([1,0])),
    ])
def test_to1hot_works_with_floats_and_ints(input, depth, expected):
    one_hot = dh.to1hot(input, depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (1,4,np.array([0,1,0,0])),
    (1,4,np.array([0,1,0,0])),
    (1,2,np.array([0,1])),
    (1,10,np.array([0,1,0,0,0,0,0,0,0,0])),
    ])
def test_to1hot_output_has_correct_depth(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert len(one_hot) == depth


@pytest.mark.parametrize("input,depth,expected",[
    (3,4,np.array([0,0,0,1])),
    (0,4,np.array([1,0,0,0])),
    (1.0,2,np.array([0,1])),
    (5.0,10,np.array([0,0,0,0,0,1,0,0,0,0])),
    ])
def test_to1hot_works_with_multiple_categories(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (np.array([3,0,1,5]),6,
     np.array([[0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1.]])),
    (np.array([0,1]),3,
     np.array([[1., 0., 0.],
               [0., 1., 0.]])),
    ])
def test_to1hot_works_with_multiple_input_values_at_once(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (pd.DataFrame({"label":[0,0,1,0,1,0]}),2,
     np.array([[1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0]]),)
    ])

def test_to1hot_works_when_when_applying_to_DataFrame(input,depth, expected):
     
    one_hot = input["label"].apply(dh.to1hot,depth=depth)
    for i in range(len(one_hot)):
        assert (one_hot[i] == expected[i]).all()


def test_find_wave_files():
    dir = os.path.join(path_to_assets,'test_find_wave_files')
    #delete directory and files within
    if os.path.exists(dir):
        shutil.rmtree(dir)
    dh.create_dir(dir)
    # create two wave files
    f1 = os.path.join(dir, "f1.wav")
    f2 = os.path.join(dir, "f2.wav")
    wave.write(f2, rate=100, data=np.array([1.,0.]))
    wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.find_wave_files(dir, return_path=False)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    files = dh.find_wave_files(dir, return_path=True)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    #delete directory and files within
    shutil.rmtree(dir)

def test_find_wave_files_from_multiple_folders():
    folder = path_to_assets + "/sub"
    # create two wave files in separate subfolders
    sub1 = folder + "/sub1"
    sub2 = folder + "/sub2"
    if not os.path.exists(sub1):
        os.makedirs(sub1)
    if not os.path.exists(sub2):
        os.makedirs(sub2)
    # clean
    for f in glob(sub1 + "/*.wav"):
        os.remove(f)  #clean
    for f in glob(sub2 + "/*.wav"):
        os.remove(f)  #clean
    f1 = sub1 + "/f1.wav"
    f2 = sub2 + "/f2.wav"
    wave.write(f2, rate=100, data=np.array([1.,0.]))
    wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.find_wave_files(folder, return_path=False, search_subdirs=True)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    files = dh.find_wave_files(folder, return_path=True, search_subdirs=True)
    assert len(files) == 2
    assert files[0] == "sub1/f1.wav"
    assert files[1] == "sub2/f2.wav"

    
################################
# from1hot() tests
################################

@pytest.mark.parametrize("input,expected",[
    (np.array([0,1]),1),
    (np.array([1,0]),0),
    (np.array([0.0,1.0]),1),
    (np.array([1.0,0.0]),0),
    ])

def test_from1hot_works_with_floats_and_ints(input, expected):
    one_hot = dh.from1hot(input)
    assert one_hot == expected


@pytest.mark.parametrize("input,expected",[
    (np.array([0,0,0,1]),3),
    (np.array([1,0,0,0]),0),
    (np.array([0,1]),1),
    (np.array([0,0,0,0,0,1,0,0,0,0]),5),
    ])

def test_from1hot_works_with_multiple_categories(input, expected):
    one_hot = dh.from1hot(input)
    assert one_hot == expected


@pytest.mark.parametrize("input,expected",[
    (np.array([[0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1.]]),np.array([3,0,1,5])),
    (np.array([[1., 0., 0.],
               [0., 1., 0.]]), np.array([0,1])),
    ])

def test_from1hot_works_with_multiple_input_values_at_once(input, expected):
    one_hot = dh.from1hot(input)
    assert (one_hot == expected).all()


def test_read_wave_file(sine_wave_file):
    rate, data = dh.read_wave(sine_wave_file)
    assert rate == 44100


def test_parse_datetime_with_urban_sharks_format():
    fname = 'empty_HMS_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(to_parse=fname, fmt=fmt)
    os.remove(full_path)
    assert dt is not None
    assert dt.year == 1984
    assert dt.month == 2
    assert dt.day == 23
    assert dt.hour == 12
    assert dt.minute == 5
    assert dt.second == 28


def test_parse_datetime_with_non_matching_format():
    fname = 'empty_HMQ_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(to_parse=fname, fmt=fmt)
    os.remove(full_path)
    assert dt == None
