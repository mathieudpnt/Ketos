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
    assert dt.year == 2084
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


def test_parse_seg_name():
    id,labels = dh.parse_seg_name('id_rb001_89_l_[0].wav')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[0]')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[1,2].wav')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[1,2]')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 


def test_creates_correct_number_of_segments():
    audio_file = path_to_assets + "/2min.wav"
    annotations = pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.34, 105.8],
                                 'end':[6.0,75.98,110.0]})

    try:
        shutil.rmtree(path_to_tmp + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=annotations, save_to=path_to_tmp + "/2s_segs")
    
    n_seg = len(glob(path_to_tmp + "/2s_segs/id_2min*.wav"))
    assert n_seg == 60



    shutil.rmtree(path_to_tmp + "/2s_segs")


def test_start_end_args():
    audio_file = path_to_assets+ "/2min.wav"
    _= pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.34, 105.8],
                                 'end':[6.0,75.98,110.0]})

    try:
        shutil.rmtree(path_to_tmp + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, start_seg=10, end_seg=19, save_to=path_to_tmp + "/2s_segs")
    
    n_seg = len(glob(path_to_tmp + "/2s_segs/id_2min*.wav"))
    assert n_seg == 10



    shutil.rmtree(path_to_tmp + "/2s_segs")

def test_seg_labels_are_correct():
    audio_file = path_to_assets+ "/2min.wav"
    annotations = pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.5, 105.0],
                                 'end':[6.0,73.0,108.0]})

    try:
        shutil.rmtree(path_to_tmp + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=annotations, save_to=path_to_tmp + "/2s_segs")
    
    label_0 = len(glob(path_to_tmp + "/2s_segs/id_2min*l_[[]0].wav"))
    assert label_0 == 53

    label_1 = len(glob(path_to_tmp + "/2s_segs/id_2min*l_[[]1].wav"))
    assert label_1 == 5

    label_2 = len(glob(path_to_tmp + "/2s_segs/id_2min*l_[[]2].wav"))
    assert label_2 == 2

    shutil.rmtree(path_to_tmp + "/2s_segs")


def test_creates_segments_without_annotations():
    audio_file = path_to_assets+ "/2min.wav"
    
    try:
        shutil.rmtree(path_to_tmp + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=None, save_to=path_to_tmp + "/2s_segs")
    
    n_seg = len(glob(path_to_tmp + "/2s_segs/id_2min*l_[[]NULL].wav"))

    assert n_seg == 60
    shutil.rmtree(path_to_tmp + "/2s_segs")


def test_seg_from_time_tag():

    
    audio_file = os.path.join(path_to_assets, "2min.wav")
    
    try:
        shutil.rmtree(os.path.join(path_to_tmp, "from_tags"))
    except FileNotFoundError:
        pass

    dh.create_dir(os.path.join(path_to_tmp, "from_tags"))
    
    dh.seg_from_time_tag(audio_file=audio_file, start=0.5, end=2.5 , name="seg_1.wav", save_to=os.path.join(path_to_tmp, "from_tags") )

    
    rate, sig  = wave.read(os.path.join(path_to_tmp, "from_tags", "seg_1.wav"))
    duration = len(sig)/rate
    assert duration == 2.0
    shutil.rmtree(os.path.join(path_to_tmp, "from_tags"))

def test_segs_from_annotations():
    audio_file_path = os.path.join(path_to_assets,'2min.wav')
    annotations = pd.DataFrame({'filename':[audio_file_path,audio_file_path,audio_file_path],
                                 'label':[1,2,1], 'start':[5.0, 70.5, 105.0],
                                 'end':[6.0,73.0,108.0]})

    try:
        shutil.rmtree(path_to_assets + "/from_annot")
    except FileNotFoundError:
        pass
    dh.segs_from_annotations(annotations,path_to_assets + "/from_annot")
    
    # label_0 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]0].wav"))
    # assert label_0 == 53

    label_1 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]1].wav"))
    assert label_1 == 2

    label_2 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]2].wav"))
    assert label_2 == 1

    shutil.rmtree(path_to_assets + "/from_annot")
    


@pytest.mark.parametrize("start,end,expected_label",[
    (4.0,5.0,'[1]'),
    (4.0,5.5,'[1]'),
    (5.0,6.0,'[1]'),
    (5.1,6.0,'[1]'),
    (100.0,100.5,'[2]'),
    (100.5,101.0,'[2]'),
    (99.0,103.0,'[2]'),
    (90.0,110.0,'[2, 1]'),
     ])

def test_get_correct_labels(start,end,expected_label):
    audio_file="2min"
    annotations = pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 100.5, 105.0],
                                 'end':[6.0,103.0,108.0]})
    
    label = dh.get_labels(file='2min',start=start, end=end,
                             annotations=annotations, not_in_annotations=0)
    print(label)
    assert label == expected_label
    

def test_filter_annotations_by_filename():
     annotations = pd.DataFrame({'filename':['2min_01.wav','2min_01.wav','2min_02.wav','2min_02.wav','2min_02.wav'],
                                 'label':[1,2,1,1,1], 'start':[5.0, 100.5, 105.0, 80.0, 90.0],
                                 'end':[6.0,103.0,108.0, 87.0, 94.0]})

     annot_01 = dh._filter_annotations_by_filename(annotations,'2min_01')
     assert annot_01.equals(pd.DataFrame({'filename':['2min_01.wav','2min_01.wav'],
                                 'label':[1,2], 'start':[5.0, 100.5],
                                 'end':[6.0,103.0]}))
                                 
     annot_02 = dh._filter_annotations_by_filename(annotations,'2min_02')
     assert annot_02.equals(pd.DataFrame({'filename':['2min_02.wav','2min_02.wav','2min_02.wav'],
                                 'label':[1,1,1], 'start':[105.0, 80.0, 90.0],
                                 'end':[108.0, 87.0, 94.0]}, index=[2,3,4]))
 
     annot_03 = dh._filter_annotations_by_filename(annotations,'2min_03')               
     assert annot_03.empty
 

def test_pad_signal():
    sig=np.ones((100))
    rate = 50 
    desired_length = 3.0
    sig_length = len(sig)/rate #sig length in seconds

    padded = dh.pad_signal(signal=sig, rate=rate, length=desired_length)
    

    assert len(padded) == desired_length * rate
    pad_1_limit = int((desired_length - sig_length) * rate / 2)
    pad_2_limit = int(desired_length * rate - pad_1_limit)
    assert sum(padded[:pad_1_limit]) == 0
    assert sum(padded[pad_2_limit:]) == 0
    assert pytest.approx(padded[pad_1_limit:pad_2_limit], sig)

    
