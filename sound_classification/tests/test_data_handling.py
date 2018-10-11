""" Unit tests for the the 'data_handling' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Institute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""

import pytest
import numpy as np
import pandas as pd
import sound_classification.data_handling as dh
import sound_classification.pre_processing as pp
import os
from glob import glob

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.mark.test_encode_database
def test_encode_database_with_one_image_and_one_label(datebase_with_one_image_col_and_one_label_col):
    db = datebase_with_one_image_col_and_one_label_col
    dh.encode_database(db, "image", "label")
    
@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_names_do_not_match(datebase_with_one_image_col_and_one_label_col):
    db = datebase_with_one_image_col_and_one_label_col
    with pytest.raises(AssertionError):
        dh.encode_database(db, "kangaroo", "label")

@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_database_does_not_have_a_label_column(datebase_with_one_image_col_and_no_label_col):
    db = datebase_with_one_image_col_and_no_label_col
    with pytest.raises(AssertionError):
        dh.encode_database(db, "image", "label")

@pytest.mark.test_encode_database
def test_encode_database_can_handle_inputs_with_multiple_columns(datebase_with_two_image_cols_and_one_label_col):
    db = datebase_with_two_image_cols_and_one_label_col
    dh.encode_database(db, "image1", "label")

@pytest.mark.test_split_database
def test_split_database_throws_exception_unless_all_three_keys_are_given(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    encoded, img_size = dh.encode_database(raw, "image", "label") 
    divisions = {"train":(0,100),"validation":(0,100)}
    with pytest.raises(AssertionError):
        split = dh.split_database(encoded, divisions)
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    split = dh.split_database(encoded, divisions)

@pytest.mark.test_stack_dataset
def test_stack_dataset_throws_exception_if_column_names_do_not_match(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    with pytest.raises(AssertionError):
        dh.stack_dataset(raw,(128,128))

@pytest.mark.test_stack_dataset
def test_stack_dataset_automatically_determines_image_size(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    encoded, img_size = dh.encode_database(raw, "image", "label")
    stacked = dh.stack_dataset(encoded, img_size)   

@pytest.mark.test_prepare_database
def test_prepare_database_executes(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    dh.prepare_database(raw, "image", "label", divisions) 

@pytest.mark.test_create_segments
def test_creates_correct_number_of_segments(sine_wave_file):
    prefix="halifax123456789"
    # clean
    for f in glob(path_to_assets + "/*" + prefix + "*"):
        os.remove(f)
    # create segment files
    dh.create_segments(sine_wave_file, 0.5, path_to_assets, prefix)
    # coun them
    files = dh.get_files(path_to_assets, prefix)
    n = len(files)
    # clean
    for f in glob(path_to_assets + "/*" + prefix + "*"):
        os.remove(f)
    assert n == 6

@pytest.mark.test_def_slice_ffmpeg
def test_sliced_audio_file_has_correct_properties(sine_wave_file):
    prefix="halifax123456789"
    out_name = path_to_assets + "/" + prefix + ".wav"
    dh.slice_ffmpeg(sine_wave_file, 0.0, 1.7, out_name)
    rate_orig, sig_orig = pp.wave.read(sine_wave_file)
    rate, sig = pp.wave.read(out_name)
    duration = len(sig) / rate
    assert rate == rate_orig
    assert duration == 1.7
#    for i in range(len(sig)):
#        assert sig[i] == sig_orig[i]
    # clean
    for f in glob(path_to_assets + "/*" + prefix + "*"):
        os.remove(f)

@pytest.mark.parametrize("input,depth,expected",[
    (1,2,np.array([0,1])),
    (0,2,np.array([1,0])),
    (1.0,2,np.array([0,1])),
    (0.0,2,np.array([1,0])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_with_floats_and_ints(input, depth, expected):
    one_hot = dh.to1hot(input, depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (1,4,np.array([0,1,0,0])),
    (1,4,np.array([0,1,0,0])),
    (1,2,np.array([0,1])),
    (1,10,np.array([0,1,0,0,0,0,0,0,0,0])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_output_has_correct_depth(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert len(one_hot) == depth


@pytest.mark.parametrize("input,depth,expected",[
    (3,4,np.array([0,0,0,1])),
    (0,4,np.array([1,0,0,0])),
    (1.0,2,np.array([0,1])),
    (5.0,10,np.array([0,0,0,0,0,1,0,0,0,0])),
    ])
@pytest.mark.test_to1hot
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
@pytest.mark.test_to1hot
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
@pytest.mark.test_to1hot
def test_to1hot_works_when_when_applying_to_DataFrame(input,depth, expected):
     
    one_hot = input["label"].apply(dh.to1hot,depth=depth)
    for i in range(len(one_hot)):
        assert (one_hot[i] == expected[i]).all()

@pytest.mark.test_get_wave_files
def test_get_wave_files():
    # clean
    for f in glob(path_to_assets + "/*.wav"):
        os.remove(f)  #clean
    # create two wave files
    f1 = path_to_assets + "/f1.wav"
    f2 = path_to_assets + "/f2.wav"
    pp.wave.write(f2, rate=100, data=np.array([1.,0.]))
    pp.wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.get_wave_files(path_to_assets, fullpath=False)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    files = dh.get_wave_files(path_to_assets, fullpath=True)
    assert len(files) == 2
    assert files[0] == f1
    assert files[1] == f2

def test_get_wave_files_from_multiple_folders():
    # clean
    for f in glob(path_to_assets + "/*.wav"):
        os.remove(f)  #clean
    # create two wave files in separate subfolders
    sub1 = path_to_assets + "/sub1"
    sub2 = path_to_assets + "/sub2"
    if not os.path.exists(sub1):
        os.mkdir(sub1)
    if not os.path.exists(sub2):
        os.mkdir(sub2)
    f1 = sub1 + "/f1.wav"
    f2 = sub2 + "/f2.wav"
    pp.wave.write(f2, rate=100, data=np.array([1.,0.]))
    pp.wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.get_wave_files(path_to_assets, fullpath=False, subdirs=True)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    files = dh.get_wave_files(path_to_assets, fullpath=True, subdirs=True)
    assert len(files) == 2
    assert files[0] == f1
    assert files[1] == f2

    
################################
# from1hot() tests
################################


@pytest.mark.parametrize("input,expected",[
    (np.array([0,1]),1),
    (np.array([1,0]),0),
    (np.array([0.0,1.0]),1),
    (np.array([1.0,0.0]),0),
    ])
@pytest.mark.test_from1hot
def test_from1hot_works_with_floats_and_ints(input, expected):
    one_hot = dh.from1hot(input)
    assert one_hot == expected


@pytest.mark.parametrize("input,expected",[
    (np.array([0,0,0,1]),3),
    (np.array([1,0,0,0]),0),
    (np.array([0,1]),1),
    (np.array([0,0,0,0,0,1,0,0,0,0]),5),
    ])
@pytest.mark.test_from1hot
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
@pytest.mark.test_from1hot
def test_from1hot_works_with_multiple_input_values_at_once(input, expected):
    one_hot = dh.from1hot(input)
    assert (one_hot == expected).all()

@pytest.mark.test_read_wave
def test_read_wave_file(sine_wave_file):
    rate, data = dh.read_wave(sine_wave_file)
    assert rate == 44100

@pytest.mark.test_parse_datetime
def test_parse_datetime_with_urban_sharks_format():
    fname = 'empty_HMS_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    pp.wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(fname=fname, fmt=fmt)
    os.remove(full_path)
    assert dt is not None
    assert dt.year == 2084
    assert dt.month == 2
    assert dt.day == 23
    assert dt.hour == 12
    assert dt.minute == 5
    assert dt.second == 28

@pytest.mark.test_parse_datetime
def test_parse_datetime_with_non_matching_format():
    fname = 'empty_HMQ_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    pp.wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(fname=fname, fmt=fmt)
    os.remove(full_path)
    assert dt == None
