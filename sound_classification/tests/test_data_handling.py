""" Unit tests for the the 'data_handling' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
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

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.fixture
def image_2x2():
    image = np.array([[1,2],[3,4]], np.float32)
    return image

@pytest.fixture
def datebase_with_one_image_col_and_one_label_col():
    img = image_2x2()
    d = {'image': [img], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df

@pytest.fixture
def datebase_with_one_image_col_and_no_label_col():
    img = image_2x2()
    d = {'image': [img]}
    df = pd.DataFrame(data=d)
    return df

@pytest.fixture
def datebase_with_two_image_cols_and_one_label_col():
    img = image_2x2()
    d = {'image1': [img], 'image2': [img], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df

@pytest.mark.test_encode_database
def test_encode_database_with_one_image_and_one_label():
    db = datebase_with_one_image_col_and_one_label_col()
    dh.encode_database(db, "image", "label")
    
@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_names_do_not_match():
    db = datebase_with_one_image_col_and_one_label_col()
    with pytest.raises(AssertionError):
        dh.encode_database(db, "kangaroo", "label")

@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_database_does_not_have_a_label_column():
    db = datebase_with_one_image_col_and_no_label_col()
    with pytest.raises(AssertionError):
        dh.encode_database(db, "image", "label")

@pytest.mark.test_encode_database
def test_encode_database_can_handle_inputs_with_multiple_columns():
    db = datebase_with_two_image_cols_and_one_label_col()
    dh.encode_database(db, "image1", "label")

@pytest.mark.test_split_database
def test_split_database_throws_exception_unless_all_three_keys_are_given():
    raw = datebase_with_one_image_col_and_one_label_col()
    encoded, img_size = dh.encode_database(raw, "image", "label") 
    divisions = {"train":(0,100),"validation":(0,100)}
    with pytest.raises(AssertionError):
        split = dh.split_database(encoded, divisions)
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    split = dh.split_database(encoded, divisions)

@pytest.mark.test_stack_dataset
def test_stack_dataset_throws_exception_if_column_names_do_not_match():
    raw = datebase_with_one_image_col_and_one_label_col()
    with pytest.raises(AssertionError):
        dh.stack_dataset(raw,(128,128))

@pytest.mark.test_stack_dataset
def test_stack_dataset_automatically_determines_image_size():
    raw = datebase_with_one_image_col_and_one_label_col()
    encoded, img_size = dh.encode_database(raw, "image", "label")
    dh.stack_dataset(encoded, img_size)   

@pytest.mark.test_prepare_database
def test_prepare_database_executes():
    raw = datebase_with_one_image_col_and_one_label_col()
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    dh.prepare_database(raw, "image", "label", divisions) 

@pytest.mark.test_create_segments
def test_creates_correct_number_of_segments(sine_wave_file):
    prefix="halifax123456789"
    n = count_files_that_contain_string(path_to_assets,prefix,delete=True) # clean asset directory
    dh.create_segments(sine_wave_file, 0.5, path_to_assets, prefix) # create segment files
    n = count_files_that_contain_string(path_to_assets,prefix,delete=True) # count number of created files and delete them
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
    count_files_that_contain_string(path_to_assets,prefix,delete=True) # clean up


def count_files_that_contain_string(dir, substr, delete=False):
    """ Counts and then deletes all files in a certain directory which 
        have a certain character sequence in the file name.

            Args:
                dir : str
                    Path to directory.
                substr : str
                    Character sequence that appears in the file name.
                delete : bool
                    Remove the files that match the search criteria.

            Returns:
                count : int
                    Number of files that match search criteria.
    """
    files = os.listdir(dir)
    count = 0
    for file in files:
        if (substr in file) & delete:
            os.remove(dir+"/"+file)
            count += 1
    return count


@pytest.mark.parametrize("input,depth,expected",[
    (1,2,np.array([0,1])),
    (0,2,np.array([1,0])),
    (1.0,2,np.array([1,0])),
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
     np.array([[0., 1., 0.],
               [0., 0., 1.]])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_with_multiple_input_values_at_once(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert (one_hot == expected).all()
