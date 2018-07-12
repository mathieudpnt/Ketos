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
def test_create_segments_from_sine_wave_file(sine_wave_file):
    x = 1