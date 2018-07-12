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
def datebase_with_one_image_and_one_label():
    img = image_2x2()
    d = {'image': [img], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df

@pytest.mark.test_encode_database
def test_encode_database_with_one_image_and_one_label():
    db = datebase_with_one_image_and_one_label()
    dh.encode_database(db, "image", "label")
    
@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_names_do_not_match():
    db = datebase_with_one_image_and_one_label()
    with pytest.raises(AssertionError):
        dh.encode_database(db, "kangaroo", "label")
