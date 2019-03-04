""" Unit tests for the the 'annotation' module in the 'sound_classification' package

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
from ketos.utils import tostring


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
