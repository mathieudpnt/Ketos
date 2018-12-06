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
from sound_classification.annotation import AnnotationHandler


def test_annotate():
    a = AnnotationHandler()
    a.annotate(labels=1, boxes=[1,2,3,4])
    a.annotate(labels=[2,3], boxes=[[1,2,3,4],[5,6,7,8]])
    assert len(a.labels) == 3
    a.delete_annotations(id=[0,1])
    assert len(a.labels) == 1
    assert a.labels[0] == 3

def test_cut_annotations():
    labels = [1,2]
    box1 = [14.,17.,0.,29.]
    box2 = [2.1,13.0,1.1,28.5]
    boxes = [box1, box2]
    a = AnnotationHandler(labels=labels, boxes=boxes)
    l, b = a._cut_annotations(t1=2, t2=5, f1=24, f2=27)
    assert len(l) == 1
    assert l[0] == 2
    assert b[0][0] == pytest.approx(0.1, abs=0.001)
    assert b[0][1] == pytest.approx(3.0, abs=0.001)
    assert b[0][2] == pytest.approx(24.0, abs=0.001)
    assert b[0][3] == pytest.approx(27.0, abs=0.001)