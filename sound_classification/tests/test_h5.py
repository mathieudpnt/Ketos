""" Unit tests for the the 'h5' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Institute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""
import pytest
import tables
import os
import sound_classification.h5 as h5
import sound_classification.data_handling as dh
from sound_classification.spectrogram import MagSpectrogram

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.mark.test_h5_open
def test_h5_open_non_existing_table():
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp3_db.h5')
    f = tables.open_file(fpath, 'w')
    # open non-existing table
    tbl = h5.open(h5file=f, table_path='/group_1/table_1')
    assert tbl == None
    # clean
    f.close()
    os.remove(fpath)

@pytest.mark.test_h5_create
def test_h5_create():
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp4_db.h5')
    f = tables.open_file(fpath, 'w')
    # create table
    _ = h5.create(h5file=f, path='/group_1/', name='table_1', shape=(20,60))
    assert '/group_1' in f
    assert '/group_1/table_1' in f    
    # clean
    f.close()
    os.remove(fpath)

@pytest.mark.test_h5_write
def test_h5_write(sine_audio):
    # create spectrogram    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "id_ex789_107_l_[1]"
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp5_db.h5')
    f = tables.open_file(fpath, 'w')
    # create table
    tbl = h5.create(h5file=f, path='/group_1/', name='table_1', shape=spec.image.shape)
    # write spectrogram to table
    h5.write(table=tbl, x=spec)
    # write spectrogram to table with optional args
    h5.write(table=tbl, x=spec, id='123%', labels=(1,2), boxes=((1,2,3,4),(1.5,2.5,3.5,4.5)))

    assert tbl[0]['id'].decode() == 'ex789_107'
    assert tbl[0]['labels'].decode() == '[1]'
    assert tbl[0]['boxes'].decode() == ''

    assert tbl[1]['id'].decode() == '123%'
    assert tbl[1]['labels'].decode() == '[1,2]'
    assert tbl[1]['boxes'].decode() == '[[1,2,3,4],[1.5,2.5,3.5,4.5]]'

    f.close()
    os.remove(fpath)

@pytest.mark.test_h5_ensure_min_length
def test_h5_ensure_min_length():
    box1 = [1.0, 2.0]
    box2 = [0.5, 4.0]
    min_length = 2.0
    boxes = [box1, box2]
    boxes = h5.ensure_min_length(boxes=boxes, min_length=min_length)
    assert len(boxes) == 2
    assert boxes[0][1]-boxes[0][0] == pytest.approx(min_length, abs=0.0001)
    assert boxes[1][1]-boxes[1][0] > min_length

@pytest.mark.test_h5_select_boxes
def test_h5_select_boxes():
    box1 = [1.0, 2.0]
    box2 = [0.5, 4.0]
    labels = [1, 2]
    label = 1
    boxes = h5.select_boxes(boxes=[box1, box2], labels=labels, label=label)
    assert len(boxes) == 1
    assert boxes[0] == [1.0, 2.0]

@pytest.mark.test_h5_get
def test_h5_get(sine_audio):
    # create spectrogram    
    spec = MagSpectrogram(sine_audio, winlen=0.2, winstep=0.02)
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp6_db.h5')
    f = tables.open_file(fpath, 'w')
    # create table
    tbl = h5.create(h5file=f, path='/group_1/', name='table_1', shape=spec.image.shape)
    # write spectrogram to table
    h5.write(table=tbl, x=spec, id='1', labels=(1), boxes=((1.0, 1.4, 50, 300)))  # Box: 1.0-1.4 s & 50-300 Hz
    # parse labels and boxes
    labels = h5.parse_labels(table=tbl[0])
    boxes = h5.parse_boxes(table=tbl[0])
    assert labels == [1]
    assert boxes[0][0] == 1.0
    assert boxes[0][1] == 1.4
    assert boxes[0][2] == 50
    assert boxes[0][3] == 300    
    # get segments with label=1
    selection, complement = h5.get(table=tbl, label=1, min_length=0.8)
    assert len(selection) == 1
    tshape = int(0.8 / spec.tres)
    assert selection[0].image.shape[0] == tshape
    fshape = int(250 / spec.fres)
    assert selection[0].image.shape[1] == fshape
    assert complement.image.shape[0] == spec.image.shape[0] - tshape
    
