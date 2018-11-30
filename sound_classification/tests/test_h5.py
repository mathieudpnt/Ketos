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
    # table description
    d = h5.spec_description(dimensions=(20,60))
    # create table
    _ = h5.create(h5file=f, path='/group_1/', name='table_1', description=d)
    assert '/group_1' in f
    assert '/group_1/table_1' in f    
    # clean
    f.close()
    os.remove(fpath)

@pytest.mark.test_h5_write_spec
def test_h5_write_spec(sine_audio):
    # create spectrogram    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "id_ex789_107_l_[1]"
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp5_db.h5')
    f = tables.open_file(fpath, 'w')
    # table description
    d = h5.spec_description(dimensions=spec.shape)
    # create table
    tbl = h5.create(h5file=f, path='/group_1/', name='table_1', description=d)
    # write spectrogram to table
    h5.write_spec(table=tbl, spectrogram=spec)
    # write spectrogram to table with optional args
    h5.write_spec(table=tbl, spectrogram=spec, id='123%', labels=(1,2), boxes=((1,2,3,4),(1.5,2.5,3.5,4.5)))

    assert tbl[0]['id'].decode() == 'ex789_107'
    assert tbl[0]['labels'].decode() == '[1]'
    assert tbl[0]['boxes'].decode() == ''

    assert tbl[1]['id'].decode() == '123%'
    assert tbl[1]['labels'].decode() == '[1,2]'
    assert tbl[1]['boxes'].decode() == '[[1,2,3,4],[1.5,2.5,3.5,4.5]]'

    f.close()
    os.remove(fpath)