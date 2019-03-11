""" Unit tests for the database_interface module within the ketos library

    
    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
             Project goal: The ketos library provides functionalities for handling data, processing audio signals and
             creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import pytest
import tables
import os
import ketos.data_handling.database_interface as di
import ketos.data_handling.data_handling as dh
from ketos.audio_processing.spectrogram import MagSpectrogram


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.mark.test_open_table
def test_open_non_existing_table():
    """ Test if the expected exception is raised when the table does not exist """
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp3_db.h5')
    h5file = tables.open_file(fpath, 'w')
    # open non-existing table
    with pytest.raises(tables.NoSuchNodeError):
        tbl = di.open_table(h5file=h5file, table_path='/group_1/table_1')
        assert tbl == None
    # clean
    h5file.close()
    os.remove(fpath)

@pytest.mark.test_open_table
def test_open_existing_table():
    """ Test if the expected table is open """
    # open h5 file
    fpath = os.path.join(path_to_assets, '15x_same_spec.h5')
    h5file = tables.open_file(fpath, 'r')
    # open non-existing table
    tbl = di.open_table(h5file=h5file, table_path='/train/species1')
    assert isinstance(tbl, tables.table.Table)
    assert tbl.nrows == 15
    # clean
    h5file.close()
   

@pytest.mark.test_create_table
def test_create_table():
    """Test if a table and its group are created"""
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp4_db.h5')
    h5file = tables.open_file(fpath, 'w')
    # create table
    _ = di.create_table(h5file=h5file, path='/group_1/', name='table_1', shape=(20,60))
    group = h5file.get_node("/group_1")
    assert isinstance(group, tables.group.Group)
    table = h5file.get_node("/group_1/table_1")
    assert isinstance(table, tables.table.Table)    
    # clean
    h5file.close()
    os.remove(fpath)

@pytest.mark.test_create_table
def test_create_table_existing():
    """Test if a table is open when it already exists"""
    # open h5 file
    fpath = os.path.join(path_to_assets, '15x_same_spec.h5')
    h5file = tables.open_file(fpath, 'a')
    # create table
    _ = di.create_table(h5file=h5file, path='/train/', name='species1', shape=(20,60))
    table = h5file.get_node("/train/species1")
    assert table.nrows == 15
    assert table[0]['data'].shape == (2413,201)
    assert table[1]['id'] == b'1'
    # clean
    h5file.close()
    

@pytest.mark.test_write_spec
def test_write_spec(sine_audio):
    """Test if spectrograms are written and have the expected ids"""
    # create spectrogram    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "dummytag"
    # add annotation
    spec.annotate(labels=(1,2), boxes=((1,2,3,4),(1.5,2.5,3.5,4.5)))
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp5_db.h5')
    h5file = tables.open_file(fpath, 'w')
    # create table
    tbl = di.create_table(h5file=h5file, path='/group_1/', name='table_1', shape=spec.image.shape)
    # write spectrogram to table
    spec = di.write_spec(table=tbl, spec=spec)
    
    assert spec is None
    # write spectrogram to table with id
    di.write_spec(table=tbl, spec=spec, id='123%')

    assert tbl[0]['id'].decode() == 'dummytag'
    assert tbl[0]['labels'].decode() == '[1,2]'
    assert tbl[0]['boxes'].decode() == '[[1.0,2.0,3.0,4.0],[1.5,2.5,3.5,4.5]]'

    assert tbl[1]['id'].decode() == '123%'
    assert tbl[1]['labels'].decode() == '[1,2]'
    assert tbl[1]['boxes'].decode() == '[[1.0,2.0,3.0,4.0],[1.5,2.5,3.5,4.5]]'

    h5file.close()
    os.remove(fpath)

@pytest.mark.test_write_spec
def test_write_spec_TypeError(sine_audio):
    """Test if a type error is raised when trying to pass an object that isn't an instance of Spectrogram (or its subclasses)"""
    sine_audio.annotate(labels=(1,2), boxes=((1,2,3,4),(1.5,2.5,3.5,4.5)))
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp5_db.h5')
    h5file = tables.open_file(fpath, 'w')
    # create table
    tbl = di.create_table(h5file=h5file, path='/group_1/', name='table_1', shape=sine_audio.data.shape)
    # write spectrogram to table
    with pytest.raises(TypeError):
        di.write_spec(table=tbl, spec=sine_audio)
       
    assert tbl.nrows == 0
    
    h5file.close()
    os.remove(fpath)


@pytest.mark.test_extract
def test_extract(sine_audio):
    """ Test if annotations are correctly extracted from spectrograms"""
    # create spectrogram    
    spec1 = MagSpectrogram(sine_audio, winlen=0.2, winstep=0.02)
    spec1.annotate(labels=(1), boxes=((1.001, 1.401, 50, 300)))
    spec2 = MagSpectrogram(sine_audio, winlen=0.2, winstep=0.02)
    spec2.annotate(labels=(1), boxes=((1.1, 1.5)))
    tshape_orig = spec1.image.shape[0]
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp6_db.h5')
    h5file = tables.open_file(fpath, 'w')
    # create table
    tbl = di.create_table(h5file=h5file, path='/group_1/', name='table_1', shape=spec1.image.shape)
    # write spectrograms to table
    di.write_spec(table=tbl, spec=spec1, id='1')  # Box: 1.0-1.4 s & 50-300 Hz
    di.write_spec(table=tbl, spec=spec2, id='2')  # Box: 1.1-1.5 s Hz
    # parse labels and boxes
    labels = di.parse_labels(item=tbl[0])
    boxes = di.parse_boxes(item=tbl[0])
    assert labels == [1]
    assert boxes[0][0] == 1.001
    assert boxes[0][1] == 1.401
    assert boxes[0][2] == 50
    assert boxes[0][3] == 300    
    # get segments with label=1
    selection, complements = di.extract(table=tbl, label=1, min_length=0.8, fpad=False, center=True)
    assert len(selection) == 2
    assert len(complements) == 2
    
    assert selection[0].image.shape == (40,50)
    assert complements[0].image.shape[0] == (spec1.image.shape[0] - selection[0].image.shape[0])

    assert selection[1].image.shape == (40,4411)
    assert complements[1].image.shape[0] == (spec2.image.shape[0] - selection[0].image.shape[0])



    tshape = int(0.8 / spec1.tres)
    assert selection[0].image.shape[0] == tshape
    fshape = int(250 / spec1.fres)
    assert selection[0].image.shape[1] == fshape
    assert selection[0].boxes[0][0] == pytest.approx(0.201, abs=0.000001)
    assert selection[0].boxes[0][1] == pytest.approx(0.601, abs=0.000001)

@pytest.mark.test_h5_select_spec
def test_h5_select_spec(sine_audio):
    # create spectrogram    
    spec = MagSpectrogram(sine_audio, winlen=0.2, winstep=0.02)
    spec.annotate(labels=(2), boxes=((1.0, 1.4, 50, 300)))
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp7_db.h5')
    f = tables.open_file(fpath, 'w')
    # create table
    tbl = h5.create(h5file=f, path='/group_1/', name='table_1', shape=spec.image.shape)
    # write spectrogram to table
    h5.write(table=tbl, x=spec, id='A') 
    h5.write(table=tbl, x=spec, id='B') 
    # select spectrograms with label=2
    rows = h5.select(table=tbl, label=2)
    assert len(rows) == 2
    assert rows[0] == 0

