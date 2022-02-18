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

""" Unit tests for the 'data_handling.database_interface' module within the ketos library
"""
import sys
import pytest
import tables
import os
import numpy as np
import pandas as pd
from io import StringIO
import ketos.data_handling.database_interface as di
import ketos.data_handling.data_handling as dh
from ketos.data_handling.selection_table import use_multi_indexing 
from ketos.audio.spectrogram import MagSpectrogram, Spectrogram, CQTSpectrogram
from ketos.audio.waveform import Waveform
from ketos.audio.gammatone import GammatoneFilterBank

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_open_file_create_dir():
    """ Test if that a directory is created if it does not already exist """
    folder = os.path.join(path_to_tmp, 'new_folder')
    if os.path.isdir(folder): os.rmdir(folder) #remove folder if it already exists
    file_path = os.path.join(folder, 'db.h5')
    for mode in ['w','a']:
        assert not os.path.isdir(folder) #check that folder does not exist before attempt to open
        h5file = di.open_file(file_path, mode='w')
        h5file.close()
        assert os.path.isfile(file_path) #check that file has been created
        os.remove(file_path) #clean
        os.rmdir(folder) #clean


def test_open_non_existing_table():
    """ Test if the expected exception is raised when the table does not exist """
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp1_db.h5')
    h5file = di.open_file(fpath, 'w')
    # open non-existing table
    with pytest.raises(tables.NoSuchNodeError):
        tbl = di.open_table(h5file=h5file, table_path='/group_1/table_1')
        assert tbl == None
    # clean
    h5file.close()
    os.remove(fpath)

def test_open_existing_table():
    """ Test if the expected table is open """
    # open h5 file
    fpath = os.path.join(path_to_assets, '15x_same_spec.h5')
    h5file = di.open_file(fpath, 'r')
    # open non-existing table
    tbl = di.open_table(h5file=h5file, table_path='/train/species1')
    assert isinstance(tbl, tables.table.Table)
    assert tbl.nrows == 15
    # clean
    h5file.close()

def test_table_descr_weak():
    """ Test that we can create a table description for weakly annotated data"""
    img = np.random.random_sample((64,20))#Create a 64 x 20 image
    descr = di.table_description(img)#Create a table description for weakly labeled spectrograms of this shape
    cols = descr.columns
    keys = sorted(cols.keys())
    assert keys[0] == 'data'
    assert cols[keys[0]] == tables.Float32Col(shape=(64, 20), dflt=0.0, pos=None)
    assert keys[1] == 'filename'
    assert cols[keys[1]] == tables.StringCol(itemsize=100, shape=(), dflt=b'', pos=None)
    assert keys[2] == 'id'
    assert cols[keys[2]] == tables.UInt32Col(shape=(), dflt=0, pos=None)
    assert keys[3] == 'label'
    assert cols[keys[3]] == tables.UInt8Col(shape=(), dflt=0, pos=None)
    assert keys[4] == 'offset'
    assert cols[keys[4]] == tables.Float64Col(shape=(), dflt=0.0, pos=None)  

def test_table_descr_mult_data():
    """ Test that we can create a table description with multiple data fields"""
    img1 = np.random.random_sample((64,20))#Create a 64 x 20 image
    img2 = np.random.random_sample((32,32))#Create a 32 x 32 image
    descr = di.table_description([img1,img2])#Create a table description
    cols = descr.columns
    keys = sorted(cols.keys())
    assert keys[0] == 'data0'
    assert cols[keys[0]] == tables.Float32Col(shape=(64, 20), dflt=0.0, pos=None)
    assert keys[1] == 'data1'
    assert cols[keys[1]] == tables.Float32Col(shape=(32, 32), dflt=0.0, pos=None)

def test_table_descr_mult_data_named():
    """ Test that we can create a table description with multiple data fields
        and user specified names"""
    img1 = np.random.random_sample((64,20))#Create a 64 x 20 image
    img2 = (32,32)#32 x 32 shape
    descr, names = di.table_description([img1,img2], data_name=['a_large_img', 'a_small_img'], return_data_name=True)#Create a table description
    cols = descr.columns
    keys = sorted(cols.keys())
    assert keys[0] == 'a_large_img'
    assert cols[keys[0]] == tables.Float32Col(shape=(64, 20), dflt=0.0, pos=None)
    assert keys[1] == 'a_small_img'
    assert cols[keys[1]] == tables.Float32Col(shape=(32, 32), dflt=0.0, pos=None)
    assert names == ['a_large_img', 'a_small_img']

def test_table_descr_attrs():
    """ Test that we can create a table description with additional columns"""
    img = np.random.random_sample((64,20))
    attrs = [{'name':'a_confidence', 'shape':(), 'type':'float'},
             {'name':'a_comment', 'shape':120, 'type':'str'},
             {'name':'a_vector', 'shape':(5), 'type':'int'}]
    descr = di.table_description(img, attrs=attrs)
    cols = descr.columns
    keys = sorted(cols.keys())
    print(keys)
    assert keys[0] == 'a_comment'
    assert cols[keys[0]] == tables.StringCol(itemsize=120)
    assert keys[1] == 'a_confidence'
    assert cols[keys[1]] == tables.Float64Col(shape=(), dflt=0.0, pos=None)
    assert keys[2] == 'a_vector'
    assert cols[keys[2]] == tables.Int32Col(shape=(5,), dflt=0.0, pos=None)

def test_create_table():
    """Test if a table and its group are created"""
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp2_db.h5')
    h5file = di.open_file(fpath, 'w')
    # create table description
    descr_data = di.table_description((32,64))
    descr_annot = di.table_description_annot()
    # create data table
    _ = di.create_table(h5file=h5file, path='/group_1/', name='table_1', description=descr_data)
    group = h5file.get_node("/group_1")
    assert isinstance(group, tables.group.Group)
    table = h5file.get_node("/group_1/table_1")
    assert isinstance(table, tables.table.Table)    
    # create annotation table
    _ = di.create_table(h5file=h5file, path='/group_1/', name='table_2', description=descr_annot)
    group = h5file.get_node("/group_1")
    assert isinstance(group, tables.group.Group)
    table = h5file.get_node("/group_1/table_2")
    assert isinstance(table, tables.table.Table)    
    # clean
    h5file.close()
    os.remove(fpath)

def test_add_row_to_annot_table():
    """Test if we can add a row to the annotation table"""
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp2_db.h5')
    h5file = di.open_file(fpath, 'w')
    # create table description
    descr_annot = di.table_description_annot()
    # create annotation table
    table = di.create_table(h5file=h5file, path='/group_1/', name='table_2', description=descr_annot)
    # add a row
    row = table.row
    row['label'] = 12
    row['data_index'] = 3
    row['start'] = 0.
    row['end'] = 1.
    row.append()
    table.flush()
    assert table.nrows == 1
    # clean
    h5file.close()
    os.remove(fpath)

def test_create_table_existing():
    """Test if a table is open when it already exists"""
    # open h5 file
    fpath = os.path.join(path_to_assets, '15x_same_spec.h5')
    h5file = di.open_file(fpath, 'a')
    # create table description
    descr_annot = di.table_description_annot()
    # create table
    _ = di.create_table(h5file=h5file, path='/train/', name='species1', description=descr_annot)
    table = h5file.get_node("/train/species1")
    assert table.nrows == 15
    assert table[0]['data'].shape == (12,12)
    assert table[1]['id'] == 1
    # clean
    h5file.close()

def test_write_mag_spec(sine_audio):
    """Test if spectrograms are written and have the expected ids"""
    # create spectrogram    
    range_trans = {'name':'adjust_range', 'range':(0,1)}
    transforms = [range_trans]
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1, transforms=transforms)
    spec.filename = 'file.wav'
    spec.offset = 0.1
    # add annotation
    spec.annotate(df={'label':[1,2], 'start':[1,1.5], 'end':[2,2.5], 'freq_min':[3,3.5], 'freq_max':[4,4.5]})
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp3_db.h5')
    h5file = di.open_file(fpath, 'w')
    # Create table descriptions for storing the spectrogram data
    descr_data = di.table_description(spec)
    descr_annot = di.table_description_annot()
    # Create tables
    tbl_data = di.create_table(h5file, "/group1/", "table_data", descr_data) 
    tbl_annot = di.create_table(h5file, "/group1/", "table_annot", descr_annot) 
    # write spectrogram to table twice
    di.write(x=spec, table=tbl_data, table_annot=tbl_annot) 
    di.write(x=spec, table=tbl_data, table_annot=tbl_annot) 
    # write spectrogram to table with id
    di.write(x=spec, table=tbl_data, table_annot=tbl_annot, id=7)
    tbl_data.flush()
    tbl_annot.flush()
    # check that annotations have been properly saved
    x = tbl_annot[0]
    assert x['data_index'] == 0
    assert x['label'] == 1
    assert x['start'] == 1.
    assert x['end'] == 2.
    x = tbl_annot[1]
    assert x['data_index'] == 0
    assert x['label'] == 2
    assert x['start'] == 1.5
    assert x['end'] == 2.5
    x = tbl_annot[2]
    assert x['data_index'] == 1
    x = tbl_annot[3]
    assert x['data_index'] == 1
    x = tbl_annot[4]
    assert x['data_index'] == 2
    assert x['label'] == 1
    assert x['start'] == 1.
    assert x['end'] == 2.
    x = tbl_data[0]
    assert x['filename'].decode() == 'file.wav'
    assert x['offset'] == 0.1
    assert tbl_data[0]['id'] == 0
    assert tbl_data[1]['id'] == 1
    assert tbl_data[2]['id'] == 7
    # check that attributes have been properly saved
    assert tbl_data.attrs.audio_repres['time_res'] == 0.1
    assert tbl_data.attrs.audio_repres['freq_min'] == 0
    assert tbl_data.attrs.audio_repres['freq_res'] == 0.5 * sine_audio.rate / spec.data.shape[1]
    assert tbl_data.attrs.audio_repres['type'] == 'MagSpectrogram'
    assert tbl_data.attrs.audio_repres['window_func'] == 'hamming'
    assert tbl_data.attrs.audio_repres['transform_log'] == transforms   
    # clean up
    h5file.close()
    os.remove(fpath)

def test_write_cqt_spec(sine_audio):
    """Test if CQT spectrograms are written with appropriate encoding"""
    # create cqt spectrogram    
    spec = CQTSpectrogram.from_waveform(audio=sine_audio, freq_min=1, freq_max=8000, step=0.1, bins_per_oct=32)
    # add annotation
    df = {'label':[1,2], 'start':[1,1.5], 'end':[2,2.5], 'freq_min':[300,300.5], 'freq_max':[400,400.5]}
    spec.annotate(df=df)
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp12_db.h5')
    h5file = di.open_file(fpath, 'w')
    # Create table descriptions for storing the spectrogram data
    descr_data = di.table_description(spec)
    descr_annot = di.table_description_annot(freq_range=True)
    # create tables
    tbl_data  = di.create_table(h5file=h5file, path='/group_1/', name='table_data', description=descr_data)
    tbl_annot = di.create_table(h5file=h5file, path='/group_1/', name='table_annot', description=descr_annot)
    # write spectrogram to table
    di.write(x=spec, table=tbl_data, table_annot=tbl_annot) 
    tbl_data.flush()
    tbl_annot.flush()
    h5file.close()
    # re-open
    h5file = di.open_file(fpath, 'r')
    tbl_d = h5file.get_node("/group_1/table_data")
    tbl_a = h5file.get_node("/group_1/table_annot")
    # check that annotations have ben properly saved
    assert tbl_d.nrows == 1
    assert tbl_a.nrows == 2
    for i in range(2):
        assert tbl_a[i]['label']    == df['label'][i]
        assert tbl_a[i]['start']    == df['start'][i]
        assert tbl_a[i]['end']      == df['end'][i]
        assert tbl_a[i]['freq_min'] == df['freq_min'][i]
        assert tbl_a[i]['freq_max'] == df['freq_max'][i]
    # check that attributes have been properly saved
    assert tbl_d.attrs.audio_repres['time_res'] > 0.
    assert tbl_d.attrs.audio_repres['freq_min'] == 1
    assert tbl_d.attrs.audio_repres['bins_per_oct'] == 32
    assert tbl_d.attrs.audio_repres['type'] == 'CQTSpectrogram'
    assert tbl_d.attrs.audio_repres['window_func'] == 'hann'
    # clean
    h5file.close()
    os.remove(fpath)

def test_write_multiple_audio_objects(sine_audio):
    """Test if multiple audio objects can be written to an HDF5 table"""
    # create spectrogram    
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    spec.filename = 'file.wav'
    spec.offset = 0.1
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp33_db.h5')
    h5file = di.open_file(fpath, 'w')
    # Create table descriptions for storing the spectrogram and waveform data
    descr, data_name = di.table_description([spec, sine_audio], data_name=['spec','waveform'], return_data_name=True)
    # Create tables
    tbl = di.create_table(h5file, "/group1/", "table_data", description=descr, data_name=data_name) 
    # write spectrogram and waveforms to table twice
    di.write(x=[spec, sine_audio], table=tbl) 
    di.write(x=[spec, sine_audio], table=tbl) 
    # write spectrogram and waveform to table with id
    di.write(x=[spec, sine_audio], table=tbl, id=7)
    tbl.flush()
    # check spectrogram 
    x = tbl[0]
    assert x['filename'].decode() == 'file.wav'
    assert x['offset'] == 0.1
    assert tbl[0]['id'] == 0
    assert tbl[1]['id'] == 1
    assert tbl[2]['id'] == 7
    # check that spectrogram representation has been properly saved
    assert tbl.attrs.audio_repres[0]['time_res'] == 0.1
    assert tbl.attrs.audio_repres[0]['freq_min'] == 0
    assert tbl.attrs.audio_repres[0]['freq_res'] == 0.5 * sine_audio.rate / spec.data.shape[1]
    assert tbl.attrs.audio_repres[0]['type'] == 'MagSpectrogram'
    assert tbl.attrs.audio_repres[0]['window_func'] == 'hamming'
    # check that waveform representation has been properly saved
    assert tbl.attrs.audio_repres[1]['rate'] == sine_audio.rate
    assert tbl.attrs.audio_repres[1]['type'] == 'Waveform'
    # test load audio
    selected_obj = di.load_audio(table=tbl)
    assert len(selected_obj) == 3
    assert len(selected_obj[0]) == 2
    assert isinstance(selected_obj[0][0], MagSpectrogram)
    assert isinstance(selected_obj[0][1], Waveform)
    # clean up
    h5file.close()
    os.remove(fpath)

def test_write_spectrogram_and_numpy_array(sine_audio):
    """Test if multiple audio objects can be written to an HDF5 table where one 
        of the objects is a numpy array"""
    # create spectrogram    
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    spec.filename = 'file.wav'
    spec.offset = 0.1
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp33_db.h5')
    h5file = di.open_file(fpath, 'w')
    # Create table descriptions for storing the spectrogram and waveform data
    descr, data_name = di.table_description([spec, (6,12)], data_name=['spec','features'], return_data_name=True)
    # Create tables
    tbl = di.create_table(h5file, "/group1/", "table_data", description=descr, data_name=data_name) 
    # write spectrogram and numpy arrays to table twice
    di.write(x=[spec, np.ones(shape=(6,12))], table=tbl) 
    di.write(x=[spec, np.ones(shape=(6,12))], table=tbl) 
    tbl.flush()
    # check spectrogram 
    x = tbl[0]
    assert x['filename'].decode() == 'file.wav'
    assert x['offset'] == 0.1
    assert tbl[0]['id'] == 0
    assert tbl[1]['id'] == 1
    # check that spectrogram representation has been properly saved
    assert tbl.attrs.audio_repres[0]['time_res'] == 0.1
    assert tbl.attrs.audio_repres[0]['freq_min'] == 0
    assert tbl.attrs.audio_repres[0]['freq_res'] == 0.5 * sine_audio.rate / spec.data.shape[1]
    assert tbl.attrs.audio_repres[0]['type'] == 'MagSpectrogram'
    assert tbl.attrs.audio_repres[0]['window_func'] == 'hamming'
    # check that numpy array has been properly saved
    assert tbl.attrs.audio_repres[1]['type'] == 'numpy.ndarray'
    # test load audio
    selected_obj = di.load_audio(table=tbl)
    assert len(selected_obj) == 2
    assert len(selected_obj[0]) == 2
    assert isinstance(selected_obj[0][0], MagSpectrogram)
    assert isinstance(selected_obj[0][1], np.ndarray)
    # clean up
    h5file.close()
    os.remove(fpath)

def test_write_spectrogram_with_attrs(sine_audio):
    """Test if an audio objects with additional attributes can be written to 
        an HDF5 table"""
    # create spectrogram    
    kwargs = {'confidence':0.3, 'comment':'funny', 'vector':np.array([1,2,3], dtype=np.int32), 'extra_attr':0.9}
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1, **kwargs)
    spec.filename = 'file.wav'
    spec.offset = 0.1
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp34_db.h5')
    h5file = di.open_file(fpath, 'w')
    # additiona attributes that we want to save
    attrs = [{'name':'confidence', 'shape':(), 'type':'float'},
             {'name':'comment', 'shape':120, 'type':'str'},
             {'name':'vector', 'shape':(3), 'type':'int'}]
    # Create table descriptions for storing the spectrogram and waveform data
    descr, data_name = di.table_description([spec, (6,12)], data_name=['spec','features'], return_data_name=True, attrs=attrs)
    # Create tables
    tbl = di.create_table(h5file, "/group1/", "table_data", description=descr, data_name=data_name) 
    # write spectrogram and numpy arrays to table twice
    di.write(x=[spec, np.ones(shape=(6,12))], table=tbl) 
    di.write(x=[spec, np.ones(shape=(6,12))], table=tbl) 
    tbl.flush()
    # check spectrogram 
    x = tbl[0]
    assert x['filename'].decode() == 'file.wav'
    assert x['offset'] == 0.1
    assert x['confidence'] == 0.3
    assert x['comment'].decode() == 'funny'
    assert np.all(x['vector'] == np.array([1,2,3], dtype=int))
    assert tbl[0]['id'] == 0
    assert tbl[1]['id'] == 1
    # check that spectrogram representation has been properly saved
    assert tbl.attrs.audio_repres[0]['time_res'] == 0.1
    assert tbl.attrs.audio_repres[0]['freq_min'] == 0
    assert tbl.attrs.audio_repres[0]['freq_res'] == 0.5 * sine_audio.rate / spec.data.shape[1]
    assert tbl.attrs.audio_repres[0]['type'] == 'MagSpectrogram'
    assert tbl.attrs.audio_repres[0]['window_func'] == 'hamming'
    # check that numpy array has been properly saved
    assert tbl.attrs.audio_repres[1]['type'] == 'numpy.ndarray'
    # test load audio
    selected_obj = di.load_audio(table=tbl)
    assert len(selected_obj) == 2
    assert len(selected_obj[0]) == 2
    assert isinstance(selected_obj[0][0], MagSpectrogram)
    assert isinstance(selected_obj[0][1], np.ndarray)
    spec_attrs = selected_obj[0][0].get_instance_attrs()
    expected = {'filename':'file.wav', 'offset':0.1, 'label':0}
    expected.update(**kwargs)
    expected.pop('extra_attr', None)
    assert np.all(spec_attrs.pop('vector') == expected.pop('vector'))
    assert spec_attrs == expected
    # clean up
    h5file.close()
    os.remove(fpath)

def test_filter_by_label(sine_audio):
    """ Test if filter_by_label works when providing an int or list of ints as the label argument"""
    # create spectrogram  
    spec1 = MagSpectrogram.from_waveform(sine_audio, window=0.2, step=0.02)
    spec1.annotate(label=1, start=1.0, end=1.4, freq_min=50, freq_max=300)
    spec2 = MagSpectrogram.from_waveform(sine_audio, window=0.2, step=0.02)
    spec2.annotate(label=2, start=1.0, end=1.4, freq_min=50, freq_max=300)
    spec3 = MagSpectrogram.from_waveform(sine_audio, window=0.2, step=0.02)
    spec3.annotate(df={'label':[2,3], 'start':[1.0,2.0], 'end':[1.4,2.4], 'freq_min':[50,60], 'freq_max':[300,200]})
    # open h5 file
    fpath = os.path.join(path_to_tmp, 'tmp8_db.h5')
    h5file = di.open_file(fpath, 'w')
    # Create table descriptions for storing the spectrogram data
    descr_data = di.table_description(spec1)
    descr_annot = di.table_description_annot(freq_range=True)
    # create tables
    tbl_data  = di.create_table(h5file=h5file, path='/group_1/', name='table_data', description=descr_data)
    tbl_annot = di.create_table(h5file=h5file, path='/group_1/', name='table_annot', description=descr_annot)
    # write spectrogram to table
    di.write(x=spec1, table=tbl_data, table_annot=tbl_annot) 
    di.write(x=spec1, table=tbl_data, table_annot=tbl_annot) 
    di.write(x=spec2, table=tbl_data, table_annot=tbl_annot) 
    di.write(x=spec2, table=tbl_data, table_annot=tbl_annot)
    di.write(x=spec3, table=tbl_data, table_annot=tbl_annot) 
    tbl_data.flush()
    tbl_annot.flush()
    # select spectrograms containing the label 1
    rows = di.filter_by_label(table=tbl_annot, label=1)
    assert len(rows) == 2
    assert rows == [0,1]
    # select spectrograms containing the label 2
    rows = di.filter_by_label(table=tbl_annot, label=[2])
    assert len(rows) == 3
    assert rows == [2,3,4]
    # select spectrograms containing the labels 1 or 3
    rows = di.filter_by_label(table=tbl_annot, label=[1,3])
    assert len(rows) == 3
    assert rows == [0,1,4]
    h5file.close()
    os.remove(fpath)

def test_filter_by_label_raises_exception(sine_audio):
    """ Test if filter_by_label raises expected exception when the the label argument is of the wrong type"""
    # open h5 file
    fpath = os.path.join(path_to_assets, '15x_same_spec.h5')
    h5file = di.open_file(fpath, 'r')
    tbl = di.open_table(h5file,"/train/species1")
    
    with pytest.raises(TypeError):
        di.filter_by_label(table=tbl, label='a')
    with pytest.raises(TypeError):
        di.filter_by_label(table=tbl, label=['a','b'])
    with pytest.raises(TypeError):        
        di.filter_by_label(table=tbl, label='1')
    with pytest.raises(TypeError):    
        di.filter_by_label(table=tbl, label='1,2')
    with pytest.raises(TypeError):    
        di.filter_by_label(table=tbl, label=b'1')
    with pytest.raises(TypeError):    
        di.filter_by_label(table=tbl, label=1.0)
    with pytest.raises(TypeError):    
        di.filter_by_label(table=tbl, label= (1,2))
    with pytest.raises(TypeError):    
        di.filter_by_label(table=tbl, label=[1.0,2])
   
    h5file.close()

def test_load_audio_no_index_list():
    """Test if load specs loads the entire table if index_list is None""" 
    fpath = os.path.join(path_to_assets, '11x_same_spec.h5')
    h5file = di.open_file(fpath, 'r')
    tbl_data = di.open_table(h5file,"/group_1/table_data")
    tbl_annot = di.open_table(h5file,"/group_1/table_annot")    
    selected_specs = di.load_audio(table=tbl_data, table_annot=tbl_annot)
    assert len(selected_specs) == tbl_data.nrows
    is_spec = [isinstance(item, Spectrogram) for item in selected_specs]
    assert all(is_spec)    
    h5file.close()

def test_load_audio_with_index_list():
    """Test if load_audio loads the spectrograms specified by index_list""" 
    fpath = os.path.join(path_to_assets, '11x_same_spec.h5')
    h5file = di.open_file(fpath, 'r')
    tbl_data = di.open_table(h5file,"/group_1/table_data")
    tbl_annot = di.open_table(h5file,"/group_1/table_annot")    
    selected_specs = di.load_audio(table=tbl_data, table_annot=tbl_annot, indices=[0,3,10])
    assert len(selected_specs) == 3
    is_spec = [isinstance(item, Spectrogram) for item in selected_specs]
    assert all(is_spec)
    h5file.close()

def test_load_audio_also_loads_annotations():
    """Test if the spectrograms returned by load_audio have annotations""" 
    fpath = os.path.join(path_to_assets, '11x_same_spec.h5')
    h5file = di.open_file(fpath, 'r')
    tbl_data = di.open_table(h5file,"/group_1/table_data")
    tbl_annot = di.open_table(h5file,"/group_1/table_annot")    
    specs = di.load_audio(table=tbl_data, table_annot=tbl_annot, indices=[0,3,10])
    # check annotations for 1st spec
    d = '''label  start  end  freq_min  freq_max
0      2    1.0  1.4      50.0     300.0
1      3    2.0  2.4      60.0     200.0'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0])
    res = specs[0].get_annotations()[ans.columns.values].astype({'freq_min': 'float64', 'freq_max': 'float64'})
    pd.testing.assert_frame_equal(ans, res)
    assert specs[0].filename == 'sine_wave'
    h5file.close()

def test_init_audio_writer():
    out = os.path.join(path_to_assets, 'tmp/db4.h5')
    di.AudioWriter(output_file=out)

def test_audio_writer_can_write_one_spec(sine_wave_file):
    range_trans = {'name':'adjust_range', 'range':(0,1)}
    transforms = [range_trans]
    noise_trans = {'name':'add_gaussian_noise', 'sigma':2.0}
    wf_transforms = [noise_trans]
    sine_audio = Waveform.from_wav(sine_wave_file, transforms=wf_transforms)
    out = os.path.join(path_to_assets, 'tmp/db5.h5')
    writer = di.AudioWriter(output_file=out)
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1, transforms=transforms)
    writer.write(spec)
    writer.close()
    fname = os.path.join(path_to_assets, 'tmp/db5.h5')
    fil = di.open_file(fname, 'r')
    assert '/audio' in fil
    specs = di.load_audio(fil.root.audio)
    assert len(specs) == 1
    assert specs[0].transform_log == transforms
    assert specs[0].waveform_transform_log == wf_transforms
    fil.close()

def test_audio_writer_can_write_one_gammatone_filter_bank(sine_audio):
    out = os.path.join(path_to_assets, 'tmp/db14.h5')
    writer = di.AudioWriter(output_file=out)
    gfb = GammatoneFilterBank.from_waveform(sine_audio, num_chan=20, freq_min=10)
    writer.write(gfb)
    writer.close()
    fname = os.path.join(path_to_assets, 'tmp/db14.h5')
    fil = di.open_file(fname, 'r')
    assert '/audio' in fil
    gfbs = di.load_audio(fil.root.audio)
    assert len(gfbs) == 1
    assert gfbs[0].weight_func == True
    assert len(gfbs[0].freqs) == 20
    assert np.min(gfbs[0].freqs) == pytest.approx(10., abs=1e-6)
    assert gfbs[0].get_data().shape[1] == 20
    fil.close()

def test_audio_writer_can_write_two_specs_to_same_node(sine_audio):
    out = os.path.join(path_to_assets, 'tmp/db6.h5')
    writer = di.AudioWriter(output_file=out)
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    writer.write(spec)
    writer.write(spec)
    writer.close()
    fname = os.path.join(path_to_assets, 'tmp/db6.h5')
    fil = di.open_file(fname, 'r')
    assert '/audio' in fil
    specs = di.load_audio(fil.root.audio)
    assert len(specs) == 2
    fil.close()

def test_audio_writer_can_write_several_specs_to_different_nodes(sine_audio):
    out = os.path.join(path_to_assets, 'tmp/db7.h5')
    writer = di.AudioWriter(output_file=out)
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    writer.write(spec, path='/first', name='test')
    writer.write(spec, path='/first', name='test')
    writer.write(spec, path='/second', name='temp')
    writer.write(spec, path='/second', name='temp')
    writer.write(spec, path='/second', name='temp')
    writer.close()
    fname = os.path.join(path_to_assets, 'tmp/db7.h5')
    fil = di.open_file(fname, 'r')
    assert '/first/test' in fil
    assert '/second/temp' in fil
    specs = di.load_audio(fil.root.first.test)
    assert len(specs) == 2
    specs = di.load_audio(fil.root.second.temp)
    assert len(specs) == 3
    fil.close()

def test_audio_writer_splits_into_several_files_when_max_size_is_reached(sine_audio):
    out = os.path.join(path_to_assets, 'tmp/db8.h5')
    writer = di.AudioWriter(output_file=out, max_size=2E6) # max size: 1 Mbyte
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    writer.write(spec)
    writer.write(spec)
    writer.write(spec)
    writer.close()

    fname = os.path.join(path_to_assets, 'tmp/db8_000.h5')
    fil = di.open_file(fname, 'r')
    assert '/audio' in fil
    specs = di.load_audio(fil.root.audio)
    assert len(specs) == 2
    fil.close()

    fname = os.path.join(path_to_assets, 'tmp/db8_001.h5')
    fil = di.open_file(fname, 'r')
    assert '/audio' in fil
    specs = di.load_audio(fil.root.audio)
    assert len(specs) == 1
    fil.close()

def test_audio_writer_change_directory(sine_audio):
    out = os.path.join(path_to_assets, 'tmp/db9.h5')
    writer = di.AudioWriter(output_file=out)
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    writer.set_table('/home/','fish')
    writer.write(spec)
    writer.write(spec)
    writer.write(spec)
    writer.set_table('/home','whale')
    writer.write(spec)
    writer.write(spec)
    writer.close()
    fname = os.path.join(path_to_assets, 'tmp/db9.h5')
    fil = di.open_file(fname, 'r')
    assert '/home/fish' in fil
    assert '/home/whale' in fil
    specs = di.load_audio(fil.root.home.fish)
    assert len(specs) == 3
    specs = di.load_audio(fil.root.home.whale)
    assert len(specs) == 2
    fil.close()

def test_two_audio_writers_simultaneously(sine_audio):
    # init two spec writers
    out1 = os.path.join(path_to_assets, 'tmp/db10.h5')
    writer1 = di.AudioWriter(output_file=out1)
    out2 = os.path.join(path_to_assets, 'tmp/db11.h5')
    writer2 = di.AudioWriter(output_file=out2)
    # create spec
    spec = MagSpectrogram.from_waveform(sine_audio, 0.5, 0.1)
    # write 
    writer1.set_table('/home/','fish')
    writer1.write(spec)
    writer1.write(spec)
    writer1.write(spec)
    writer2.set_table('/home/','whale')
    writer2.write(spec)
    writer2.write(spec)
    # close
    writer1.close()
    writer2.close()
    # check file 1
    fil1 = di.open_file(out1, 'r')
    assert '/home/fish' in fil1
    specs = di.load_audio(fil1.root.home.fish)
    assert len(specs) == 3
    fil1.close()
    # check file 2
    fil2 = di.open_file(out2, 'r')
    assert '/home/whale' in fil2
    specs = di.load_audio(fil2.root.home.whale)
    assert len(specs) == 2
    fil2.close()

def test_create_database_with_single_wav_file(sine_wave_file):
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db12.h5')
    rep = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav','sine_wave.wav'], 'start':[0.1,0.2], 'end':[2.0,2.1], 'label':[1,2]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=False, progress_bar=False)
    # check database contents
    fil = di.open_file(out, 'r')
    assert '/assets/data' in fil
    specs = di.load_audio(table=fil.root.assets.data)
    assert len(specs) == 2
    fil.close()
    os.remove(out)

def test_create_database_ids(sine_wave_file):
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db13.h5')
    rep = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav','sine_wave.wav', 'sine_wave.wav','sine_wave.wav'], 'start':[0.1, 0.2, 0.1, 0.2], 'end':[2.0, 2.1, 2.0, 2.1], 'label':[1, 2, 1, 2]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, dataset_name='test', selections=sel, audio_repres=rep, verbose=False, progress_bar=False)
    # check database contents
    db = di.open_file(out, 'r')
    data_table = db.get_node("/test/data")
    np.testing.assert_array_equal(data_table[:]['id'],[0,1,2,3])
    db.close()
    os.remove(out)

def test_create_database_with_single_wav_file_mult_repres(sine_wave_file):
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db12.h5')
    rep1 = {'type': 'Waveform'}
    rep2 = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav','sine_wave.wav'], 'start':[0.1,0.2], 'end':[2.0,2.1], 'label':[1,2]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=[rep1,rep2], verbose=False, progress_bar=False, data_name=['wf','spec'])
    # check database contents
    fil = di.open_file(out, 'r')
    assert '/assets/data' in fil
    specs = di.load_audio(table=fil.root.assets.data)
    assert len(specs) == 2
    assert type(specs[0][0]) == Waveform
    assert type(specs[0][1]) == MagSpectrogram
    assert fil.root.assets.data.attrs.data_name == ['wf','spec']
    fil.close()
    os.remove(out)

def test_create_database_with_attrs(sine_wave_file):
    """ Create a database including extra attributes from the selection table"""
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db13.h5')
    rep = {'type': 'Waveform'}
    sel = pd.DataFrame({'filename':['sine_wave.wav','sine_wave.wav'], 'start':[0.1,0.2], 'end':[2.0,2.1], 'comment':['big','bigger']})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=False, progress_bar=False, include_attrs=True)
    # check database contents
    fil = di.open_file(out, 'r')
    assert '/assets/data' in fil
    wfs = di.load_audio(table=fil.root.assets.data)
    wf = wfs[0]
    assert type(wf) == Waveform
    attrs = wf.get_instance_attrs()
    assert 'comment' in attrs.keys()
    assert attrs['comment'] == 'big'
    wf = wfs[1]
    attrs = wf.get_instance_attrs()
    assert attrs['comment'] == 'bigger'
    fil.close()
    os.remove(out)

def test_create_database_with_indices(sine_wave_file):
    """ Create a database with indices for some columns"""
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db13.h5')
    rep = {'type': 'Waveform'}
    sel = pd.DataFrame({'filename':['sine_wave.wav','sine_wave.wav'], 'start':[0.1,0.2], 'end':[2.0,2.1], 'extra_id':[13,14]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=False, 
        progress_bar=False, include_attrs=True, index_cols=["filename", "extra_id"])
    # check database contents
    fil = di.open_file(out, 'r')
    assert '/assets/data' in fil
    wfs = di.load_audio(table=fil.root.assets.data)
    wf = wfs[0]
    assert type(wf) == Waveform
    attrs = wf.get_instance_attrs()
    assert 'extra_id' in attrs.keys()
    assert attrs['extra_id'] == 13
    wf = wfs[1]
    attrs = wf.get_instance_attrs()
    assert attrs['extra_id'] == 14
    fil.close()
    os.remove(out)

def test_create_database_with_exception_handling(sine_wave_file):
    """ Check if database can be created if file does not exist """
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db15.h5')
    rep = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav', 'unknown_file_1.wav', 'sine_wave.wav', 'unknown_file_2.wav'], 
                        'start':[0.1, 0.2, 0.3, 0.4], 
                        'end':[2.0, 2.1, 2.2, 2.3], 
                        'label':[1, 2, 1, 1]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=True, progress_bar=False)
    # check database contents
    db = di.open_file(out, 'r')
    assert '/assets/data' in db
    specs = di.load_audio(table=db.root.assets.data)
    assert len(specs) == 2
    db.close()
    os.remove(out)
    
def test_create_database_check_file_duration(sine_wave_file):
    """ Check if database can be created if selections are outside file """
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db16.h5')
    rep = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav', 'sine_wave.wav', 'sine_wave.wav', 'sine_wave.wav', 'sine_wave.wav'], 
                        'start':[2.6, 0.1, -11.0, 4.5, -0.7],
                        'end':[3.6, 1.1, -8.0, 5.5, 0.3],
                        'label':[1, 1, 2, 1, 2]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=True, progress_bar=False)
    # check database contents
    db = di.open_file(out, 'r')
    assert '/assets/data' in db
    specs = di.load_audio(table=db.root.assets.data)
    assert len(specs) == 3
    db.close()
    os.remove(out)

def test_create_database_discard_outside(sine_wave_file):
    """ Check selections outside the file are not saved """
    data_dir = os.path.dirname(sine_wave_file)
    out = os.path.join(path_to_assets, 'tmp/db16.h5')
    rep = {'type': 'Mag', 'window':0.5, 'step':0.1}
    sel = pd.DataFrame({'filename':['sine_wave.wav', 'sine_wave.wav', 'sine_wave.wav', 'sine_wave.wav'], 
                        'start':[0.6, 0.1, -0.2, 4.5],
                        'end':[1.6, 1.1, 0.8, 5.5],
                        'label':[1, 1, 2, 1]})
    sel = use_multi_indexing(sel, 'sel_id')
    di.create_database(out, data_dir=data_dir, selections=sel, audio_repres=rep, verbose=True, progress_bar=False, discard_outside=True)
    # check database contents
    db = di.open_file(out, 'r')
    assert '/assets/data' in db
    specs = di.load_audio(table=db.root.assets.data)
    db.close()
    os.remove(out)
    assert len(specs) == 2
