"""This module contains functions for working with hdf5 tables.

Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: To package code useful for handling data, deriving features and 
             creating Deep Neural Networks for sound classification projects.
     
    License:

"""
import tables
import os
import ast
import numpy as np
import sound_classification.data_handling as dh

def open(h5file, table_path):
    """ Open a table from an HDF5 file.
        Returns None if the table does not exist.

        Args:
            h5: tables.file.File object
                HDF5 file handler.
            table_path: str
                The table's full path.

        Returns:
            table: table.Table object
                The table.    
    """
    try:
       table = h5file.get_node(table_path)
    
    except tables.NoSuchNodeError:  
        print('Attempt to open non-existing table {0} in file {1}'.format(table_path, h5file))
        table = None

    return table

def create(h5file, path, name, shape, chunkshape=None, verbose=False):
    """ Create a new table.

        Args:
            h5file: tables.file.File object
                HDF5 file handler.
            path: str
                The group where the table will be located. Ex: '/features/spectrograms'
            name: str
                The name of the table.
            shape : tuple (ints)
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels).
            chunkshape: tuple
                The chunk shape to be used for compression

        Returns:
            table: table.Table object
                The created table.    
    """
    try:
       group = h5file.get_node(path)
    
    except tables.NoSuchNodeError:
        if verbose:
            print("group '{0}' not found. Creating it now...".format(path))
    
        if path.endswith('/'): 
            path = path[:-1]

        group_name = os.path.basename(path)
        path = path.split(group_name)[0]
        if path.endswith('/'): 
             path = path[:-1]
        
        group = h5file.create_group(path, group_name, createparents=True)
        
    try:
       table = h5file.get_node("{0}/{1}".format(path, name))
    
    except tables.NoSuchNodeError:    
        filters = tables.Filters(complevel=1, fletcher32=True)
        descrip = description(shape=shape)
        table = h5file.create_table(group, "{0}".format(name), descrip, filters=filters, chunkshape=chunkshape)

    return table

def description(shape):
    """ Create the class that describes the table structure for the HDF5 database.
             
        Args:
            shape : tuple (ints)
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels).

        Results:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure to be used when creating tables that 
                will store images in the HDF5 database.
    """
    class TableDescription(tables.IsDescription):
            id = tables.StringCol(25)
            labels = tables.StringCol(100)
            data = tables.Float32Col(shape=shape)
            boxes = tables.StringCol(100) 
    
    return TableDescription

def write(table, x, id=None, labels=None, boxes=None):
    """ Write data into the HDF5 table.

        Note: If the id and labels fields are left blank, an attempt 
        will be made to extract these information from the tag 
        attribute assuming the tag format id_*_[l]_*.
        Example: spec.tag="id_78536_l_[1]"

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored
                (described by spec_description()).

            x: instance of :class:`spectrogram.MagSpectrogram', \
            :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
            :class:`audio_signal.AudioSignal`.
                Data

            id: str
                Identifier (overwrites the id parsed from the tag).

            labels: tuple(int)
                Labels (overwrites the labels parsed from the tag).

            boxes: tuple(tuple(int))
                Boxes confining the regions of interest in time-frequency space

        Returns:
            None.
    """
    id_parsed, labels_parsed = dh.parse_seg_name(x.tag)

    if id is None:
        id_str = id_parsed
    else:
        id_str = id

    if labels is None:
        labels_str = labels_parsed
    else:
        labels_str = dh.tup2str(labels)

    if boxes is not None:          
        if np.ndim(boxes) == 1:
            boxes = (boxes,)

    boxes_str = dh.tup2str(boxes)

    seg_r = table.row
    seg_r["data"] = x.get_data()
    seg_r["id"] = id_str
    seg_r["labels"] = labels_str
    seg_r["boxes"] = boxes_str
    seg_r.append()

def get(table, label, min_length, center=False, folder=None, save_complement=True):
    # selected segments
    segs = list()
    # complement
#    compl
    # loop over items in table
    for it in table:
        labels = parse_labels(it)
        boxes = parse_boxes(it)
        boxes = ensure_min_length(boxes=boxes, min_length=min_length)
        data = it['data']

                

def parse_labels(table):
    labels_str = table['labels'].decode()
    labels = np.fromstring(string=labels_str, dtype=int, sep=',')
    return labels

def parse_boxes(table):
    boxes_str = ast.literal_eval(table['boxes'].decode())
    boxes = np.array(boxes_str)
    return boxes

def ensure_min_length(boxes, min_length):
    return boxes