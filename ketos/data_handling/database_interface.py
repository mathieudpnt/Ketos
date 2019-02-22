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
import math
import numpy as np
from ketos.audio_processing.annotation import tostring
from ketos.audio_processing.audio import AudioSignal
from ketos.audio_processing.spectrogram import Spectrogram

def open(h5file, table_path):
    """ Open a table from an HDF5 file.
        Returns None if the table does not exist.

        Args:
            h5file: tables.file.File object
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

def create(h5file, path, name, shape, max_annotations=10, chunkshape=None, verbose=False):
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
            max_annotations: int
                Maximum number of annotations allowed for any of the items in this table.
            chunkshape: tuple
                The chunk shape to be used for compression

        Returns:
            table: table.Table object
                The created table.    
    """
    max_annotations = max(1, int(max_annotations))

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
        labels_len = 2 + max_annotations * 3 - 1 # assumes that labels have at most 2 digits (i.e. 0-99)
        boxes_len = 2 + max_annotations * (6 + 4*9) - 1 # assumes that box values have format xxxxx.xxx  
        descrip = description(shape=shape, labels_len=labels_len, boxes_len=boxes_len)
        table = h5file.create_table(group, "{0}".format(name), descrip, filters=filters, chunkshape=chunkshape)

    return table

def description(shape, id_len=25, labels_len=100, boxes_len=100, files_len=100):
    """ Create the class that describes the table structure for the HDF5 database.
             
        Args:
            shape : tuple (ints)
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels).

            id_len : int
                The number of character for the 'id' field
            
            labels_len : int
                The number of character for the 'labels' field

            boxes_len : int
                The number of character for the 'boxes' field

        Results:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure to be used when creating tables that 
                will store images in the HDF5 database.
    """
    class TableDescription(tables.IsDescription):
            id = tables.StringCol(id_len)
            labels = tables.StringCol(labels_len)
            data = tables.Float32Col(shape)
            boxes = tables.StringCol(boxes_len) 
            files = tables.StringCol(files_len)
            file_vector = tables.UInt8Col(shape[0])
            time_vector = tables.Float32Col(shape[0])
    
    return TableDescription

def write(table, x, id=None):
    """ Write data into the HDF5 table.

        Note: If the id field is left blank, it 
        will be replaced with the tag attribute.

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored
                (described by spec_description()).

            x: instance of :class:`spectrogram.MagSpectrogram', \
            :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
            :class:`audio_signal.AudioSignal`.
                Data

        Returns:
            None.
    """
    if isinstance(x, AudioSignal):
        table.attrs.sample_rate = x.rate
    elif isinstance(x, Spectrogram):
        table.attrs.time_res = x.tres
        table.attrs.freq_res = x.fres
        table.attrs.freq_min = x.fmin

    if id is None:
        id_str = x.tag
    else:
        id_str = id

    if x.labels is not None:
        labels_str = tostring(x.labels, decimals=0)
    else:
        labels_str = ''

    if x.boxes is not None:          
        boxes_str = tostring(x.boxes, decimals=3)
    else:
        boxes_str = ''

    files_str = '['
    for it in x.get_file_dict().items():
        val = it[1]
        files_str += val + ','
    if files_str[-1] == ',':
        files_str = files_str[:-1] 
    files_str += ']'    

    seg_r = table.row
    seg_r["data"] = x.get_data()
    seg_r["id"] = id_str
    seg_r["labels"] = labels_str
    seg_r["boxes"] = boxes_str
    seg_r["files"] = files_str
    seg_r["file_vector"] = x.get_file_vector()
    seg_r["time_vector"] = x.get_time_vector()

    seg_r.append()

def select(table, label):
    """ Find all objects in the table with the specified label.

        Args:
            table: tables.Table
                Table

            label: int
                Label

        Returns:
            rows: list(int)
                List of row numbers of the objects that have the specified label.
    """
    # selected rows
    rows = list()

    # loop over items in table
    for i, it in enumerate(table):

        # parse labels
        labels = parse_labels(it)

        # check if the specified label is present
        if label not in labels:
            continue

        rows.append(i)

    return rows

def get_objects(table):
    """ Get all objects (spectrogram or audio signal) that are stored in the table.

        Args:
            table: tables.Table
                Table

        Returns:
            res: list
                List of objects.
    """
    res = list()

    # loop over items in table
    for it in table:

        # parse labels and boxes
        labels = parse_labels(it)
        boxes = parse_boxes(it)
        
        # get the data (audio signal or spectrogram)
        data = it['data']

        # create audio signal or spectrogram object
        if np.ndim(data) == 1:
            x = AudioSignal(rate=table.attrs.sample_rate, data=data, tag=it['id'])
        elif np.ndim(data) == 2:
            x = Spectrogram(image=data, tres=table.attrs.time_res, fres=table.attrs.freq_res, fmin=table.attrs.freq_min, tag='')

        # annotate
        x.annotate(labels=labels, boxes=boxes)

        # handle file and time info
        x.time_vector = it['time_vector']
        x.file_vector = it['file_vector']
        files = it['files'].decode()[1:-1]
        if len(files) == 0:
            x.file_dict = {0: ''}
        else:
            files = files.split(',')
            file_dict = {}
            for i, f in enumerate(files):
                file_dict[i] = f
            x.file_dict = file_dict

        res.append(x)

    return res

def extract(table, label, min_length, center=False, fpad=True, preserve_time=False):
    """ Extract segments that match the specified label.

        Args:
            table: tables.Table
                Table
            label: int
                Label
            min_length: float
                Minimum individual duration of extracted segments
            center: bool
                Place labels in the center of the segments
            fpad: bool
                Ensure that all extracted spectrograms have the same 
                frequency range by padding with zeros, if necessary

        Returns:
            selection: list
                List of segments matching the specified label
            complement: spectrogram or audio signal
                Segments (joined) that did not match the specified label
    """

    # selected segments
    selection = list()
    complement = None

    items = get_objects(table)

    # loop over items in table
    for x in items:

        # extract segments of interest
        segs = x.extract(label=label, min_length=min_length, fpad=fpad, center=center, preserve_time=preserve_time)
        selection = selection + segs

        # collect
        if complement is None:
            complement = x
        else:
            complement.append(x)

    return selection, complement

def parse_labels(item):
    """ Parse labels string.

        Args:
            item: 
                Table item

        Returns:
            labels: list(int)
                List of labels
    """
    labels_str = item['labels'].decode()
    labels = np.fromstring(string=labels_str[1:-1], dtype=int, sep=',')
    return labels

def parse_boxes(item):
    """ Parse boxes string.

        Args:
            item: 
                Table item

        Returns:
            labels: list(tuple)
                List of boxes
    """
    boxes_str = item['boxes'].decode()
    boxes_str = boxes_str.replace("inf", "-99")
    try:
        boxes_str = ast.literal_eval(boxes_str)
    except:
        boxes = []

    boxes = np.array(boxes_str)
    boxes[boxes == -99] = math.inf
    return boxes