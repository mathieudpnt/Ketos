""" database_interface module within the ketos library

    This module provides functions to create and use HDF5 databases as storage for acoustic data. 

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
import tables
import os
import ast
import math
import numpy as np
from ketos.utils import tostring
from ketos.audio_processing.audio import AudioSignal
from ketos.audio_processing.spectrogram import Spectrogram

def open_table(h5file, table_path):
    """ Open a table from an HDF5 file.
        
        Args:
            h5file: tables.file.File object
                HDF5 file handler.
            table_path: str
                The table's full path.

        Raises: 
            NoSuchNodeError if table does not exist.

        Returns:
            table: table.Table object or None
                The table, if it exists. Otherwise, raises an exeption and returns None.

        Examples:
        >>> import tables
        >>> from ketos.data_handling.database_interface import open_table

        >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
        >>> data = open_table(h5file, "/train/species1")
        >>> type(data)
        <class 'tables.table.Table'>

        >>> data.nrows
        15
        
    """
    try:
       table = h5file.get_node(table_path)
    
    except tables.NoSuchNodeError:  
        print('Attempt to open non-existing table {0} in file {1}'.format(table_path, h5file))
        table = None

    return table

def create_table(h5file, path, name, shape, max_annotations=10, chunkshape=None, verbose=False):
    """ Create a new table.
        
        If the table already exists, open it.

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
                The created/open table.    

        Examples:

            >>> import tables
            >>> from ketos.data_handling.database_interface import create

            >>> h5file = tables.open_file("database.h5", 'w')
            >>> my_table = create_table(h5file, "/group1/", "table1", shape=(64,20)) 
            >>> my_table
            /group1/table1 (Table(0,), fletcher32, shuffle, zlib(1)) ''
              description := {
              "boxes": StringCol(itemsize=421, shape=(), dflt=b'', pos=0),
              "data": Float32Col(shape=(64, 20), dflt=0.0, pos=1),
              "file_vector": UInt8Col(shape=(64,), dflt=0, pos=2),
              "files": StringCol(itemsize=100, shape=(), dflt=b'', pos=3),
              "id": StringCol(itemsize=25, shape=(), dflt=b'', pos=4),
              "labels": StringCol(itemsize=31, shape=(), dflt=b'', pos=5),
              "time_vector": Float32Col(shape=(64,), dflt=0.0, pos=6)}
              byteorder := 'little'
              chunkshape := (21,)
            
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

def table_description(shape, id_len=25, labels_len=100, boxes_len=100, files_len=100):
    """ Create the class that describes the table structure for the HDF5 database.

        The columns in the table are described as the class Attributes (see Attr section below)
             
        Args:
            shape : tuple (ints)
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels).
            id_len : int
                The number of characters for the 'id' field
            labels_len : int
                The number of characters for the 'labels' field
            boxes_len : int
                The number of characters for the 'boxes' field
            files_len: int
                The number of charecter for the 'files' field.


        Attr:
            id: A string column to store a unique identifier for each entry
            labels: A string column to store the labels associated with each entry
                    Conventional format: '[1], [2], [1]'
            data: A Float32  columns to store arrays with shape defined by the 'shape' argument
            boxes: A string column to store the coordinates (as start time, end time, start frequency, end frequency)\
                    of the acoustic events labelled in the 'labels' field.
                    Conventional format: '[[2, 5, 200, 400],[8, 14, 220, 450],[21, 25, 200, 400]]'
                    Where the first box correspond to the first label ([1]), the second box to the second label ([2]) and so forth. 
            files: A string column to store file names. In case the spectrogram is created from multiple files, this field stores all of them.
            file_vector: An Int8 column to store an array with length equal to the number of bins in the spectrogram.
                         Each value indicates the file that originated that bin, with 0 representing the first file listed in the 'files' field, 1 the second and so on. 
            time_vector: A Float32 column to store an array with length equal to the number of bins in the spectrogram. The array maps each bin to the time (seconds from start) in the original audio file (as stored in the file_vector_field)


        Results:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure to be used when creating tables that 
                will store images in the HDF5 database.

        Examples:
            >>> from ketos.data_handling.database_interface import description
            >>> table_description =  description(shape=(64,20))
            >>> table_description.columns
            {'id': StringCol(itemsize=25, shape=(), dflt=b'', pos=None), 'labels': StringCol(itemsize=100, shape=(), dflt=b'', pos=None), 'data': Float32Col(shape=(64, 20), dflt=0.0, pos=None), 'boxes': StringCol(itemsize=100, shape=(), dflt=b'', pos=None), 'files': StringCol(itemsize=100, shape=(), dflt=b'', pos=None), 'file_vector': UInt8Col(shape=(64,), dflt=0, pos=None), 'time_vector': Float32Col(shape=(64,), dflt=0.0, pos=None)}


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

def write_spec(table, spec, id=None):
    """ Write data into the HDF5 table.

        Note: If the id field is left blank, it 
        will be replaced with the tag attribute.

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored
                (described by spec_description()).

            spec: instance of :class:`spectrogram.MagSpectrogram', \
            :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
                the spectrogram object to be stored in the table.

        Raises:
            TypeError: if spec is not an Spectrogram object    

        Returns:
            None.
    """

    try:
        assert(isinstance(spec, Spectrogram))
        table.attrs.time_res = spec.tres
        table.attrs.freq_res = spec.fres
        table.attrs.freq_min = spec.fmin
    except AssertionError:
        raise TypeError("spec must be an instance of Spectrogram")      

    if id is None:
        id_str = spec.tag
    else:
        id_str = id

    if x.labels is not None:
        labels_str = tostring(spec.labels, decimals=0)
    else:
        labels_str = ''

    if x.boxes is not None:          
        boxes_str = tostring(spec.boxes, decimals=3)
    else:
        boxes_str = ''

    files_str = '['
    for it in spec.get_file_dict().items():
        val = it[1]
        files_str += val + ','
    if files_str[-1] == ',':
        files_str = files_str[:-1] 
    files_str += ']'    

    seg_r = table.row
    seg_r["data"] = spec.get_data()
    seg_r["id"] = id_str
    seg_r["labels"] = labels_str
    seg_r["boxes"] = boxes_str
    seg_r["files"] = files_str
    seg_r["file_vector"] = spec.get_file_vector()
    seg_r["time_vector"] = spec.get_time_vector()

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