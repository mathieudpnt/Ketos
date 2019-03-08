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
from ketos.audio_processing.spectrogram import Spectrogram,MagSpectrogram,PowerSpectrogram, MelSpectrogram

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
            >>> from ketos.data_handling.database_interface import create_table

            >>> h5file = tables.open_file("ketos/tests/assets/tmp/database1.h5", 'w')
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

            >>> h5file.close()
            
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
        descrip = table_description(shape=shape, labels_len=labels_len, boxes_len=boxes_len)
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
                The number of characters for the 'files' field.


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
            >>> from ketos.data_handling.database_interface import table_description
            >>> descr =  table_description(shape=(64,20))
            >>> descr.columns
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

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import create_table
            >>> from ketos.audio_processing.spectrogram import MagSpectrogram
            >>> from ketos.audio_processing.audio import AudioSignal

            >>> audio = AudioSignal.from_wav('ketos/tests/assets/2min.wav')
            >>> spec = MagSpectrogram(audio,winlen=0.2, winstep=0.05)
            >>> spec.labels = [1,2]
            >>> spec.boxes = [[5.3,8.9,200,350], [103.3,105.8,180,320]]
           
            >>> h5file = tables.open_file("ketos/tests/assets/tmp/database2.h5", 'w')
            >>> my_table = create_table(h5file, "/group1/", "table1", shape=spec.image.shape)
            >>> write_spec(my_table, spec)
            >>> my_table.nrows
            1
            >>> my_table[0]['labels']
            b'[1,2]'
            >>> my_table[0]['boxes']
            b'[[5.3,8.9,200.0,350.0],[103.3,105.8,180.0,320.0]]'
            
            >>> h5file.close()

            

    """

    try:
        assert(isinstance(spec, Spectrogram))
    except AssertionError:
        raise TypeError("spec must be an instance of Spectrogram")      

    table.attrs.time_res = spec.tres
    table.attrs.freq_res = spec.fres
    table.attrs.freq_min = spec.fmin

    if id is None:
        id_str = spec.tag
    else:
        id_str = id

    if spec.labels is not None:
        labels_str = tostring(spec.labels, decimals=0)
    else:
        labels_str = ''

    if spec.boxes is not None:          
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

def filter_by_label(table, label):
    """ Find all spectrograms in the table with the specified label.

        Args:
            table: tables.Table
                The table containing the spectrograms
            label: int or list of ints
                The labels to be searched
        Raises:
            TypeError: if label is not an int or list of ints.

        Returns:
            rows: list(int)
                List of row numbers of the objects that have the specified label.
                If there are no spectrograms that match the label, returs an empty list.

        Examples:

            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table

            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> table = open_table(h5file, "/train/species1")

            >>> filter_by_label(table, 1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

            >>> filter_by_label(table, 2)
            []
    """
    if isinstance(label, (list)):
        if not all (isinstance(l, int) for l in label):
            raise TypeError("label must be an int or a list of ints")    
    elif isinstance(label, int):
        label = [label]
    else:
        raise TypeError("label must be an int or a list of ints")    
    
    condition = "|".join(["(labels == b'[{0}]')".format(l) for l in label])
    rows = list(table.get_where_list(condition))

    return rows

def load_specs(table, index_list=None):
    """ Retrieve all the spectrograms in a table or a subset specified by the index_list

        Warnings: Loading all spectrograms in a table might cause memory problems.

        Args:
            table: tables.Table
                The table containing the spectrogtrams
            index_list: list of ints or None
                A list with the indices of the spectrograms that will be retrieved.
                If set to None, loads all spectrograms in the table.

        Returns:
            res: list
                List of spectrogram objects.


        Examples:

            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table

            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> table = open_table(h5file, "/train/species1")

            >>> selected_specs = load_specs(table, [0,3,10])
            >>> len(selected_specs)
            3
            >>> type(selected_specs[0])
            <class 'ketos.audio_processing.spectrogram.Spectrogram'>

    """
    res = list()
    if index_list is None:
        index_list = list(range(table.nrows))

    # loop over items in table
    for idx in index_list:

        it = table[idx]
        # parse labels and boxes
        labels = parse_labels(it)
        boxes = parse_boxes(it)
        
        # get the spectrogram data
        data = it['data']

        # create audio signal or spectrogram object
        x = Spectrogram(image=data, tres=table.attrs.time_res, fres=table.attrs.freq_res, fmin=table.attrs.freq_min, tag='')

        # annotate
        #import pdb; pdb.set_trace()
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
    """ Create new spectrograms by croping segments annotated with the specified label.

        Filter the table by the specified label. In each of the selected spectrograms,
        search for the individual annotations (i.e.: boxes that match the specified label).
        Each annotation will result in a new spectrogram. After the matching segments are extracted,
        what is left is stiched together to create a spectrogram object. This spectrogram contains all
        the parts of the original spectrogram that did not contain any matching annotations and is treated
        as a complement to the extracted spectrograms. All the extracted spectrograms are returned in a list and 
        the complements in another.
        
        Any spectrogram in the table that does not contain any annotations of interest is added to the complements list. 

        Args:
            table: tables.Table
                The table containing the spectrograms.
            label: int
                The label
            min_length: float
                Minimum duration (in seconds) the of extracted segments.
            center: bool
                If True, place the annotation box in the center of the segments.
                Otherwise, place it randomly within the spectrogram.

            fpad: bool
                If True, ensure that all extracted spectrograms have the same 
                frequency range by padding with zeros, if necessary.
                If False, the resulting spectrograms will be cropped at the minimum
                and maximum frequencies specified by the bounding box.

        Returns:
            selection: list
                List of spectrograms segments matching the specified label
            complement: spectrogram or audio signal
                A list of spectrograms containing the joined segments that did not match the specified label.
                There will be on such spectrogram for each of the original spectrograms in the table.

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table

            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> table = open_table(h5file, "/train/species1")

            >>> extracted_specs, spec_complements = extract(table, label=1, min_length=2)
            >>> len(extracted_specs)
            15
            >>> len(specs_complements)
            15

            >>> spec_1_fig = extracted_specs[0].plot()
            >>> comp_1_fig = spec_complements[0].plot()


    """

    # selected segments
    selection = list()
    complement = None

    items = load_specs(table)

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
    if (boxes == -99).any():
         boxes[boxes == -99] = math.inf
    return boxes