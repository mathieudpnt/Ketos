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

""" 'data_handling.database_interface' module within the ketos library

    This module provides functions to create and use HDF5 databases as storage for acoustic data. 
"""

import librosa
import tables
import os
import ast
import math
import numpy as np
import pandas as pd
from ketos.utils import tostring
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram, MagSpectrogram, PowerSpectrogram, CQTSpectrogram, MelSpectrogram
from ketos.data_handling.data_handling import find_wave_files, AnnotationTableReader, rel_path_unix, SpecProvider
from ketos.data_handling.parsing import SpectrogramConfiguration
from tqdm import tqdm
from sys import getsizeof
from psutil import virtual_memory


def open_file(path, mode):
    """ Open an HDF5 database file.

        Wrapper function around tables.open_file: 
        https://www.pytables.org/usersguide/libref/top_level.html
        
        Args:
            path: str
                The file's full path.
            mode: str
                The mode to open the file. It can be one of the following:
                    * ’r’: Read-only; no data can be modified.
                    * ’w’: Write; a new file is created (an existing file with the same name would be deleted).
                    * ’a’: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    * ’r+’: It is similar to ‘a’, but the file must already exist.

        Returns:
            : table.File object
                The h5file.
    """
    return tables.open_file(path, mode)

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
            >>> from ketos.data_handling.database_interface import open_file, open_table
            >>> h5file = open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> data = open_table(h5file, "/train/species1")
            >>> #data is a pytables 'Table' object
            >>> type(data)
            <class 'tables.table.Table'>
            >>> # with 15 items (rows)
            >>> data.nrows
            15
            >>> h5file.close()       
    """
    try:
       table = h5file.get_node(table_path)
    
    except tables.NoSuchNodeError:  
        print('Attempt to open non-existing table {0} in file {1}'.format(table_path, h5file))
        raise
        table = None

    return table

def create_table(h5file, path, name, description, chunkshape=None, verbose=False):
    """ Create a new table.
        
        If the table already exists, open it.

        Args:
            h5file: tables.file.File object
                HDF5 file handler.
            path: str
                The group where the table will be located. Ex: '/features/spectrograms'
            name: str
                The name of the table.
            table_description: class (tables.IsDescription)
                The class describing the table structure.            
            chunkshape: tuple
                The chunk shape to be used for compression

        Returns:
            table: table.Table object
                The created/open table.    

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, table_description_new, create_table_new
            >>> # Open a connection to the database
            >>> h5file = open_file("ketos/tests/assets/tmp/database1.h5", 'w')
            >>> # Create table descriptions for weakly labeled spectrograms with shape (32,64)
            >>> descr_data, descr_annot = table_description_new((32,64))
            >>> # Create 'table_data' within 'group1'
            >>> my_table = create_table_new(h5file, "/group1/", "table_data", descr_data) 
            >>> # Show the table description, with the field names (columns)
            >>> # and information about types and shapes
            >>> my_table
            /group1/table_data (Table(0,), fletcher32, shuffle, zlib(1)) ''
              description := {
              "data": Float32Col(shape=(32, 64), dflt=0.0, pos=0),
              "filename": StringCol(itemsize=100, shape=(), dflt=b'', pos=1),
              "id": UInt32Col(shape=(), dflt=0, pos=2),
              "offset": Float64Col(shape=(), dflt=0.0, pos=3)}
              byteorder := 'little'
              chunkshape := (15,)
            >>> # Close the HDF5 database file
            >>> h5file.close()            
    """
    if path.endswith('/'): 
        path = path[:-1]

    try:
       group = h5file.get_node(path)
    
    except tables.NoSuchNodeError:
        if verbose:
            print("group '{0}' not found. Creating it now...".format(path))
    
        group_name = os.path.basename(path)
        path_to_group = path.split(group_name)[0]
        if path_to_group.endswith('/'): 
            path_to_group = path_to_group[:-1]
        
        group = h5file.create_group(path_to_group, group_name, createparents=True)
        
    try:
        table = h5file.get_node("{0}/{1}".format(path, name))
    
    except tables.NoSuchNodeError:    
        filters = tables.Filters(complevel=1, fletcher32=True)
        table = h5file.create_table(group, "{0}".format(name), description, filters=filters, chunkshape=chunkshape)

    return table

def table_description_data(data_shape, track_source=True, filename_len=100):
    """ Description of table structure for storing audio signals or spectrograms.

        Args:
            data_shape: tuple (ints)
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels). 
            track_source: bool
                If True, the name of the wav file from which the audio signal or 
                spectrogram was generated and the placement within that file, is 
                saved to the table. Default is True.
            filename_len: int
                Maximum allowed length of filename. Only used if track_source is True.

        Returns:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure.
    """
    class TableDescription(tables.IsDescription):
        id = tables.UInt32Col()
        data = tables.Float32Col(data_shape)
        if track_source:
            filename = tables.StringCol(filename_len)
            offset = tables.Float64Col()

    return TableDescription

def table_description_weak_annot():
    """ Table description for weak annotations.

        Returns:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure.
    """
    class TableDescription(tables.IsDescription):
        data_id = tables.UInt32Col()
        label = tables.UInt8Col()

    return TableDescription

def table_description_strong_annot(freq_range=False):
    """ Table descriptions for strong annotations.

        Args:
            freq_range: bool
                Set to True, if your annotations include frequency range. Otherwise, 
                set to False (default). Only used for strong annotations.

        Returns:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure.
    """
    class TableDescription(tables.IsDescription):
        data_id = tables.UInt32Col()
        label = tables.UInt8Col()
        start = tables.Float64Col()
        end = tables.Float64Col()
        if freq_range:
            freq_min = tables.Float32Col()
            freq_max = tables.Float32Col()

    return TableDescription

def table_description(data_shape, annot_type='weak', track_source=True, filename_len=100, freq_range=False):
    """ Create HDF5 table structure description.

        The annotation type must be specified as either 'weak' or 'strong'.

        An audio segment or spectrogram is said to be 'weakly annotated', if it is  assigned a single 
        (integer) label, and is said to be 'strongly annotated', if it is assigned one or several 
        labels, each accompanied by a start and end time, and potentially also a minimum and maximum 
        frequecy.

        When the annotation type is set to 'weak', the method returns a single table description.

        When the annotation type is set to 'strong', the method returns two table descriptions, one for 
        the data table and one for the annotation table.

        Args:
            data_shape: tuple (ints) or numpy array or :class:`spectrogram.Spectrogram'
                The shape of the audio signal (n_samples) or spectrogram (n_rows,n_cols) 
                to be stored in the table. Optionally, a third integer can be added if the 
                spectrogram has multiple channels (n_rows, n_cols, n_channels). 
                If a numpy array is provided, the shape is deduced from this array.
                If an instance of the Spectrogram class is provided, the shape is deduced from 
                the image attribute.
            annot_type: str
                The annotation type. Permitted values are 'weak' and 'strong'. The default 
                value is 'weak'.
            track_source: bool
                If True, the name of the wav file from which the audio signal or 
                spectrogram was generated and the placement within that file, is 
                saved to the table. Default is True.
            filename_len: int
                Maximum allowed length of filename. Only used if track_source is True.
            freq_range: bool
                Set to True, if your annotations include frequency range. Otherwise, 
                set to False (default). Only used for strong annotations.

        Returns:
            tbl_descr_data: class (tables.IsDescription)
                The class describing the table structure for the data.
            tbl_descr_annot: class (tables.IsDescription)
                The class describing the table structure for the annotations.

        Examples:
            >>> import numpy as np
            >>> from ketos.data_handling.database_interface import table_description
            >>> 
            >>> #Create a 64 x 20 image
            >>> spec = np.random.random_sample((64,20))
            >>>
            >>> #Create a table description for weakly labeled spectrograms of this shape
            >>> descr_data, descr_annot = table_description(spec)
            >>>
            >>> #Inspect the table structure
            >>> cols = descr_data.columns
            >>> for key in sorted(cols.keys()):
            ...     print("%s: %s" % (key, cols[key]))
            data: Float32Col(shape=(64, 20), dflt=0.0, pos=None)
            filename: StringCol(itemsize=100, shape=(), dflt=b'', pos=None)
            id: UInt32Col(shape=(), dflt=0, pos=None)
            offset: Float64Col(shape=(), dflt=0.0, pos=None)
            >>> cols = descr_annot.columns
            >>> for key in sorted(cols.keys()):
            ...     print("%s: %s" % (key, cols[key]))
            data_id: UInt32Col(shape=(), dflt=0, pos=None)
            label: UInt8Col(shape=(), dflt=0, pos=None)
            >>>
            >>> #Create a table description for strongly labeled spectrograms
            >>> descr_data, descr_annot =  table_description(spec, annot_type='strong')
            >>>
            >>> #Inspect the annotation table structure
            >>> cols = descr_annot.columns
            >>> for key in sorted(cols.keys()):
            ...     print("%s: %s" % (key, cols[key]))
            data_id: UInt32Col(shape=(), dflt=0, pos=None)
            end: Float64Col(shape=(), dflt=0.0, pos=None)
            label: UInt8Col(shape=(), dflt=0, pos=None)
            start: Float64Col(shape=(), dflt=0.0, pos=None)
    """
    assert annot_type in ['weak','strong'], 'Invalid annotation type. Permitted types are weak and strong.'

    if isinstance(data_shape, np.ndarray):
        data_shape = data_shape.shape
    elif isinstance(data_shape, Spectrogram):
        data_shape = data_shape.image.shape

    tbl_descr_data = table_description_data(data_shape=data_shape,  track_source=track_source, filename_len=filename_len)

    if annot_type == 'weak':
        tbl_descr_annot = table_description_weak_annot()
    
    elif annot_type == 'strong':
        tbl_descr_annot = table_description_strong_annot(freq_range=freq_range)

    return tbl_descr_data, tbl_descr_annot

def write_spec_attrs(table, spec):
    """ Writes the spectrogram attributes into the HDF5 table.

        The attributes include,

            * Time resolution in seconds (time_res)
            * Minimum frequency in Hz (freq_min)
            * Spectrogram type (type)
            * Frequency resolution in Hz (freq_res) or, in the case of
              CQT spectrograms, the number of bins per octave (bins_per_octave).

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored
                (described by spec_description()).
            spec: instance of :class:`spectrogram.MagSpectrogram', \
                :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
                :class:`spectrogram.CQTSpectrogram'    
                The spectrogram object to be stored in the table.

        Raises:
            TypeError: if spec is not an Spectrogram object    

        Returns:
            None.

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, create_table, table_description_data, write_spec_attrs
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> from ketos.audio.waveform import Waveform
            >>>
            >>> # Create an Waveform object from a .wav file
            >>> audio = Waveform.from_wav('ketos/tests/assets/2min.wav')
            >>> # Use that signal to create a spectrogram
            >>> spec = MagSpectrogram(audio, winlen=0.2, winstep=0.05)
            >>>
            >>> # Open a connection to a new HDF5 database file
            >>> h5file = open_file("ketos/tests/assets/tmp/database2.h5", 'w')
            >>> # Create table descriptions for storing the spectrogram data
            >>> descr = table_description_data(spec.image.shape)
            >>> # Create 'table_data' within 'group1'
            >>> my_table = create_table(h5file, "/group1/", "table_data", descr) 
            >>> # Write spectrogram attributes to the table
            >>> write_spec_attrs(my_table, spec)
            >>>
            >>> # The table now has the following attributes
            >>> my_table.attrs.freq_min
            0
            >>> round(my_table.attrs.freq_res, 3)
            4.975
            >>> my_table.attrs.time_res
            0.05
            >>> my_table.attrs.type
            'Mag'
            >>> h5file.close()
    """
    try:
        assert(isinstance(spec, Spectrogram))
    except AssertionError:
        raise TypeError("spec must be an instance of Spectrogram")      

    table.attrs.time_res = spec.time_res()
    table.attrs.freq_min = spec.freq_min()

    if isinstance(spec, CQTSpectrogram):
        table.attrs.type = 'CQT'
        table.attrs.bins_per_octave = spec.bins_per_octave()
    
    elif isinstance(spec, MagSpectrogram):
        table.attrs.type = 'Mag'
        table.attrs.freq_res = spec.freq_res()
    
    elif isinstance(spec, PowerSpectrogram):
        table.attrs.type = 'Pow'
        table.attrs.freq_res = spec.freq_res()

    elif isinstance(spec, MelSpectrogram):
        table.attrs.type = 'Mel'
        table.attrs.freq_res = spec.freq_res() #OBS: this will fail

def write_spec_annot(spec, table, id):
    """ Write a spectrogram's annotations to a HDF5 table.

        Args:
            spec: instance of :class:`spectrogram.MagSpectrogram', \
                :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
                :class:`spectrogram.CQTSpectrogram'    
                The spectrogram object the annotations of which will be stored in the table.
            table: tables.Table
                Table in which the annotations will be stored.
                (described by table_description_data()).
            id: int
                Spectrogram unique identifier.

        Raises:
            TypeError: if spec is not an Spectrogram object    

        Returns:
            None.
    """
    try:
        assert(isinstance(spec, Spectrogram))
    except AssertionError:
        raise TypeError("spec must be an instance of Spectrogram")      

    write_time = ("start" in table.colnames)
    write_freq = ("freq_min" in table.colnames)

    for box,label in zip(spec.boxes, spec.labels):
        row = table.row

        row["data_id"] = id
        row["label"] = label

        if write_time:
            row["start"] = box[0]
            row["end"] = box[1]

        if write_freq:
            row["freq_min"] = box[2]
            row["freq_max"] = box[3]

        row.append()

def write_spec_data(spec, table, id=None):
    """ Write spectrogram to a HDF5 table.

        Args:
            spec: instance of :class:`spectrogram.MagSpectrogram', \
                :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
                :class:`spectrogram.CQTSpectrogram'    
                The spectrogram object to be stored in the table.
            table: tables.Table
                Table in which the spectrogram will be stored.
                (described by table_description_data()).
            id: int
                Spectrogram unique identifier. Optional

        Raises:
            TypeError: if spec is not an Spectrogram object    

        Returns:
            id: int
                Unique identifier given to spectrogram.
    """
    try:
        assert(isinstance(spec, Spectrogram))
    except AssertionError:
        raise TypeError("spec must be an instance of Spectrogram")      

    write_source = ("filename" in table.colnames)

    row = table.row
    
    if id is None:
        id = table.nrows

    row['id'] = id   
    row['data'] = spec.get_data()

    if write_source:
        row['filename'] = spec.tag
        row['offset'] = spec.tmin

    row.append()

    return id

def write_spec(spec, table_data, table_annot=None, id=None):
    """ Write the spectrogram and its annotations to HDF5 tables.

        Note: If the id argument is not specified, the row number will 
        will be used as a unique identifier for the spectrogram.

        Note: If table_annot is not specified, the annotation data 
        will not be written to file.        

        Args:
            spec: instance of :class:`spectrogram.MagSpectrogram', \
                :class:`spectrogram.PowerSpectrogram', :class:`spectrogram.MelSpectrogram', \
                :class:`spectrogram.CQTSpectrogram'    
                The spectrogram object to be stored in the table.
            table_data: tables.Table
                Table in which the spectrogram will be stored.
                (described by table_description_data()).
            table_annot: tables.Table
                Table in which the annotations will be stored.
                (described by table_description_weak_annot() or table_description_strong_annot()).
            id: int
                Spectrogram unique identifier. Optional.

        Raises:
            TypeError: if spec is not an Spectrogram object    

        Returns:
            None.

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, create_table_new, table_description_data, write_spec_new
            >>> from ketos.audio_processing.spectrogram import MagSpectrogram
            >>> from ketos.audio_processing.audio import Waveform
            >>>
            >>> # Create an Waveform object from a .wav file
            >>> audio = Waveform.from_wav('ketos/tests/assets/2min.wav')
            >>> # Use that signal to create a spectrogram
            >>> spec = MagSpectrogram(audio, winlen=0.2, winstep=0.05)
            >>> # Add a single annotation
            >>> spec.annotate(1, [0.,2.])
            >>>
            >>> # Open a connection to a new HDF5 database file
            >>> h5file = open_file("ketos/tests/assets/tmp/database2.h5", 'w')
            >>> # Create table descriptions for storing the spectrogram data
            >>> descr_data, descr_annot = table_description_new(spec, annot_type='strong')
            >>> # Create tables
            >>> tbl_data = create_table_new(h5file, "/group1/", "table_data", descr_data) 
            >>> tbl_annot = create_table_new(h5file, "/group1/", "table_annot", descr_annot) 
            >>> # Write spectrogram and its annotation to the tables
            >>> write_spec_new(spec, tbl_data, tbl_annot)
            >>> # flush memory to ensure data is put in the tables
            >>> tbl_data.flush()
            >>> tbl_annot.flush()
            >>>
            >>> # Check that the spectrogram data have been saved 
            >>> tbl_data.nrows
            1
            >>> tbl_annot.nrows
            1
            >>> # Check annotation data
            >>> tbl_annot[0]['label']
            1
            >>> tbl_annot[0]['start']
            0.0
            >>> tbl_annot[0]['end']
            2.0
            >>> # Check audio source data
            >>> tbl_data[0]['filename'].decode()
            '2min.wav'
            >>> h5file.close()
    """
    id = write_spec_data(spec, table=table_data, id=id)

    if table_annot is not None:
        write_spec_annot(spec, table=table_annot, id=id)





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
            >>>
            >>> # Open a database and an existing table
            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> table = open_table(h5file, "/train/species1")
            >>>
            >>> # Retrieve the indices for all spectrograms that contain the label 1
            >>> # (all spectrograms in this table)
            >>> filter_by_label(table, 1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            >>>
            >>> # Since none of the spectrograms in the table include the label 2, 
            >>> # an empty list is returned
            >>> filter_by_label(table, 2)
            []
            >>>
            >>> h5file.close()
    """
    if isinstance(label, (list)):
        if not all (isinstance(l, int) for l in label):
            raise TypeError("label must be an int or a list of ints")    
    elif isinstance(label, int):
        label = [label]
    else:
        raise TypeError("label must be an int or a list of ints")    
    
    
    matching_rows = []

    for i,row in enumerate(table.iterrows()):
        r_labels = row['labels']
        r_labels = parse_labels(r_labels)

        if any([l in label for l in r_labels]):
            matching_rows.append(i)
    

    return matching_rows

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
            >>>
            >>> # Open a connection to the database.
            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> # Open the species1 table in the train group
            >>> table = open_table(h5file, "/train/species1")
            >>>
            >>> # Load the spectrograms stored on rows 0, 3 and 10 of the species1 table
            >>> selected_specs = load_specs(table, [0,3,10])
            >>> # The resulting list has the 3 spectrogram objects
            >>> len(selected_specs)
            3
            >>> type(selected_specs[0])
            <class 'ketos.audio_processing.spectrogram.Spectrogram'>
            >>>
            >>> h5file.close()

    """
    res = list()
    if index_list is None:
        index_list = list(range(table.nrows))

    # loop over items in table
    for idx in index_list:

        it = table[idx]
        # parse labels and boxes
        labels = it['labels']
        labels = parse_labels(labels)
        boxes = it['boxes']
        boxes = parse_boxes(boxes)
        
        # get the spectrogram data
        data = it['data']

        # create spectrogram object
        if table.attrs.freq_res >= 0:
            x = Spectrogram(image=data, tres=table.attrs.time_res, fres=table.attrs.freq_res, fmin=table.attrs.freq_min, tag='')
        else:
            x = CQTSpectrogram(image=data, winstep=table.attrs.time_res, bins_per_octave=int(-table.attrs.freq_res), fmin=table.attrs.freq_min, tag='')

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

def extract(table, label, length=None, min_length=None, center=False, fpad=True, keep_time=False):
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
            length: float
                Extend or divide the annotation boxes as necessary to ensure that all 
                extracted segments have the specified length (in seconds).  
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
            keep_time: bool
                If True, the initial time in the extracted spectrograms will maintained
                 (i.e.: will be equal to the start_time of the box).
                If false, the initial time is set to 0. 

            

        Returns:
            extractated: list
                List of spectrograms, created from segments matching the specified label
            complements: spectrogram or audio signal
                A list of spectrograms containing the joined segments that did not match the specified label.
                There will be on such spectrogram for each of the original spectrograms in the table.
                In case nothing is left after the extraction (i.e.: the whole spectrogram was covered by 
                the box/label of interest), the item in the complement list will have a None value.

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table
            >>>
            >>> # Open a connection to the database.
            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> # Open the species1 table in the train group
            >>> table = open_table(h5file, "/train/species1")
            >>>
            >>> # Extract the portions of the spectrograms annotated with label 1
            >>> extracted_specs, spec_complements = extract(table, label=1, min_length=2)
            >>> h5file.close()
            >>>
            >>> # The results show 
            >>> len(extracted_specs)
            15
            >>> len(spec_complements)
            15
            >>> 
            >>> #Plot one of the extracted spectrograms        
            >>> spec_1_fig = extracted_specs[0].plot()
            >>> spec_1_fig.savefig("ketos/tests/assets/tmp/extract_spec_1.png")
              
            .. image:: ../../../../ketos/tests/assets/tmp/extract_spec_1.png
                           
            >>> # Plot the portion without any annotations
            >>> comp_1_fig = spec_complements[0].plot()
            >>> comp_1_fig.savefig("ketos/tests/assets/tmp/extract_comp_1.png")
           
            .. image:: ../../../../ketos/tests/assets/tmp/extract_comp_1.png

            

    """

    # selected segments
    extracted = list()
    complements = list()

    items = load_specs(table)

    # loop over items in table
    for spec in items:

        # extract segments of interest
        segs = spec.extract(label=label, length=length, min_length=min_length, fpad=fpad, center=center, keep_time=keep_time)
        extracted = extracted + segs

        # collect
        complements.append(spec)
    

    return extracted, complements

def parse_labels(label):
    """ Parse the 'labels' field from an item in a hdf5 spectrogram table 
        

        Args:
            label: bytes str
            The bytes string containing the label (e.g.:b'[1]', b'[1,2]')

        Returns:
            parsed_labels: list(int)
            List of labels

        Example:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table
            >>>
            >>> # Open a connection to the database.
            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> # Open the species1 table in the train group
            >>> table = open_table(h5file, "/train/species1")
            >>>
            >>> #The labels are stored as byte strings in the table
            >>> type(table[0]['labels'])
            <class 'numpy.bytes_'>
            >>> table[0]['labels']
            b'[1]'
            >>>
            >>> label =table[0]['labels']
            >>> parsed_label = parse_labels(label)
            >>> type(parsed_label)
            <class 'list'>
            >>> # After parsing, they are lists of integers and can be used as such
            >>> parsed_label
            [1]
            >>>
            >>> h5file.close()
  
    """
    labels_str = label.decode()
    parsed_labels = np.fromstring(string=labels_str[1:-1], dtype=int, sep=',')
    parsed_labels = list(parsed_labels)
    return parsed_labels

def parse_boxes(boxes):
    """ Parse the 'boxes' field from an item in a hdf5 spectrogram table

        Args:
            boxex: bytes str
            The bytes string containing the label (e.g.:b'[[105, 107, 200,400]]', b'[[105,107,200,400], [230,238,220,400]]')
        Returns:
            labels: list(tuple)
                List of boxes
        Example:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_table
            >>>
            >>> # Open a connection to the database.
            >>> h5file = tables.open_file("ketos/tests/assets/15x_same_spec.h5", 'r')
            >>> # Open the species1 table in the train group
            >>> table = open_table(h5file, "/train/species1")
            >>>
            >>> #The boxes are stored as byte strings in the table
            >>> boxes = table[0]['boxes']
            >>> type(boxes)
            <class 'numpy.bytes_'>
            >>> boxes
            b'[[10,15,200,400]]'
            >>>
            >>> parsed_box = parse_boxes(boxes)
            >>> type(parsed_box)
            <class 'list'>
            >>> # After parsing, the all of boxes becomes a list, which
            >>> # has each box as a list of integers
            >>> parsed_box
            [[10, 15, 200, 400]]
            >>>
            >>> h5file.close()
    """
    
    boxes_str = boxes.decode()
    boxes_str = boxes_str.replace("inf", "-99")
    try:
        boxes_str = ast.literal_eval(boxes_str)
    except:
        parsed_boxes = []

    parsed_boxes = np.array(boxes_str)

    if (parsed_boxes == -99).any():
         parsed_boxes[parsed_boxes == -99] = math.inf

    parsed_boxes = parsed_boxes.tolist()
    
    return parsed_boxes

def create_spec_database(output_file, input_dir, annotations_file=None, spec_config=None,\
        sampling_rate=None, channel=0, window_size=0.2, step_size=0.02, duration=None,\
        overlap=0, flow=None, fhigh=None, max_size=1E9, progress_bar=False, verbose=True, cqt=False,\
        bins_per_octave=32, pad=True, **kwargs):
    """ Create a database with magnitude spectrograms computed from raw audio (*.wav) files
        
        One spectrogram is created for each audio file using either a short-time Fourier transform (STFT) or
        a constant-Q transform (CQT).
        
        However, if the spectrogram is longer than the specified duration, the spectrogram 
        will be split into segments, each with the desired duration.

        On the other hand, if a spectrogram is shorter than the specified duration, the spectrogram will 
        be padded with zeros to achieve the desired duration. 

        If duration is not specified (default), it will be set equal to the duration 
        of the first spectrogram that is processed.

        Thus, all saved spectrograms will have the same duration.

        If the combined size of the spectrograms exceeds max_size (1 GB by default), the output database 
        file will be split into several files, with _000, _001, etc, appended to the filename.

        The internal file structure of the database file will mirror the structure of the 
        data directory where the audio data is stored. See the example below.

        Note that if spec_config is specified, the following arguments are ignored: 
        sampling_rate, window_size, step_size, duration, overlap, flow, fhigh, cqt, bins_per_octave.

        TODO: Modify implementation so that arguments are not ignored when spec_config is specified.

        Args:
            output_file: str
                Full path to output database file (*.h5)
            input_dir: str
                Full path to folder containing the input audio files (*.wav)
            annotations_file: str
                Full path to file containing annotations (*.csv)
            spec_config: SpectrogramConfiguration
                Spectrogram configuration object.
            sampling_rate: float
                If specified, audio data will be resampled at this rate
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            window_size: float
                Window size (seconds) used for computing the spectrogram
            step_size: float
                Step size (seconds) used for computing the spectrogram
            duration: float
                Duration in seconds of individual spectrograms.
            overlap: float
                Overlap in seconds between consecutive spectrograms.
            flow: float
                Lower cut on frequency (Hz)
            fhigh: float
                Upper cut on frequency (Hz)
            max_size: int
                Maximum size of output database file in bytes
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte)
            progress_bar: bool
                Option to display progress bar.
            verbose: bool
                Print relevant information during execution such as files written to disk
            cqt: bool
                Compute CQT magnitude spectrogram instead of the standard STFT magnitude 
                spectrogram.
            bins_per_octave: int
                Number of bins per octave. Only applicable if cqt is True.
            pad: bool
                If True (default), audio files will be padded with zeros at the end to produce an 
                integer number of spectrogram if necessary. If False, audio files 
                will be truncated at the end.

            Example:

                >>> # create a few audio files and save them as *.wav files
                >>> from ketos.audio_processing.audio import Waveform
                >>> cos7 = Waveform.cosine(rate=1000, frequency=7.0, duration=1.0)
                >>> cos8 = Waveform.cosine(rate=1000, frequency=8.0, duration=1.0)
                >>> cos21 = Waveform.cosine(rate=1000, frequency=21.0, duration=1.0)
                >>> folder = "ketos/tests/assets/tmp/harmonic/"
                >>> cos7.to_wav(folder+'cos7.wav')
                >>> cos8.to_wav(folder+'cos8.wav')
                >>> cos21.to_wav(folder+'highfreq/cos21.wav')
                >>> # now create a database of spectrograms from these audio files
                >>> from ketos.data_handling.database_interface import create_spec_database
                >>> fout = folder + 'harmonic.h5'
                >>> create_spec_database(output_file=fout, input_dir=folder)
                3 spectrograms saved to ketos/tests/assets/tmp/harmonic/harmonic.h5
                >>> # inspect the contacts of the database file
                >>> import tables
                >>> f = tables.open_file(fout, 'r')
                >>> print(f.root.spec)
                /spec (Table(2,), fletcher32, shuffle, zlib(1)) ''
                >>> print(f.root.highfreq.spec)
                /highfreq/spec (Table(1,), fletcher32, shuffle, zlib(1)) ''
                >>> f.close()
    """
    # annotation reader
    if annotations_file is None:
        areader = None
        max_ann = 1
    else:
        areader = AnnotationTableReader(annotations_file)
        max_ann = areader.get_max_annotations()

    # spectrogram writer
    swriter = SpecWriter(output_file=output_file, max_size=max_size, max_annotations=max_ann, verbose=verbose, ignore_wrong_shape=True)

    if spec_config is None:
        spec_config = SpectrogramConfiguration(rate=sampling_rate, window_size=window_size, step_size=step_size,\
            bins_per_octave=bins_per_octave, window_function=None, low_frequency_cut=flow, high_frequency_cut=fhigh,\
            length=duration, overlap=overlap, type=['Mag', 'CQT'][cqt])

    # spectrogram provider
    provider = SpecProvider(path=input_dir, channel=channel, spec_config=spec_config, pad=pad)

    # subfolder unix structure
    files = provider.files
    subfolders = list()
    for f in files:
        sf = rel_path_unix(f, input_dir)
        subfolders.append(sf)

    # loop over files    
    num_files = len(files)
    for i in tqdm(range(num_files), disable = not progress_bar):

        # loop over segments
        for _ in range(provider.num_segs):

            # get next spectrogram
            spec = next(provider)

            # add annotations
            if areader is not None:
                labels, boxes = areader.get_annotations(spec.file_dict[0])
                spec.annotate(labels, boxes) 

            # save spectrogram(s) to file        
            path = subfolders[i] + 'spec'
            swriter.cd(path)
            swriter.write(spec)

    if swriter.num_ignored > 0:
        print('Ignored {0} spectrograms with wrong shape'.format(swriter.num_ignored))

    swriter.close()


class SpecWriter():
    """ Saves spectrograms to a database file (*.h5).

        If the combined size of the spectrograms exceeds max_size (1 GB by default), the output database 
        file will be split into several files, with _000, _001, etc, appended to the filename.

        Args:
            output_file: str
                Full path to output database file (*.h5)
            max_annotations: int
                Maximum number of annotations allowed for any spectrogram
            max_size: int
                Maximum size of output database file in bytes
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte)
            verbose: bool
                Print relevant information during execution such as files written to disk
            ignore_wrong_shape: bool
                Ignore spectrograms that do not have the same shape as previously saved spectrograms. Default is False.

        Attributes:
            base: str
                Output filename base
            ext: str
                Output filename extension (*.h5)
            file: tables.File
                Database file
            file_counter: int
                Keeps track of how many files have been written to disk
            spec_counter: int
                Keeps track of how many spectrograms have been written to files
            path: str
                Path to table within database filesystem
            name: str
                Name of table 
            max_annotations: int
                Maximum number of annotations allowed for any spectrogram
            max_file_size: int
                Maximum size of output database file in bytes
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte).
                Disabled if writing in 'append' mode.
            verbose: bool
                Print relevant information during execution such as files written to disk
            mode: str
                The mode to open the file. It can be one of the following:
                    ’r’: Read-only; no data can be modified.
                    ’w’: Write; a new file is created (an existing file with the same name would be deleted).
                    ’a’: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    ’r+’: It is similar to ‘a’, but the file must already exist.
            ignore_wrong_shape: bool
                Ignore spectrograms that do not have the same shape as previously saved spectrograms. Default is False.
            num_ignore: int
                Number of ignored spectrograms
            spec_shape: tuple
                Spectrogram shape

            Example:

                >>> # create a few cosine wave forms
                >>> from ketos.audio_processing.audio import Waveform
                >>> cos7 = Waveform.cosine(rate=1000, frequency=7.0, duration=1.0)
                >>> cos8 = Waveform.cosine(rate=1000, frequency=8.0, duration=1.0)
                >>> cos21 = Waveform.cosine(rate=1000, frequency=21.0, duration=1.0)
                >>> # compute spectrograms
                >>> from ketos.audio_processing.spectrogram import MagSpectrogram
                >>> s7 = MagSpectrogram(cos7, winlen=0.2, winstep=0.02)
                >>> s8 = MagSpectrogram(cos8, winlen=0.2, winstep=0.02)
                >>> s21 = MagSpectrogram(cos21, winlen=0.2, winstep=0.02)
                >>> # save the spectrograms to a database file
                >>> from ketos.data_handling.database_interface import SpecWriter
                >>> fname = "ketos/tests/assets/tmp/db_harm.h5"
                >>> writer = SpecWriter(output_file=fname)
                >>> writer.write(s7)
                >>> writer.write(s8)
                >>> writer.write(s21)
                >>> writer.close()
                3 spectrograms saved to ketos/tests/assets/tmp/db_harm.h5
                >>> # inspect the contacts of the database file
                >>> import tables
                >>> f = tables.open_file(fname, 'r')
                >>> print(f.root.spec)
                /spec (Table(3,), fletcher32, shuffle, zlib(1)) ''
    """
    def __init__(self, output_file, max_size=1E9, verbose=True, max_annotations=100, mode='w', ignore_wrong_shape=False):
        
        self.base = output_file[:output_file.rfind('.')]
        self.ext = output_file[output_file.rfind('.'):]
        self.file = None
        self.file_counter = 0
        self.max_annotations = max_annotations
        self.max_file_size = max_size
        self.path = '/'
        self.name = 'spec'
        self.verbose = verbose
        self.mode = mode
        self.ignore_wrong_shape = ignore_wrong_shape
        self.spec_counter = 0
        self.num_ignored = 0
        self.spec_shape = None

    def cd(self, fullpath='/'):
        """ Change the current directory within the database file system

            Args:
                fullpath: str
                    Full path to the table. For example, /data/spec
        """
        self.path = fullpath[:fullpath.rfind('/')+1]
        self.name = fullpath[fullpath.rfind('/')+1:]

    def write(self, spec, path=None, name=None):
        """ Write spectrogram to a table in the database file

            If path and name are not specified, the spectrogram will be 
            saved to the current directory (as set with the cd() method).

            Args:
                spec: Spectrogram
                    Spectrogram to be saved
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
        """
        if path is None:
            path = self.path
        if name is None:
            name = self.name

        # ensure a file is open
        self._open_file() 

        # open/create table
        tbl = self._open_table(path=path, name=name, shape=spec.image.shape) 

        if self.spec_counter == 0:
            self.spec_shape = spec.image.shape

        # write spectrogram to table
        if spec.image.shape == self.spec_shape or not self.ignore_wrong_shape:
            write_spec(tbl, spec)
            self.spec_counter += 1

            # close file if size reaches limit
            siz = self.file.get_filesize()
            if siz > self.max_file_size:
                self.close(final=False)

        else:
            self.num_ignored += 1

    def close(self, final=True):
        """ Close the currently open database file, if any

            Args:
                final: bool
                    If True, this instance of SpecWriter will not be able to save more spectrograms to file
        """        
        if self.file is not None:

            # TODO: loop over tables and apply table.flush() to each one before closing HDF5 file

            actual_fname = self.file.filename
            self.file.close()
            self.file = None

            if final and self.file_counter == 1:
                fname = self.base + self.ext
                os.rename(actual_fname, fname)
            else:
                fname = actual_fname

            if self.verbose:
                plural = ['', 's']
                print('{0} spectrogram{1} saved to {2}'.format(self.spec_counter, plural[self.spec_counter > 1], fname))

            self.spec_counter = 0

    def _open_table(self, path, name, shape):
        """ Open the specified table.

            If the table does not exist, create it.

            Args:
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
                shape: tuple
                    Shape of spectrogram image

            Returns:
                tbl: tables.Table
                    Table
        """        
        if path == '/':
            x = path + name
        elif path[-1] == '/':
            x = path + name
            path = path[:-1]
        else:
            x = path + '/' + name

        if x in self.file:
            tbl = self.file.get_node(path, name)
        else:
            tbl = create_table(h5file=self.file, path=path, name=name, shape=shape, max_annotations=self.max_annotations)

        return tbl

    def _open_file(self):
        """ Open a new database file, if none is open
        """                
        if self.file is None:
            if self.mode == 'a':
                fname = self.base + self.ext
            else:
                fname = self.base + '_{:03d}'.format(self.file_counter) + self.ext

            self.file = tables.open_file(fname, self.mode)
            self.file_counter += 1
