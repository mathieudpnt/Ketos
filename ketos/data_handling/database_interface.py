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

    This module provides functions to create and use HDF5 databases as storage for acoustic data 
    including metadata and annotations.

    An audio segment or spectrogram is said to be 'weakly annotated', if it is  assigned a single 
    (integer) label, and is said to be 'strongly annotated', if it is assigned one or several 
    labels, each accompanied by a start and end time, and potentially also a minimum and maximum 
    frequecy. 
"""

from email.mime import audio
import os
import warnings
import tables
import librosa
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from ketos.utils import tostring, ensure_dir
from ketos.audio.base_audio import BaseAudio
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram, MagSpectrogram, PowerSpectrogram, CQTSpectrogram, MelSpectrogram
import ketos.audio.audio_loader as al


def create_table_col(dtype, shape=()):
    """ Create column 

        If an invalid data type is given, None is returned

        Args:
            dtype: str
                Data type
            shape: tuple
                Data shape

        Returns:
            column
    """
    dtype = dtype.lower()

    if dtype == 'int': dtype = 'int32'
    if dtype == 'uint': dtype = 'uint32'
    if dtype == 'float': dtype = 'float64'

    if dtype == 'str': return tables.StringCol(shape)
    elif dtype == 'int8': return tables.Int8Col(shape)
    elif dtype == 'uint8': return tables.UInt8Col(shape)
    elif dtype == 'int32': return tables.Int32Col(shape)
    elif dtype == 'uint32': return tables.UInt32Col(shape)
    elif dtype == 'float32': return tables.Float32Col(shape)
    elif dtype == 'float64': return tables.Float64Col(shape)
    else:
        print(f'Warning: column type {dtype} not recognized. Column will not be created')
        return None

def open_file(path, mode, create_dir=True):
    """ Open an HDF5 database file.

        Wrapper function around tables.open_file: 
        https://www.pytables.org/usersguide/libref/top_level.html
        
        Args:
            path: str
                The file's full path.
            mode: str
                The mode to open the file. It can be one of the following:
                    * `r`: Read-only; no data can be modified.
                    * `w`: Write; a new file is created (an existing file with the same name would be deleted).
                    * `a`: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    * `r+`: It is similar to `a`, but the file must already exist.
            create_dir: bool
                If the directory does not exist, it will be automatically created. Default is True.
                Only applies if the mode is `w` or `a`, 

        Returns:
            : table.File object
                The h5file.
    """
    if mode in ['w', 'a'] and create_dir: ensure_dir(path)

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

def create_table(h5file, path, name, description, data_name='data', chunkshape=None, verbose=False):
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
            data_name: str or list(str)
                Name(s) of the table column(s) used to store the data array(s).          
            chunkshape: tuple
                The chunk shape to be used for compression

        Returns:
            table: table.Table object
                The created/open table.    

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, table_description, create_table
            >>> # Open a connection to the database
            >>> h5file = open_file("ketos/tests/assets/tmp/database1.h5", 'w')
            >>> # Create table descriptions for weakly labeled spectrograms with shape (32,64)
            >>> descr = table_description((32,64), include_label=False)
            >>> # Create 'table_data' within 'group1'
            >>> my_table = create_table(h5file, "/group1/", "table_data", descr) 
            >>> # Show the table description, with the field names (columns)
            >>> # and information about types and shapes
            >>> my_table
            /group1/table_data (Table(0,)fletcher32, shuffle, zlib(1)) ''
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
    if path.endswith('/') and path != '/':
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
        
        if not isinstance(data_name, list): data_name = [data_name]
        table.attrs.data_name = data_name

    return table

def table_description(data_shape, data_name=None, include_label=True, include_source=True, 
    attrs=None, filename_len=100, return_data_name=False):
    """ Description of table structure for storing audio signals or spectrograms.

        Args:
            data_shape: tuple (ints) or numpy array or :class:`audio.base_audio.BaseAudio` or list
                The shape of the waveform or spectrogram to be stored in the table. 
                If a numpy array is provided, the shape is deduced from this array.
                If an instance of BaseAudio is provided, the shape is deduced from 
                the data attribute. It is also possible to specify a list of data shapes, 
                in which case the table will have multiple data columns.
            data_name: str or list(str) 
                Name(s) of the data columns. If None is specified, the data column is named 'data', 
                or 'data0', 'data1', ... if the table contains multiple data columns.
            include_label: bool
                Include integer label column. Default is True.
            include_source: bool
                If True, the name of the wav file from which the audio signal or 
                spectrogram was generated and the placement within that file, is 
                saved to the table. Default is True.
            attrs: list()
                Specify additional attributes that you want saved to the table. 
                For each attribute, provide the name, shape, and type, in the form 
                of a dictionary, e.g., {'name':'confidence', 'shape':() 'type':'float'} 
            filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.
            return_data_name: bool
                Return the names of the columns used to store the data arrays. 

        Returns:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure.
            data_name: list(str)
                The names of the columns used to store the data arrays.
                Only returned if return_data_name=True.

        Examples:
            >>> import numpy as np
            >>> from ketos.data_handling.database_interface import table_description
            >>> 
            >>> #Create a 64 x 20 image
            >>> spec = np.random.random_sample((64,20))
            >>>
            >>> #Create a table description for weakly labeled spectrograms of this shape
            >>> descr = table_description(spec)
            >>>
            >>> #Inspect the table structure
            >>> cols = descr.columns
            >>> for key in sorted(cols.keys()):
            ...     print("%s: %s" % (key, cols[key]))
            data: Float32Col(shape=(64, 20), dflt=0.0, pos=None)
            filename: StringCol(itemsize=100, shape=(), dflt=b'', pos=None)
            id: UInt32Col(shape=(), dflt=0, pos=None)
            label: UInt8Col(shape=(), dflt=0, pos=None)
            offset: Float64Col(shape=(), dflt=0.0, pos=None)
            >>>
            >>> #Create a table description for strong annotations
            >>> descr_annot =  table_description_annot()
            >>>
            >>> #Inspect the annotation table structure
            >>> cols = descr_annot.columns
            >>> for key in sorted(cols.keys()):
            ...     print("%s: %s" % (key, cols[key]))
            data_index: UInt32Col(shape=(), dflt=0, pos=None)
            end: Float64Col(shape=(), dflt=0.0, pos=None)
            label: UInt8Col(shape=(), dflt=0, pos=None)
            start: Float64Col(shape=(), dflt=0.0, pos=None)
    """
    if attrs is None: attrs = []

    if isinstance(data_shape, list): data_shape_list = data_shape
    else: data_shape_list = [data_shape]

    # set/create name list
    if data_name is None:
        data_name_list = [f'data{i}' for i in range(len(data_shape_list))]
        if len(data_name_list) == 1:
            data_name_list = ['data']
    
    elif isinstance(data_name, list): data_name_list = data_name
    else: data_name_list = [data_name]

    assert len(data_shape_list) == len(data_name_list), f'data_shape and data_name have mismatched lengths ({len(data_shape_list)} and {len(data_name_list)})'

    # deduce shape from class objects
    _data_shape_list = []
    for i in range(len(data_shape_list)):
        ds = data_shape_list[i]
        if isinstance(ds, np.ndarray): _data_shape_list.append(ds.shape)
        elif isinstance(ds, BaseAudio): _data_shape_list.append(ds.data.shape)
        elif isinstance(ds, tuple): _data_shape_list.append(ds)

    class TableDescription(tables.IsDescription):
        id = create_table_col('uint32')
        
        for ds,dn in zip(_data_shape_list, data_name_list):
            vars()[dn] = create_table_col('float32', ds) #data columns
    
        del ds, dn #delete loop variables or else they will be interpreted as table columns

        if include_source:
            filename = create_table_col('str', filename_len)
            offset = create_table_col('float64')

        if include_label:
            label = create_table_col('uint8')

        for attr in attrs:
            vars()[attr['name']] = create_table_col(attr['type'], attr['shape'])

        try:
            del attr #delete loop variables or else they will be interpreted as table columns
        except NameError:
            pass

    if return_data_name:
        return TableDescription, data_name_list
    else:
        return TableDescription


def table_description_annot(freq_range=False):
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
        data_index = tables.UInt32Col()
        label = tables.UInt8Col()
        start = tables.Float64Col()
        end = tables.Float64Col()
        if freq_range:
            freq_min = tables.Float32Col()
            freq_max = tables.Float32Col()

    return TableDescription

def write_repres_attrs(table, x):
    """ Writes the audio representation attributes into the HDF5 database, 
        where they become stored as table attributes.

        The audio representation attributes include,

            * Time resolution in seconds (time_res)
            * Minimum frequency in Hz (freq_min)
            * Spectrogram type (type)
            * Frequency resolution in Hz (freq_res) or, in the case of
              CQT spectrograms, the number of bins per octave (bins_per_octave).

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored
                (described by spec_description()).
            x: instance of :class:`spectrogram.MagSpectrogram`, \
                :class:`spectrogram.PowerSpectrogram`, :class:`spectrogram.MelSpectrogram`, \
                :class:`spectrogram.CQTSpectrogram`, numpy.array    
                The audio object to be stored in the table.

        Returns:
            None.
    """
    if not isinstance(x, list): x = [x]

    attrs = []
    for xx in x:
        if isinstance(xx, BaseAudio): attrs.append(xx.get_repres_attrs())
        elif isinstance(xx, np.ndarray): attrs.append({'type': 'numpy.ndarray'})
        else:
            print('Warning: Unknown format. Data could not be written to disk.')
            return

    if len(attrs) == 1: attrs = attrs[0]

    table.attrs.audio_repres = attrs

def write_annot(table, data_index, annots):
    """ Write annotations to a HDF5 table.

        Args:
            table: tables.Table
                Table in which the annotations will be stored.
                (described by table_description()).
            data_index: int
                Audio object unique identifier.
            annots: pandas DataFrame
                Annotations

        Returns:
            None.
    """
    write_freq = ("freq_min" in table.colnames)
    for idx,annot in annots.iterrows():
        row = table.row
        row["data_index"] = data_index
        row["label"] = annot['label']
        row["start"] = annot['start']
        row["end"]   = annot['end']
        if write_freq:
            row["freq_min"] = annot['freq_min']
            row["freq_max"] = annot['freq_max']

        row.append()
        table.flush()

def write_audio(table, data, attrs={}, id=None):
    """ Write an audio object, typically a waveform or spectrogram, to a HDF5 table.

        Args:
            table: tables.Table
                Table in which the audio data will be stored.
                (described by table_description()).
            data: numpy.array or list(numpy.array)
                Audio data array(s). The number of data arrays must match 
                the number of data columns in the table.
            attrs: dict()
                Instance attributes
            filename: str
                Filename
            offset: float
                Offset with respect to beginning of file in seconds.
            label: int
                Integer valued label. Optional
            id: int
                Unique identifier. Optional

        Returns:
            index: int
                Index of row that the audio object was saved to.
    """
    row = table.row
    index = table.nrows

    if id is None: id = index

    row['id'] = id  #pass id to table

    if not isinstance(data, list): data = [data]

    assert len(data) == len(table.attrs.data_name), f'mismatch between lengths of data and table.attrs.data_name ({len(data)} and {len(table.attrs.data_name)})'
    
    for d,n in zip(data, table.attrs.data_name): row[n] = d #pass data array(s) to table

    for key,value in attrs.items(): #loop over instance attributes
        if key in table.colnames: #check that table has appropriate column
            if value is not None: 
                if isinstance(value, (dt.datetime,np.datetime64)):
                    if pd.isnull(value):
                        value = ""
                    else:
                        if isinstance(value, np.datetime64):
                            value = np.datetime_as_string(value)
                        else:
                            value = value.strftime("%Y%m%d_%H:%M:%S") #convert datetime object to string

                row[key] = value #write value to appropriate field
    
    row.append()
    table.flush()

    return index

def write(x, table, table_annot=None, id=None):
    """ Write waveform or spectrogram and annotations to HDF5 tables.

        Note: If the id argument is not specified, the row number will 
        will be used as a unique identifier for the spectrogram.

        When a list of audio objects is provided, only the instance
        attributes (filename, offset, label, annotations, etc.) of the first 
        object is written to the table.

        Args:
            x: instance of :class:`audio.waveform.Waveform`,
                :class:`audio.spectrogram.MagSpectrogram`, 
                :class:`audio.spectrogram.PowerSpectrogram`,
                :class:`audio.spectrogram.MelSpectrogram`, 
                :class:`audio.spectrogram.CQTSpectrogram`,
                numpy.ndarray 
                The audio object to be stored in the table.
                It is also possible to specify a list of audio objects.
                The number of objects must match the number of data columns in the table.
            table: tables.Table
                Table in which the audio data will be stored.
                (described by table_description()).
            table_annot: tables.Table
                Table in which the annotations will be stored.
                (described by table_description_weak_annot() or table_description_strong_annot()).
            id: int
                Audio object unique identifier. Optional.

        Returns:
            None.

        Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, create_table, table_description, table_description_annot, write
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> from ketos.audio.waveform import Waveform
            >>>
            >>> # Create an Waveform object from a .wav file
            >>> audio = Waveform.from_wav('ketos/tests/assets/2min.wav')
            >>> # Use that signal to create a spectrogram
            >>> spec = MagSpectrogram.from_waveform(audio, window=0.2, step=0.05)
            >>> # Add a single annotation
            >>> spec.annotate(label=1, start=0., end=2.)
            >>>
            >>> # Open a connection to a new HDF5 database file
            >>> h5file = open_file("ketos/tests/assets/tmp/database2.h5", 'w')
            >>> # Create table descriptions for storing the spectrogram data
            >>> descr_data = table_description(spec)
            >>> descr_annot = table_description_annot()
            >>> # Create tables
            >>> tbl_data = create_table(h5file, "/group1/", "table_data", descr_data) 
            >>> tbl_annot = create_table(h5file, "/group1/", "table_annot", descr_annot) 
            >>> # Write spectrogram and its annotation to the tables
            >>> write(spec, tbl_data, tbl_annot)
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
    if not isinstance(x, list): x = [x]

    if table.nrows == 0: write_repres_attrs(table, x)

    data, attrs = [], []
    for xx in x:
        if isinstance(xx, BaseAudio): 
            data.append(xx.get_data())
            attrs.append(xx.get_instance_attrs())
        elif isinstance(xx, np.ndarray): 
            data.append(xx)
            attrs.append({})
        else:
            print('Warning: Unknown format. Data could not be written to disk.')
            return

    data_index = write_audio(table=table, data=data, attrs=attrs[0], id=id)

    if table_annot is not None:
        annots = x[0].get_annotations()
        if annots is not None:
            write_annot(table=table_annot, data_index=data_index, annots=annots)

def filter_by_label(table, label):
    """ Find all audio objects in the table with the specified label.

        Args:
            table: tables.Table
                The table containing the annotations
            label: int or list of ints
                The labels to be searched
        Raises:
            TypeError: if label is not an int or list of ints.

        Returns:
            indices: list(int)
                Indices of the audio objects with the specified label(s).
                If there are no objects that match the label, returs an empty list.

        Examples:
            >>> from ketos.data_handling.database_interface import open_file, open_table
            >>>
            >>> # Open a database and an existing table
            >>> h5file = open_file("ketos/tests/assets/11x_same_spec.h5", 'r')
            >>> table = open_table(h5file, "/group_1/table_annot")
            >>>
            >>> # Retrieve the indices for all spectrograms that contain the label 1
            >>> # (all spectrograms in this table)
            >>> filter_by_label(table, 2)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>>
            >>> # Since none of the spectrograms in the table include the label 4, 
            >>> # an empty list is returned
            >>> filter_by_label(table, 4)
            []
            >>> h5file.close()
    """
    if isinstance(label, (list)):
        if not all (isinstance(l, int) for l in label):
            raise TypeError("label must be an int or a list of ints")    
    elif isinstance(label, int):
        label = [label]
    else:
        raise TypeError("label must be an int or a list of ints")    
    
    col_name = 'data_index' if 'data_index' in table.colnames else 'id'

    indices = []
    for index,row in enumerate(table.iterrows()):
        if row['label'] in label:
            if col_name == 'data_index': indices.append(row[col_name])
            else: indices.append(index)

    indices = np.unique(indices).tolist()    
    return indices

def load_audio(table, indices=None, table_annot=None, stack=False):
    """ Retrieve all the audio objects in a table or a subset specified by the index_list

        Warnings: Loading all objects in a table might cause memory problems.

        Args:
            table: tables.Table
                The table containing the audio objects
            indices: list of ints or None
                A list with the indices of the audio objects that will be retrieved.
                If set to None, loads all objects in the table.
            table_annot: tables.Table
                The table containing the annotations. If no such table is provided, 
                the audio objects are still loaded, but without annotations.
            stack: bool
                Stack the audio objects into a single object

        Returns:
            audio_objs: list or instance of Waveform, MagSpectrogram, PowerSpectrogram, MelSpectrogram, CQTSpectrogram
                Audio objects, or numpy array

        Examples:
            >>> from ketos.data_handling.database_interface import open_file, open_table, load_audio
            >>> # Open a connection to the database.
            >>> h5file = open_file("ketos/tests/assets/11x_same_spec.h5", 'r')
            >>> # Open the tables in group_1
            >>> tbl_data = open_table(h5file,"/group_1/table_data")
            >>> tbl_annot = open_table(h5file,"/group_1/table_annot")    
            >>> # Load the spectrograms stored on rows 0, 3 and 10, including their annotations
            >>> selected_specs = load_audio(table=tbl_data, table_annot=tbl_annot, indices=[0,3,10])
            >>> # The resulting list has the 3 spectrogram objects
            >>> len(selected_specs)
            3
            >>> type(selected_specs[0])
            <class 'ketos.audio.spectrogram.MagSpectrogram'>
            >>>
            >>> h5file.close()
    """
    res = list()
    if indices is None: indices = list(range(table.nrows))

    # keyword arguments needed for initializing object
    if 'audio_repres' in table._v_attrs._f_list():
        data_name = table.attrs.data_name
        kwargs = table.attrs.audio_repres
    else: #include this option for backward compatability (OK 2021-01-12)
        data_name = ['data']
        kwargs = {}
        for name in table._v_attrs._f_list():
            kwargs[name] = table._v_attrs[name]

    if not isinstance(kwargs, list): kwargs = [kwargs]

    # get the names of all columns except the data column(s) and the id column
    col_names = table.colnames.copy()
    col_names.remove('id')
    for dn in data_name: col_names.remove(dn)

    # loop over items in table
    audio_objs = []
    for idx in indices:
        #current item
        it = table[idx] 

        # annotations, if any
        if table_annot is None: 
            annot = None
        else: 
            annot_data = table_annot.read_where("""data_index == {0}""".format(idx))
            if len(annot_data) > 0:
                annot = pd.DataFrame()
                for col_name in ['label','start','end','freq_min','freq_max']:
                    if col_name in table_annot.colnames: annot[col_name] = annot_data[col_name] 

        obj = []
        for kwa, dn in zip(kwargs, data_name):
            for col_name in col_names:
                val = it[col_name]
                if table.coltypes[col_name] == 'string': val = val.decode()
                kwa[col_name] = val

            # initialize audio object
            if kwa['type'] == 'numpy.ndarray':
                obj.append(it[dn])
            else:
                audio_class = al.audio_repres_dict[kwa['type']]
                obj.append(audio_class(data=it[dn], annot=annot, **kwa))

        if len(obj) == 1: obj = obj[0]

        audio_objs.append(obj)

    if stack:
        audio_objs = audio_class.stack(audio_objs)

    return audio_objs


def _infer_unique_labels(selections):
    ''' Helper function for the `create_database` function.
    
        Attempts to infer unique labels from the selection table. 

        Args:
            selections: pandas DataFrame
                Selection table

        Returns:
            : numpy array
                List of unique labels. None, if the labels could not be inferred.
    '''
    if 'label' in selections.columns.values:
        return np.sort(pd.unique(selections.label.ravel('K')))
    
    else:
        warnings.warn("Labels could not be inferred because the selection table "\
            "does not contain a column named 'label'. The database will be created "\
            "without labels. Note that this may cause issues if the database is supposed "\
            "to be used as a training or test set for machine learning.")
        
        return None


def _is_nested_dict(x):
    """ Helper function for the `create_database` function.
    
        Determines if the input is a nested dictionary.    

        Args:
            x: dict
                Input
        
        Returns:
            : bool
                True if `x` is a nested dictionary.
    """
    return isinstance(x, dict) and isinstance(list(x.values())[0], dict)


def create_database(output_file, data_dir, selections, channel=0, 
    audio_repres={'type': 'Waveform'}, annotations=None, unique_labels=None, 
    dataset_name=None, table_name='data', max_size=None, verbose=True, progress_bar=True, 
    discard_wrong_shape=False, allow_resizing=1, include_source=True, 
    include_label=True, include_attrs=False, attrs=None, 
    index_cols=None, mode='a', create_dir=True):
    """ Create a database from a selection table.

        Note that all selections must have the same duration. This is necessary to ensure 
        that all the objects stored in the database have the same dimension.

        If each entry in the selection table can have multiple annotations, these can be 
        specified with the 'annotations' argument. On the other hand, if each entry in 
        the selection table is chacterized by a single, integer label, these should be 
        included as a column named 'label' in the selection table.

        If 'dataset_name' is not specified, the name of the folder containing the audio 
        files ('data_dir') will be used.
        
        Warnings will be printed if the method encounters problems loading/writing audio 
        data or if the start/end time of a selection is outside the range of the audio 
        file. The warnings can be suppressed by setting `verbose=False`.
    
        Args:
            output_file:str
                The name of the HDF5 file in which the data will be stored.
                Can include the path (e.g.:'/home/user/data/database_abc.h5').
                If the file does not exist, it will be created.
                If the file already exists, new data will be appended to it.
            data_dir:str
                Path to folder containing \*.wav files.
            selections: pandas DataFrame or list
                Selection table. 
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            audio_repres: dict or list
                A dictionary containing the parameters used to generate the spectrogram or waveform
                segments. See :class:~ketos.audio.auio_loader.AudioLoader for details on the 
                required and optional fields for each type of signal. 
                It is also possible to specify one or several audio representations as a nested 
                dictionary, in which case the dictionary keys are used as column names in the output table.
            annotations: pandas DataFrame
                Annotation table. Optional.
            unique_labels: list(int)
                List of labels occurring in the dataset. If not specified, the labels will be inferred 
                from the selections or the annotations.
            dataset_name:str
                Name of the node (HDF5 group) within the database (e.g.: 'train')
                Under this node, two tables will be created, 'data' and 'data_annot',
                containing the data samples (spectrograms and/or waveforms) and the 
                annotations associated with each sample, respectively.         
            table_name: str
                Table name. Default is 'data'.               
            max_size: int
                Maximum size of output database file in bytes.
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte). 
                If None, no restriction is imposed on the file size (i.e. the file 
                is never split).
            verbose: bool
                Print relevant information during execution such as no. of files written to disk
            progress_bar: bool
                Show progress bar.  
            discard_wrong_shape: bool
                Discard objects that do not have the same shape as previously saved objects. Default is False.
            allow_resizing: int
                If the object shape differs from previously saved objects, the object 
                will be resized using the resize method of the scikit-image package, provided the mismatch 
                is no greater than allow_resizing in either dimension. 
            include_source: bool
                If True, the name of the wav file from which the waveform or 
                spectrogram was generated and the offset within that file, is 
                saved to the table. Default is True.
            include_label: bool
                Include integer label column in data table. Default is True.
            include_attrs: bool
                If True, load data from attribute columns in the selection table. Default is False.
            attrs: list(str)
                Specify the names of the attribute columns that you wish to load data from. 
                Overwrites include_attrs if specified. If None, all columns will be loaded provided that 
                include_attrs=True.
            index_cols: str og list(str)
                Create indices for the specified columns in the data table to allow for faster queries.
                For example, `index_cols="filename"` or `index_cols=["filename", "label"]`
            mode: str
                The mode to open the file. It can be one of the following:
                    `w`: Write; a new file is created (an existing file with the same name would be deleted). 
                    `a`: Append; an existing file is opened for reading and writing, and if the file does not 
                        exist it is created. This is the default.
                    `r+`: It is similar to `a`, but the file must already exist.
            create_dir: bool
                If the output directory does not exist, it will be automatically created. Default is True.
                Only applies if the mode is `w` or `a`, 
    """
    # attempt to automatically infer unique labels
    if unique_labels is None:
        unique_labels = _infer_unique_labels(selections)

    # if audio_repres is a nested dictionary, use the keys as names for the data fields
    if _is_nested_dict(audio_repres):
        data_name = [key for key in audio_repres.keys()]
    else:
        data_name = None

    # initialize an audio loader
    loader = al.AudioSelectionLoader(path=data_dir, selections=selections, channel=channel, 
        repres=audio_repres, annotations=annotations, include_attrs=include_attrs, attrs=attrs)

    # if the group path is not specified, use the name of the data directory
    if dataset_name is None: dataset_name = os.path.basename(data_dir)
    path_to_dataset = dataset_name if dataset_name.startswith('/') else '/' + dataset_name
    
    # initialize a writer
    annot_type = "weak" if annotations is None else "strong"
    writer = AudioWriter(output_file=output_file, max_size=max_size, verbose=verbose, mode=mode,
        discard_wrong_shape=discard_wrong_shape, allow_resizing=allow_resizing, 
        include_source=include_source, include_label=include_label, data_name=data_name, 
        index_cols=index_cols, create_dir=create_dir, table_path=path_to_dataset, table_name=table_name,
        annot_type=annot_type)
    
    # loop over all selections and write to database
    for i in tqdm(range(loader.num()), disable = not progress_bar):
        x = next(loader)
        try:
            writer.write(x)
        except Exception as e:
            warnings.warn(f"While writing entry {loader.counter}, Message: {str(e)}", category=UserWarning)

    if unique_labels is not None:
        writer.write_attr("unique_labels", unique_labels)

    writer.close()


class AudioWriter():
    """ Saves waveform or spectrogram objects to a database file (\*.h5).

        If the combined size of the saved data exceeds max_size (1 GB by default), the output database 
        file will be split into several files, with _000, _001, etc, appended to the filename.

        Args:
            output_file: str
                Full path to output database file (\*.h5)
            max_size: int
                Maximum size of output database file in bytes.
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte). 
                If None, no restriction is imposed on the file size (i.e. the file 
                is never split).
            verbose: bool
                Print relevant information during execution such as no. of files written to disk
            mode: str
                The mode to open the file. It can be one of the following:
                    `w`: Write; a new file is created (an existing file with the same name would be deleted). This is the default.
                    `a`: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    `r+`: It is similar to `a`, but the file must already exist.
            discard_wrong_shape: bool
                Discard objects that do not have the same shape as previously saved objects. Default is False.
            allow_resizing: int
                If the object shape differs from previously saved objects, the object 
                will be resized using the resize method of the scikit-image package, provided the mismatch 
                is no greater than allow_resizing in either dimension. 
            include_source: bool
                If True, the name of the wav file from which the waveform or 
                spectrogram was generated and the offset within that file, is 
                saved to the table. Default is True.
            include_label: bool
                Include integer label column in data table. Default is True.
            include_attrs: bool
                If True, attributes returned by the `get_instance_attrs()` method will also be saved 
                to the table. Default is True.
            max_filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.
            data_name: str or list(str) 
                Name(s) of the data columns. If None is specified, the data column is named 'data', 
                or 'data0', 'data1', ... if the table contains multiple data columns.
            create_dir: bool
                If the output directory does not exist, it will be automatically created. Default is True.
                Only applies if the mode is `w` or `a`, 
            annot_type: str
                Specify the annotation type. Options are `weak` and `strong`. If not specified, the type will be 
                inferred from the first instance to be written to the database file. For weakly labelled data, a 
                extra column named `label` is include in the data table. For strongly labelled data, the annotations 
                are saved to a separate table.
            table_path: str
                Path to the group containing the table
            table_name: str
                Name of the table

        Attributes:
            base: str
                Output filename base
            ext: str
                Output filename extension (\*.h5)
            file: tables.File
                Database file
            file_counter: int
                Keeps track of how many files have been written to disk
            item_counter: int
                Keeps track of how many audio objects have been written to files
            path: str
                Path to table within database filesystem
            name: str
                Name of table 
            max_size: int
                Maximum size of output database file in bytes
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte).
                Disabled if writing in 'append' mode.
            verbose: bool
                Print relevant information during execution such as files written to disk
            mode: str
                The mode to open the file. It can be one of the following:
                    `w`: Write; a new file is created (an existing file with the same name would be deleted).
                    `a`: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    `r+`: It is similar to `a`, but the file must already exist.
            discard_wrong_shape: bool
                Discard objects that do not have the same shape as previously saved objects. Default is False.
            allow_resizing: int
                If the object shape differs from previously saved objects, the object 
                will be resized using the resize method of the scikit-image package, provided the mismatch 
                is no greater than allow_resizing in either dimension. 
            num_ignore: int
                Number of ignored objects
            data_shape: tuple
                Data shape
            include_source: bool
                If True, the name of the wav file from which the waveform or 
                spectrogram was generated and the offset within that file, is 
                saved to the table. Default is True.
            include_label: bool
                Include integer label column in data table. Default is True.
            include_attrs: bool
                If True, attributes returned by the `get_instance_attrs()` method will also be saved 
                to the table. Default is True.
            filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.
            data_name: str or list(str) 
                Name(s) of the data columns. If None is specified, the data column is named 'data', 
                or 'data0', 'data1', ... if the table contains multiple data columns.
            index_cols: str og list(str)
                Create indices for the specified columns in the data table to allow for faster queries.
                For example, `index_cols="filename"` or `index_cols=["filename", "label"]`
            create_dir: bool
                If the output directory does not exist, it will be automatically created. Default is True.
                Only applies if the mode is `w` or `a`, 
            annot_type: str
                Annotation type. Options are `weak` and `strong`. If not specified, the type will be 
                inferred from the first instance to be written to the database file. For strongly labelled 
                data, the annotations are saved to a separate table.
    """
    def __init__(self, output_file, max_size=1E9, verbose=False, mode='w', discard_wrong_shape=False,
        allow_resizing=1, include_source=True, include_label=True, include_attrs=True, 
        max_filename_len=100, data_name=None, index_cols=None, create_dir=True, annot_type=None,
        table_path='/', table_name='audio'):
        
        self.base = output_file[:output_file.rfind('.')]
        self.ext = output_file[output_file.rfind('.'):]
        self.file = None
        self.file_counter = 0
        self.max_size = max_size
        self.path = table_path
        self.name = table_name
        self.verbose = verbose
        self.mode = mode
        self.item_counter = 0
        self.num_discarded = 0
        self.num_resized = 0
        self.data_shape = None
        self.discard_wrong_shape = discard_wrong_shape
        self.allow_resizing = allow_resizing
        self.include_source = include_source
        self.include_label = include_label
        self.include_attrs = include_attrs
        self.filename_len = max_filename_len
        self.data_name = data_name
        self.create_dir = create_dir
        self.annot_type = annot_type
        if index_cols is None:
            self.index_cols = []
        elif isinstance(index_cols, str):
            self.index_cols = [index_cols]
        else:   
            self.index_cols = index_cols
            
    def set_table(self, path, name):
        """ Change the current table

            Args:
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
        """
        self.path = path
        self.name = name

    def write_attr(self, attr_name, attr_value, path=None, name=None):
        """ Write attribute to a table in the database file

            If path and name are not specified, the object will be 
            saved to the current directory (as set with the cd() method).

            See https://www.pytables.org/usersguide/libref/declarative_classes.html#the-attributeset-class 
            for details on how various Python types are saved as attributes 
            to HDF5 tables.

            Args:
                attr_name: str
                    Attribute name
                attr_value: 
                    Value to be saved
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
        """
        if path is None: path = self.path
        if name is None: name = self.name
        self.set_table(path, name)
        self._open_file() 
        tbl_dict = self._open_tables(path=path, name=name)
        for _,tbl in tbl_dict.items():
            setattr(tbl.attrs, attr_name, attr_value)


    def write(self, x, path=None, name=None):
        """ Write waveform or spectrogram object to a table in the database file

            If path and name are not specified, the object will be 
            saved to the current directory (as set with the cd() method).

            Args:
                x: instance of BaseAudio or list
                    Object(s) to be saved
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
        """
        if not isinstance(x, list): x = [x]

        if path is None: path = self.path
        if name is None: name = self.name
        self.set_table(path, name)

        # ensure a file is open
        self._open_file() 

        # record shape of first audio object
        if self.item_counter == 0: self._save_shape(x)

        # open tables, create if they do not already exist
        tbl_dict = self._open_tables(path=path, name=name, x=x) 

        # resize, if needed and allowed
        do_write = 1
        for i in range(len(x)):
            shape_diff = np.abs(np.subtract(x[i].data.shape, self.data_shape[i]))
            if np.sum(shape_diff) > 0 and np.all(shape_diff <= self.allow_resizing): 
                x[i].data = resize(x[i].data, self.data_shape[i], anti_aliasing=True)
                self.num_resized += 1

            if x[i].data.shape != self.data_shape[i] and self.discard_wrong_shape: do_write = False

        # write spectrogram to table
        if do_write:
            write(x=x, **tbl_dict)
            self.item_counter += 1

            # close file if size reaches limit
            siz = self.file.get_filesize()
            if self.max_size is not None and siz > self.max_size:
                self.close(final=False)

        else:
            self.num_discarded += 1

    def _save_shape(self, x):
        """ Record the shape of the data

            Args:
                x: list of BaseAudio objects and numpy arrays
                    Object(s) to be saved
        """
        self.data_shape = []
        for xx in x:
            if isinstance(xx, BaseAudio): self.data_shape.append(xx.get_data().shape)
            elif isinstance(xx, np.ndarray): self.data_shape.append(xx.shape)

    def close(self, final=True):
        """ Close the currently open database file, if any

            Args:
                final: bool
                    If True, this instance of AudioWriter will not be able to save more spectrograms to file
        """        
        if self.file is not None:

            actual_fname = self.file.filename

            # create index for data_index column in annotation 
            # table to allow faster queries
            # https://www.pytables.org/usersguide/optimization.html
            tbl_dict = self._open_tables(path=self.path, name=self.name)
            if 'table_annot' in tbl_dict.keys(): tbl_dict['table_annot'].cols.data_index.create_index()

            # create user-specified indices
            for index_col in self.index_cols:
                if index_col in tbl_dict['table'].cols._v_colnames:
                    tbl_dict['table'].cols._f_col(index_col).create_index()
                else:
                    if self.verbose:
                        print(' Warning: Attempting to build index for column {index_col} which does not exist. Ignore.')

            self.file.close()
            self.file = None

            if final and self.file_counter == 1:
                fname = self.base + self.ext
                os.rename(actual_fname, fname)
            else:
                fname = actual_fname

            if self.verbose:
                plural = ['', 's']
                print('{0} item{1} saved to {2}'.format(self.item_counter, plural[self.item_counter > 1], fname))
                if self.num_discarded > 0: print('Discarded {0} objects due to shape mismatch'.format(self.num_discarded))
                if self.num_resized > 0: print('Resized {0} objects due to shape mismatch'.format(self.num_resized))

            self.item_counter = 0

    def _open_tables(self, path, name, x=None):
        """ Open the specified table.

            If the table does not exist, create it.
            (This requires that x is specified)

            Args:
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
                x: list of BaseAudio objects and numpy arrays
                    Object(s) to be saved

            Returns:
                tbl_dict: dict
                    Data and annotation tables in a dictionary
        """        
        if path == '/':
            fullpath = path + name
        elif path[-1] == '/':
            fullpath = path + name
            path = path[:-1]
        else:
            fullpath = path + '/' + name

        if fullpath in self.file:
            tbl_dict = {'table': self.file.get_node(path, name)}
            if fullpath+'_annot' in self.file: 
                tbl_dict['table_annot'] = self.file.get_node(path, name+'_annot')

        elif x is not None:
            if self.annot_type is None:
                annot_type, freq_range = self._detect_annot_type(x)
            else:
                annot_type = self.annot_type
                freq_range = (annot_type == "strong")
            
            if self.include_attrs: 
                attrs = self._create_attr_list(x)
            else: 
                attrs = []
            
            descr, self.data_name = table_description(data_shape=x, 
                                      include_label=self.include_label, 
                                      include_source=self.include_source,
                                      attrs=attrs, 
                                      filename_len=self.filename_len,
                                      data_name=self.data_name,
                                      return_data_name=True)

            tbl = create_table(h5file=self.file, path=path, name=name, description=descr, data_name=self.data_name)
            tbl_dict = {'table': tbl}

            if annot_type == 'strong': 
                descr_annot = table_description_annot(freq_range=freq_range)
                tbl_annot = create_table(h5file=self.file, path=path, name=name+'_annot', description=descr_annot)
                tbl_dict['table_annot'] = tbl_annot

        else:
            tbl_dict = None

        return tbl_dict

    def _open_file(self):
        """ Open a new database file, if none is open
        """            
        if self.file is None:
            if self.mode == 'a':
                fname = self.base + self.ext
            else:
                fname = self.base + '_{:03d}'.format(self.file_counter) + self.ext

            self.file = open_file(path=fname, mode=self.mode, create_dir=self.create_dir)

            self.file_counter += 1

    def _detect_annot_type(self, x):
        """ Detect the annotation type (weak or strong)
        """                
        if x[0].get_annotations() is None: 
            annot_type = 'weak'
            freq_range = False
        else:
            annot_type = 'strong'
            freq_range = ('freq_min' in x[0].get_annotations().columns)

        return annot_type, freq_range

    def _create_attr_list(self, x):
        """ Create list of attributes to be included in table description
        """
        attrs = x[0].get_instance_attrs()
        attr_list = []
        for key,value in attrs.items():
            if key in ['filename','offset','label']: continue
            attr = {}
            attr['name'] = key
            if isinstance(value, np.ndarray):
                attr['shape'] = value.shape
                attr['type'] = str(value.dtype)
            elif isinstance(value, (str,dt.datetime,np.datetime64)):
                attr['shape'] = self.filename_len
                attr['type'] = 'str'
            elif isinstance(value, float):
                attr['shape'] = ()
                attr['type'] = 'float'
            elif isinstance(value, (int,np.int32,np.int64)):
                attr['shape'] = ()
                attr['type'] = 'int'
            else:
                warnings.warn(f'Warning: attribute {key} with type {type(value)} cannot be saved to table', UserWarning)
                continue

            attr_list.append(attr)
        
        return attr_list