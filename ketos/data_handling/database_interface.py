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

import os
import tables
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from ketos.utils import tostring
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram, MagSpectrogram, PowerSpectrogram, CQTSpectrogram, MelSpectrogram
import ketos.audio.audio_loader as al


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

    return table

def table_description(data_shape, include_label=True, include_source=True, filename_len=100):
    """ Description of table structure for storing audio signals or spectrograms.

        Args:
            data_shape: tuple (ints) or numpy array or :class:`audio.base_audio.BaseAudio' 
                The shape of the waveform or spectrogram to be stored in the table. 
                If a numpy array is provided, the shape is deduced from this array.
                If an instance of BaseAudio is provided, the shape is deduced from 
                the data attribute.
            include_label: bool
                Include integer label column. Default is True.
            include_source: bool
                If True, the name of the wav file from which the audio signal or 
                spectrogram was generated and the placement within that file, is 
                saved to the table. Default is True.
            filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.

        Returns:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure.

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
    if isinstance(data_shape, np.ndarray): data_shape = data_shape.shape
    elif isinstance(data_shape, Spectrogram): data_shape = data_shape.data.shape

    class TableDescription(tables.IsDescription):
        id = tables.UInt32Col()
        data = tables.Float32Col(data_shape)
        if include_source:
            filename = tables.StringCol(filename_len)
            offset = tables.Float64Col()
        if include_label:
            label = tables.UInt8Col()

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

def write_attrs(table, x):
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

        Returns:
            None.
    """
    for key, value in x.get_attrs().items():
        table.attrs.tmp = value
        table.attrs._f_rename('tmp',key)

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

def write_audio(table, data, filename=None, offset=0, label=None, id=None):
    """ Write waveform or spectrogram to a HDF5 table.

        Args:
            table: tables.Table
                Table in which the spectrogram will be stored.
                (described by table_description()).
            data: numpy array
                Waveform or spectrogram data array 
            filename: str
                Filename
            offset: float
                Offset with respect to beginning of file in seconds.
            label: int
                Integer valued label. Optional
            id: int
                Spectrogram unique identifier. Optional

        Returns:
            index: int
                Index of row that the audio object was saved to.
    """
    write_source = ("filename" in table.colnames)
    write_label  = ("label" in table.colnames)

    row = table.row
    index = table.nrows

    if id is None: id = index

    row['id'] = id   
    row['data'] = data

    if write_source:
        if filename is not None: row['filename'] = filename
        if offset is not None:   row['offset'] = offset

    if write_label:
        if label is not None: row['label'] = label

    row.append()
    table.flush()

    return index

def write(x, table, table_annot=None, id=None):
    """ Write waveform or spectrogram and annotations to HDF5 tables.

        Note: If the id argument is not specified, the row number will 
        will be used as a unique identifier for the spectrogram.

        Args:
            x: instance of :class:`audio.waveform.Waveform',
                :class:`audio.spectrogram.MagSpectrogram', 
                :class:`audio.spectrogram.PowerSpectrogram',
                :class:`audio.spectrogram.MelSpectrogram', 
                :class:`audio.spectrogram.CQTSpectrogram'    
                The audio object to be stored in the table.
            table: tables.Table
                Table in which the audio data will be stored.
                (described by table_description()).
            table_annot: tables.Table
                Table in which the annotations will be stored.
                (described by table_description_weak_annot() or table_description_strong_annot()).
            id: int
                Audio object unique identifier. Optional.

        Raises:
            TypeError: if spec is not an audio object    

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
    if table.nrows == 0: write_attrs(table, x)

    data_index = write_audio(table=table, data=x.get_data(), filename=x.get_filename(), 
        offset=x.get_offset(), label=x.get_label(), id=id)

    if table_annot is not None:
        write_annot(table=table_annot, data_index=data_index, annots=x.get_annotations())

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
                Audio objects

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
    if indices is None:
        indices = list(range(table.nrows))

    # loop over items in table
    audio_objs = []
    for idx in indices:
        #current item
        it = table[idx] 

        # keyword arguments needed for initializing object
        kwargs = {}
        for name in table._v_attrs._f_list():
            kwargs[name] = table._v_attrs[name]

        # add filename, offset, and label, if available
        for col_name in ['filename','offset','label']:
            if col_name in table.colnames: 
                val = it[col_name]
                if col_name == 'filename': val = val.decode()
                kwargs[col_name] = val

        # annotations, if any
        if table_annot is None: 
            annot = None
        else: 
            annot_data = table_annot.read_where("""data_index == {0}""".format(idx))
            if len(annot_data) > 0:
                annot = pd.DataFrame()
                for col_name in ['label','start','end','freq_min','freq_max']:
                    if col_name in table_annot.colnames: annot[col_name] = annot_data[col_name] 

        # initialize audio object
        audio_class = al.audio_repres_dict[table.attrs.type]
        audio_objs.append(audio_class(data=it['data'], annot=annot, **kwargs))

    if stack:
        audio_objs = audio_class.stack(audio_objs)

    return audio_objs
    

def create_database(output_file, data_dir, selections, channel=0, 
    audio_repres={'type': 'Waveform'}, annotations=None, dataset_name=None,
    max_size=None, verbose=True, progress_bar=True, discard_wrong_shape=False, 
    allow_resizing=1, include_source=True, include_label=True):
    """ Create a database from a selection table.

        Note that all selections must have the same duration. This is necessary to ensure 
        that all the objects stored in the database have the same dimension.

        If each entry in the selection table can have multiple annotations, these can be 
        specified with the 'annotations' argument. On the other hand, if each entry in 
        the selection table is chacterized by a single, integer label, these should be 
        included as a column named 'label' in the selection table.

        If 'dataset_name' is not specified, the name of the folder containing the audio 
        files ('data_dir') will be used.
    
        Args:
            output_file:str
                The name of the HDF5 file in which the data will be stored.
                Can include the path (e.g.:'/home/user/data/database_abc.h5').
                If the file does not exist, it will be created.
                If the file already exists, new data will be appended to it.
            data_dir:str
                Path to folder containing *.wav files.
            selections: pandas DataFrame
                Selection table
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            audio_repres: dict
                A dictionary containing the parameters used to generate the spectrogram or waveform
                segments. See :class:~ketos.audio.auio_loader.AudioLoader for details on the 
                required and optional fields for each type of signal.
            annotations: pandas DataFrame
                Annotation table. Optional.
            dataset_name:str
                Name of the node (HDF5 group) within the database (e.g.: 'train')
                Under this node, two datasets will be created: 'data' and 'data_annot',
                containing the data (spectrograms or waveforms) and the annotations for each
                entry in the selections_table.                
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
                Include integer label column in data table. Only relevant for weakly annotated samples. Default is True.
    """
    loader = al.AudioSelectionLoader(path=data_dir, selections=selections, channel=channel, 
        repres=audio_repres, annotations=annotations)

    writer = AudioWriter(output_file=output_file, max_size=max_size, verbose=verbose, mode = 'a',
        discard_wrong_shape=discard_wrong_shape, allow_resizing=allow_resizing, 
        include_source=include_source, include_label=include_label)
    
    if dataset_name is None: dataset_name = os.path.basename(data_dir)
    path_to_dataset = dataset_name if dataset_name.startswith('/') else '/' + dataset_name
    
    for _ in tqdm(range(loader.num()), disable = not progress_bar):
        x = next(loader)
        writer.write(x=x, path=path_to_dataset, name='data')

    writer.close()

class AudioWriter():
    """ Saves waveform or spectrogram objects to a database file (*.h5).

        If the combined size of the saved data exceeds max_size (1 GB by default), the output database 
        file will be split into several files, with _000, _001, etc, appended to the filename.

        Args:
            output_file: str
                Full path to output database file (*.h5)
            max_size: int
                Maximum size of output database file in bytes.
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte). 
                If None, no restriction is imposed on the file size (i.e. the file 
                is never split).
            verbose: bool
                Print relevant information during execution such as no. of files written to disk
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
            max_filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.

        Attributes:
            base: str
                Output filename base
            ext: str
                Output filename extension (*.h5)
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
                    ’r’: Read-only; no data can be modified.
                    ’w’: Write; a new file is created (an existing file with the same name would be deleted).
                    ’a’: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    ’r+’: It is similar to ‘a’, but the file must already exist.
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
                Include integer label column in data table. Only relevant for weakly annotated samples. Default is True.
            filename_len: int
                Maximum allowed length of filename. Only used if include_source is True.
    """
    def __init__(self, output_file, max_size=1E9, verbose=False, mode='w', discard_wrong_shape=False,
        allow_resizing=1, include_source=True, include_label=True, max_filename_len=100):
        
        self.base = output_file[:output_file.rfind('.')]
        self.ext = output_file[output_file.rfind('.'):]
        self.file = None
        self.file_counter = 0
        self.max_size = max_size
        self.path = '/'
        self.name = 'audio'
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
        self.filename_len = max_filename_len

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

    def write(self, x, path=None, name=None):
        """ Write waveform or spectrogram object to a table in the database file

            If path and name are not specified, the object will be 
            saved to the current directory (as set with the cd() method).

            Args:
                x: Waveform or Spectrogram
                    Object to be saved
                path: str
                    Path to the group containing the table
                name: str
                    Name of the table
        """
        if path is None: path = self.path
        if name is None: name = self.name
        self.set_table(path, name)

        # ensure a file is open
        self._open_file() 

        # record shape of first audio object
        if self.item_counter == 0:
            self.data_shape = x.data.shape

        # open tables, create if they do not already exist
        tbl_dict = self._open_tables(path=path, name=name, x=x) 

        # write spectrogram to table
        shape_diff = np.abs(np.subtract(x.data.shape, self.data_shape))
        if np.sum(shape_diff) > 0 and np.all(shape_diff <= self.allow_resizing): # resize, if necessary and allowed
            x.data = resize(x.data, self.data_shape, anti_aliasing=True)
            self.num_resized += 1

        if x.data.shape == self.data_shape or not self.discard_wrong_shape:
            write(x=x, **tbl_dict)
            self.item_counter += 1

            # close file if size reaches limit
            siz = self.file.get_filesize()
            if self.max_size is not None and siz > self.max_size:
                self.close(final=False)

        else:
            self.num_discarded += 1

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
                x: Waveform or Spectrogram
                    Object to be saved

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
            annot_type, freq_range = self._detect_annot_type(x)
            include_label = self.include_label and (annot_type == 'weak')
            descr = table_description(data_shape=x.data.shape, 
                                      include_label=include_label, 
                                      include_source=self.include_source, 
                                      filename_len=self.filename_len)

            tbl = create_table(h5file=self.file, path=path, name=name, description=descr)
            tbl_dict = {'table': tbl}

            if annot_type is 'strong': 
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

            self.file = open_file(fname, self.mode)
            self.file_counter += 1

    def _detect_annot_type(self, x):
        """ Detect the annotation type (weak or strong)
        """                
        if x.get_annotations() is None: 
            annot_type = 'weak'
            freq_range = False
        else:
            annot_type = 'strong'
            freq_range = ('freq_min' in x.get_annotations().columns)

        return annot_type, freq_range