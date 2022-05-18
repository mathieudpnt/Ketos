Overview
========

The data handling modules provide high-level interfaces for storing audio samples in 
databases along with relevant metadata and annotations, and for retrieving stored data 
for efficient ingestion into neural networks.
Ketos uses the `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ database 
format, a file format designed to store and organize large amounts of data which is 
widely used in scientific computing. 
The data handling modules also provide high-level functionalities for working with
annotation data and selection tables. 


Annotation and Selection Tables 
--------------------------------

The :ref:`selection_table` module provides functions for manipulating annotation 
tables and creating selection tables. The tables are saved in .csv format and 
loaded into memory as `pandas DataFrames 
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.

A Ketos annotation table always has the column 'label'. 
For call-level annotations, the table also contains the columns 'start' 
and 'end', giving the start and end time of the call measured in seconds 
since the beginning of the file. 
The table may also contain the columns 'freq_min' and 'freq_max', giving the 
minimum and maximum frequencies of the call in Hz, but this is not required.    
The user may add any number of additional columns.
Note that the table uses two levels of indices, the first index being the 
filename and the second index an annotation identifier. 

Here is a minimum example::

                        label
    filename  annot_id                    
    file1.wav 0             2
              1             1
              2             2
    file2.wav 0             2
              1             2
              2             1


And here is a table with call-level annotations and a few extra columns::

                         start   end  label  min_freq  max_freq        file_time_stamp
    filename  annot_id                    
    file1.wav 0           7.0    8.1      2     180.6     294.3    2019-02-24 13:15:00
              1           8.5   12.5      1     174.2     258.7    2019-02-24 13:15:00
              2          13.1   14.0      2     183.4     292.3    2019-02-24 13:15:00
    file2.wav 0           2.2    3.1      2     148.8     286.6    2019-02-24 13:30:00
              1           5.8    6.8      2     156.6     278.3    2019-02-24 13:30:00
              2           9.0   13.0      1     178.2     304.5    2019-02-24 13:30:00

Selection tables look similar to annotation tables, except that they are not 
required to have 'label' column. Instead, they typically only have the columns 
'start' and 'end', supplemented by a selection index and a filename index.

When working with annotation tables, the first step is typically to standardize the 
table format to match the format expected by Ketos. For example, given the annotation 
table::

    >>> import pandas as pd
    >>> annot = pd.read_csv('annotations.csv')
    >>> annot
          source  start_time  stop_time       species           time_stamp
    0  file1.wav         7.0        8.1      humpback  2019-02-24 13:15:00
    1  file1.wav         8.5       12.5  killer whale  2019-02-24 13:15:00
    2  file2.wav         2.2        3.1  killer whale  2019-02-24 13:30:00
    3  file2.wav         5.8        6.8          boat  2019-02-24 13:30:00
    4  file2.wav         9.0       13.0      humpback  2019-02-24 13:30:00

we apply the :meth:`standardize() <ketos.data_handling.selection_table.standardize>` 
method to obtain::

    >>> from ketos.data_handling.selection_table import standardize
    >>> annot_std, label_dict = standardize(annot, mapper={'source':'filename', 'start_time':'start', 'stop_time':'end', 'species':'label'}, return_label_dict=True)
    >>> label_dict
    {'boat': 1, 'humpback': 2, 'killer whale': 3}
    >>> annot_std
                        start   end  label           time_stamp
    filename  annot_id                                         
    file1.wav 0           7.0   8.1      2  2019-02-24 13:15:00
              1           8.5  12.5      3  2019-02-24 13:15:00
    file2.wav 0           2.2   3.1      3  2019-02-24 13:30:00
              1           5.8   6.8      1  2019-02-24 13:30:00
              2           9.0  13.0      2  2019-02-24 13:30:00

Having transformed the annotation table to the standard Ketos format, we can now 
use it to create a selection table. The :ref:`selection_table` module provides 
a few methods for this task such as :meth:`select() <ketos.data_handling.selection_table.select>`, 
:meth:`select_by_segmenting() <ketos.data_handling.selection_table.select_by_segmenting>`, and 
:meth:`create_rndm_backgr_selections() <ketos.data_handling.selection_table.create_rndm_backgr_selections>`.
Here, we will demonstrate a simple use case of the :meth:`select() <ketos.data_handling.selection_table.select>` method::

    >>> from ketos.data_handling.selection_table import select
    >>> st = select(df_std, length=6.0, center=True) #create 6-s wide selection windows, centered on each annotation
    >>> st
                      label           time_stamp  start    end
    filename  sel_id                                          
    file1.wav 0           2  2019-02-24 13:15:00   4.55  10.55
              1           3  2019-02-24 13:15:00   7.50  13.50
    file2.wav 0           3  2019-02-24 13:30:00  -0.35   5.65
              1           1  2019-02-24 13:30:00   3.30   9.30
              2           2  2019-02-24 13:30:00   8.00  14.00

Based on this selection table, one can create a database of sound clips using 
:meth:`create_database() <ketos.data_handling.database_interface.create_database>`, 
as discussed below.

The :ref:`selection_table` module provides several other useful methods, e.g., for querying 
annotation tables. See the documentation of the :ref:`selection_table` module for more information.


Database Interface
-------------------
The :ref:`database_interface` module provides high-level functions for managing audio data 
stored in the `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ databases. 
For the implementation of these functionalities, we rely extensively on the 
`PyTables <https://www.pytables.org/index.html>`_ package.

The :class:`AudioWriter <ketos.data_handling.database_interface.AudioWriter>` class provides a convenient 
interface for saving Ketos audio objects such :class:`Waveform <ketos.audio.waveform.Waveform>` 
or :class:`Spectrogram <ketos.audio.spectrogram.Spectrogram>` to a database,::

    >>> from ketos.data_handling.database_interface import AudioWriter
    >>> aw = AudioWriter('db.h5') #create an audio writer instance
    >>> from ketos.audio.spectrogram import MagSpectrogram
    >>> spec = MagSpectrogram.from_wav('sound.wav', window=0.2, step=0.01) #load a spectrogram
    >>> aw.write(spec) #save the spectrogram to the database (by default, the spectrogram is stored under /audio)
    >>> aw.close() #close the database file

The spectrogram is saved along with relevant metadata such as the filename, 
the window and step sizes used, etc. Any annotations associated with the spectrogram 
are also saved.

The spectrogram can be loaded back into memory as follows,::

    >>> import ketos.data_handling.database_interface as dbi
    >>> fil = dbi.open_file('db.h5', 'r')
    >>> tbl = dbi.open_table(fil, '/audio')
    >>> spec = load_audio(tbl)[0]

The :ref:`database_interface` module provides several other useful methods, including 
:meth:`create_database() <ketos.data_handling.database_interface.create_database>` 
for creating a database of audio samples directly from a set of .wav files.

See the documentation of the :ref:`database_interface` module for more information.



Data Feeding
-------------

The :class:`ketos.data_handling.data_feeding.BatchGenerator` class provides a high-level 
interface for loading waveform and spectrogram objects stored in the Ketos HDF5 database 
format and feeding them in batches to a machine learning model. 
See the class documentation for more information.