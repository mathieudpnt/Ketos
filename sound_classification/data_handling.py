"""Data Handling module within the sound_classification package

This module includes functions useful when preparing data for 
Deep Neural Networks applied to sound classification within MERIDIAN.

Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: To package code useful for handling data, deriving features and 
             creating Deep Neural Networks for sound classification projects.
     
    License:

"""
import numpy as np
import librosa
import os
import errno
from subprocess import call
import scipy.io.wavfile as wave
import sound_classification.external.wavfile as wave_bit
import datetime
import datetime_glob

def parse_datetime(fname, fmt=None, replace_spaces='0'):
    """
        Parse date-time data from string.

        Uses the datetime_glob package (https://pypi.org/project/datetime-glob/).

        Returns None if parsing fails.
        
        Args:
            fname: str
                String with date-time data.
            datetime_fmt: str
                Date-time format
            replace_spaces: str
                If string contains spaces, replaces them with this string

        Returns:
            datetime
                datetime object
    """

    # replace spaces with zeros
    for i in range(len(fname)):
        if fname[i] == ' ':
            fname = fname[:i] + replace_spaces + fname[i+1:]

    if fmt is not None:
        matcher = datetime_glob.Matcher(pattern=fmt)
        match = matcher.match(path=fname)
        if match is None:
            return None
        else:
            return match.as_datetime()

    return None

def get_files(path, substr, fullpath=True):
    """ Find all files in the specified directory containing the specified substring in their file name

        Args:
            path: str
                Directory path
            substr: str
                Substring contained in file name
            fullpath: bool
                Return full path to each file or just the file name 

        Returns:
            files: list (str)
                Alphabetically sorted list of file names
    """
    allfiles = os.listdir(path)
    files = list()
    for f in allfiles:
        if substr in f:
            if fullpath:
                x = path
                if path[-1] is not '/':
                    x += '/'
                files.append(x + f)
            else:
                files.append(f)
    files.sort()
    return files


def get_wave_files(path, fullpath=True):
    """ Find all wave files in the specified directory

        Args:
            path: str
                Directory path
            fullpath: bool
                Return full path to each file or just the file name 

        Returns:
            wavefiles: list (str)
                Alphabetically sorted list of file names
    """
    wavefiles = get_files(path, '.wav', fullpath)
    return wavefiles


def read_wave(file, channel=0):
    """ Read wave file in either mono or stereo mode

        Args:
            file: str
                Wave file path
            channel: bool
                Which channel should be used in case of stereo data (0: left, 1: right) 
    """
    try:
        rate, signal, _ = wave_bit.read(file)
    except TypeError:
        rate, signal = wave.read(file)
        
    if len(signal.shape) == 2:
        data = signal[:, channel]
    else:
        data = signal[:]
    return rate, data

def create_dir(dir):
    """ Create a new directory only if it does not exist

        Args:
            dir: str
                The path to the new directory
        Raises:
                EEXIST (17) if dir already exists
    """
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def slice_ffmpeg(file,start,end,out_name):
    """ Creates an audio segment from a longer audio file using ffmpeg package.

        Args:
            file: str
                The path to the original file.
            start: float
                The start time in seconds.
            end: float
                The end time in seconds.
            out_name: str
                The path to the output file.
        
    """
    call(["ffmpeg", "-loglevel", "quiet", "-i", file, "-ss", str(start), "-to", str(end), "-y", out_name])



def to1hot(value,depth):
    """Converts the binary label to one hot format

            Args:
                value: scalar or numpy.array | int or float
                    The the label to be converted.
                depth: int
                    The number of possible values for the labels 
                    (number of categories).
                                
            Returns:
                one_hot:numpy array (dtype=float64)
                    A len(value) by depth array containg the one hot encoding
                    for the given value(s).

            Example:
                >>> values = np.array([0,1])
                >>> to1hot(values,depth=2)
                array([[1., 0.],
                      [0., 1.]])
     """
    value = np.int64(value)
    one_hot = np.eye(depth)[value]
    return one_hot


def from1hot(value):
    """Converts the one hot label to binary format

            Args:
                value: scalar or numpy.array | int or float
                    The the label to be converted.
            
            Returns:
                output: int or numpy array (dtype=int64)
                    An int representing the category if 'value' has 1 dimension or an
                    array of m ints if  input values is an n by m array.

            Example:
                >>> from1hot(np.array([0,0,0,1,0]))
                3
                >>> from1hot(np.array([[0,0,0,1,0],
                   [0,1,0,0,0]]))
                array([3, 1])

     """

    if value.ndim > 1:
        output = np.apply_along_axis(arr=value, axis=1, func1d=np.argmax)
        output.dtype = np.int64
    else:
        output = np.argmax(value)

    return output


def encode_database(database, x_column, y_column):
    """ Encodes database in a format suitable for machine learning:
         - the input images are flattened (i.e. matrices converted to row vectors).
         - the labels are one-hot encoded.

    Args:
        database: pandas DataFrame
            A the database containing at least one column of input images
            and one column of labels
        x_column: str
            The name of the column to be used as input
        y_column: str
            The name of the column to be used as label.
            Must be binary (only have 1s or 0s).

        Returns:
            database: pandas DataFrame
                The encoded database with two columns: 'x_flatten' containing
                the flatten input images (as vectors instead of matrices) and
                'one_hot_encoding' containing the one hot version of the labels.
            image_shape: tuple (int,int)
                Tuple specifying the shape of the input images in pixels. Example: (128,128)
    """

    # assert that columns exist
    assert x_column in database.columns, "database does not contain image column named '{0}'".format(x_column)   
    assert y_column in database.columns, "database does not contain label column named '{0}'".format(y_column)

    # check data sanity
    check_data_sanity(database[x_column], database[y_column])

    # determine image size
    image_shape = get_image_size(database[x_column])

    depth = database[y_column].max() + 1 #number of classes
    database["one_hot_encoding"] = database[y_column].apply(to1hot,depth=depth)
    database["x_flatten"] = database[x_column].apply(lambda x: x.flatten())

    return database, image_shape


def split_database(database, divisions):
    """ Split the database into 3 datasets: train, validation, test.

        Args:
        database : pandas.DataFrame
            The database to be split. Must contain at least 2 colummns (x, y).
            Each row is an example.
        divisions: dict
            Dictionary indicating the initial and final rows for each dataset (Obs: final row is *not* included).
            Keys must be "train", "validation" and "test".
            values are tuples with initial and final rows.
            Example: {"train":(0,1000),
                        "validation": (1000,1200),
                        "test": (1200:1400)}
         Returns:
            datasets : dict
                Dictionary with "train", "validation" and "test" as keys
                and the respective datasets (pandas.Dataframes) as values.
    """
    assert "train" in divisions, "'divisions' does not contain key 'train'"   
    assert "validation" in divisions, "'divisions' does not contain key 'validation'"   
    assert "test" in divisions, "'divisions' does not contain key 'test'"   

    train_data = database[divisions["train"][0]:divisions["train"][1]]
    validation_data = database[divisions["validation"][0]:divisions["validation"][1]]
    test_data = database[divisions["test"][0]:divisions["test"][1]]

    datasets = {"train": train_data,
                "validation": validation_data,
                "test": test_data}

    return datasets


def stack_dataset(dataset, input_shape):
    """ Stack and reshape a dataset.

     
        Args:
            dataset: pandas DataFrame
                A pandas dataset with two columns:'x_flatten' and
                'one_hot_encoding' (the output of the 'encode_database' function)

        Results:
            stacked_dataset: dict (of numpy arrays)
            A dictionary containing the stacked versions of the input and labels, 
            respectively under the keys 'x' and 'y'
    """

    assert "x_flatten" in dataset.columns, "'dataset' does not contain column named 'x_flatten'"   
    assert "one_hot_encoding" in dataset.columns, "'dataset' does not contain column named 'one_hot_encoding'"

    x = np.vstack(dataset.x_flatten).reshape(dataset.shape[0], input_shape[0], input_shape[1],1).astype(np.float32)
    y = np.vstack(dataset.one_hot_encoding)

    stacked_dataset = {'x': x,
                       'y': y}

    return stacked_dataset


def prepare_database(database, x_column, y_column, divisions):
    """ Encode data base, split it into training, validation and test sets
        and stack those sets.

        This function is a wrap around the 'encode_database', 'split_databases'
        and 'stack_datasets'

        Args:
            database: pandas DataFrame
                A database containing at least one column of input images
                and one column of labels
            x_column: str
                The name of the column to be used as input
            y_column: str
                The name of the column to be used as label.
                Must be binary (only have 1s or 0s).
            divisions: dict
                Dictionary indicating the initial and final rows for each dataset.
                Keys must be "train", "validation" and "test".
                values are tuples with initial and final rows.
                Example: {"train":(0,1000),
                            "validation": (1000,1200),
                            "test": (1200:1400)}

        Returns:
            stacked_datasets: dict
                A dictionary containing the stacked datasets.
                Keys are: train_x, train_y, validation_x, validation_y
                          test_x and test_y. Values are the respective
                                             stacked datasets (numpy arrays)
    """

    encoded_data, input_shape = encode_database(database=database, x_column=x_column, y_column=y_column)
    datasets = split_database(database=encoded_data, divisions=divisions)
    
    stacked_train = stack_dataset(dataset=datasets["train"], input_shape=input_shape)
    stacked_validation = stack_dataset(dataset=datasets["validation"], input_shape=input_shape)
    stacked_test = stack_dataset(dataset=datasets["test"], input_shape=input_shape)
    
    stacked_datasets = {"train_x": stacked_train["x"],
                        "train_y": stacked_train["y"],
                        "validation_x": stacked_validation["x"],
                        "validation_y": stacked_validation["y"],
                        "test_x": stacked_test["x"],
                        "test_y": stacked_test["y"]}

    return stacked_datasets

def check_data_sanity(images, labels):
    """ Check that all images have same size, all labels have values, 
        and number of images and labels match.
     
        Args:
            images: numpy array
                Images
            labels: numpy array
                Labels

        Results:
            image_size: tuple (int,int)
                Image size
    """
    # check that number of images matches numbers of labels
    assert len(images) == len(labels), "Image and label columns have different lengths"

    # determine image size and check that all images have same size
    image_shape = images[0].shape
    assert all(x.shape == image_shape for x in images), "Images do not all have the same size"

    # check that all labels have values
    b = np.isnan(labels)    
    n = np.count_nonzero(b)
    assert n == 0, "Some labels are NaN"

def get_image_size(images):
    """ Get image size and check that all images have same size.
     
        Args:
            images: numpy array
                Images

        Results:
            image_size: tuple (int,int)
                Image size
    """
    # determine image size and check that all images have same size
    image_shape = images[0].shape
    assert all(x.shape == image_shape for x in images), "Images do not all have the same size"

    return image_shape


def create_raw_signal_table_description(signal_rate, segment_length):
    """ Create the class that describes the raw signal table structure for the HDF5 database.
     
        Args:
            signal_rate: int
                The sampling rate of the signals to be stored in this table
            segment_length: float
                The duration of each segment (in seconds) that will be stored in this table.
                All segments must have the same length

        Results:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure. To be used when creating tables 
                in the HDF5 database.
    """


    signal_length = int(np.ceil(signal_rate * segment_length))

    class TableDescription(tables.IsDescription):
            id = tables.StringCol(25)
            labels = tables.StringCol(100)
            signal = tables.Float32Col(shape=(signal_length))
            boxes = tables.StringCol(100)

    
    return TableDescription


def create_image_table_description(dimensions):
    """ Create the class that describes an image (e.g.: a spectogram) table structure for the HDF5 database.
     
        
        Args:
            dimension : tuple (ints)
            A tuple with ints describing the number of rows and number of collumns of each 
            image to be stored in the table (n_rows,n_cols). Optionally, a third integer 
            can be added if the image has multiple channels (n_rows, n_cols, n_channels)
        Results:
            TableDescription: class (tables.IsDescription)
                The class describing the table structure to be used when creating tables that 
                will store images in the HDF5 database.
    """


    class TableDescription(tables.IsDescription):
            id = tables.StringCol(25)
            labels = tables.StringCol(100)
            signal = tables.Float32Col(shape=dimensions)
            boxes = tables.StringCol(100) 
    
    return TableDescription


def get_data_from_seg_name(seg_name):
    """ Retrieves the segment id and label from the segment name

        Args:
            segment_name: str
            Name of the segment in the format id_*_l_*.wav, where 'id' is 
            followed by any number of characters identifying the segment and 'l' is
            followed by any number of characters describing the label(s).

        Returns:
            (id,label) : tuple (str,str)
            A tuple with the id and label strings.

    """

    id = seg_name.split("_")[1]
    tmp = seg_name.split("_")[3]
    labels = tmp.split(".")[0]

    return (id,labels)

def write_sig_to_h5_database(seg_file_name,table):
    """ Write data form .wav files containing segments into the h5 database.

        Args:
            seg_file: str
                .wav file name (including path).
                Expected to follow the format:format id_*_l_*.wav,
                where * denotes 'any number of characters'.
            table: tables.Table
                Table in which the segment will be stored
                (described by create_raw_signal_table_description()).
        Returns:
            None.
    """

    _, seg_data = read_wave(seg)
    id, labels = get_data_from_seg_name(seg)

    seg_r = table.row
    seg_r["id"] = id
    seg_r["labels"] = labels
    seg_r["signal"] = seg_data 
    seg_r.append()

#TODO: Consider passing an instance of the spectogram class. 
# It could include the id+label string in it's tag attribute.
# Provided that the  id_*_l_* format is used, the get_data_from_seg_name()
# function could then be used to extract the id and label.
def write_spectogram_to_h5_database(spectogram,table):
    """ Write data from spectogram object into the h5 database.

        Note: the spectrogram object is expected to have the id and label information in it's 
        .tag attribute, following the format id_*_l_*.
        Example: spec.tag="id_78536_l_1"

        Args:
            spectogram: instance of :class:`spectrogram.MagSpectrogram', \
            :class:`spectrogram.PowerSpectrogram' or :class:`spectrogram.MelSpectrogram'.
                Spectrogram object.

            table: tables.Table
                Table in which the spectogram will be stored
                (described by create_image_table_description()).
        Returns:
            None.
    """

    id, labels = get_data_from_seg_name(spectogram.tag)
    seg_r = table.row
    seg_r["id"] = id
    seg_r["labels"] = labels
    seg_r["signal"] = spectogram.image
    seg_r.append()

def open_or_create_table(h5, group, table_name,table_description,chunkshape):
    """ Open the specified table or creates it if it does not exist.

        Args:
            h5: tables.file.File object
            HDF5 file handler for the database where the table is/will be located
            group: str
            The path to the group from the root node. Ex: "/group_1/subgroup_1"
            table_name: str
            The name of the table. This name will be part of the table's path.
            Ex: 'table_a' passed along with group="/group_1/subgroup_1" would result in "/group_1/subgroup_1/table_a"
            table_description: tables.IsDescription object
            The descriptor class. See :func:`create_raw_signal_table_description` and :func:create_image_table_description
            chunkshape: tuple

        Returns:
            table: table.Table object
            The opened/created table.    
    """

    try:
       group = h5.get_node(group)
    
    except tables.NoSuchNodeError:
        print("group '{0}' not found. Creating it now...".format(group))
        group = h5.create_group(group)
        
    try:
       table = h5.get_node("{0}/{1}".format(group_name,table_name))
    
    except tables.NoSuchNodeError:    
        filters = tables.Filters(complevel=1, fletcher32=True)
        table = h5.create_table(group,"{0}".format(table_name),table_description,filters=filters,chunkshape=chunkshape)

    return table



def divide_audio_into_segs(audio_file, seg_duration, save_to, prefix=None):
    """ Divides a large .wav file into a sequence of smaller segments with the same duration.
        Names the resulting segments sequentially and save them as .wav files in the specified directory.

        Args:
            audio_file:str
            .wav file name (including path).

            seg_duration: float
            desired duration for each segment
            
            save_to: str
            path to the directory where segments will be saved.
            
            prefix: str
            Prefix to be added to the segment file name.
            Segments will be numbered sequentially.
            Ex: prefix_1.wav, prefix_2.wav...


         Returns:
            None   

    """
    pass

def seg_from_time_tag(audio_file, start, end, name, save_to):
    """ Extracts a segment from the audio_file according to the start and end tags.

        Args:
            audio_file:str
            .wav file name (including path).

            start:float
            Start point for segment (in seconds from start of the source audio_file)

            end:float
            End point for segment (in seconds from start of the source audio_file)
            
            save_to: str
            Path to the directory where segments will be saved.
            
            name: str
            Name of segment file name (including '.wav')
            

         Returns:
            None   

    """

    pass

def segs_from_list_of_tags(audio_file, list_of_tags, save_to, list_of_names=None):
     """ Applies :func:`seg_from_time_tag` to audio_file to extract
      multiple segments based on a list of (start, end) tags.


        Args:
            audio_file:str
            .wav file name (including path).

            save_to: str
            path to the directory where segments will be saved.
            
            prefix: list of str
            Names to be used as file names.
            If None, segments will start with the original audio_file name,
            followed by a sequence number.

         Returns:
            None   

    """ 
    pass 


