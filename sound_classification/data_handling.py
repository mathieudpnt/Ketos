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

def slice_ffmpeg(file,start,end, out_name):
    """ Creates an audio segment from a longer audio file

        Args:
            file: str
                The path to the original file.
            start: str
                The start time in seconds.
            end: str
                The end time in seconds.
            out_name: str
                The path to the autput file.
        
    """
    call(["ffmpeg","-loglevel", "quiet", "-i", file, "-ss", start, "-to", end, "-y", out_name])


def create_segments(audio_file, seg_duration, destination, prefix=None):
    """ Creates a series of segments of the same length

        Args:
            audio_file: str
                Path to the original audio file.
            seg_duration:float
                Duration of each segment (in seconds).
            destination:str
                Path to the folder where the segments will be saved.
            prefix: str
                The prefix to be used in the name of segment files.
                The file name will have the format <prefix>_xx.wav,
                where 'xx' is the segment number in the sequence.
                If set to none, the prefix will be the name of the original file.

    """

    create_dir(destination)
    orig_audio_duration = librosa.get_duration(filename=audio_file)
    n_seg = round(orig_audio_duration/seg_duration)
    
    for s in range(n_seg):
        start = str(s)
        end = str(s + seg_duration)

        if prefix is None:
            prefix = os.path.basename(audio_file).split(".wav")[0]

        out_name = prefix + "_" + str(s) + ".wav"
        path_to_seg = os.path.join(destination, out_name)    
        slice_ffmpeg(file=audio_file, start=start, end=end, out_name=path_to_seg)
        print("Creating segment......", path_to_seg)
    


def to1hot(row):
    """Converts the binary label to one hot format

            Args:
                row: bool/int(0 or 1)
                    The the label to be converted.
            
            Returns:
                one_hot:numpy array
                    A 1 by 2 array containg [1,0] if row was 0
                    and [0,1] if it was 1.
     """

    one_hot = np.zeros(2)
    one_hot[row] = 1.0
    return one_hot


def from1hot(row):
    """Converts the one hot label to binary format

            Args:
                row: numpy array
                    The the label to be converted. ([0,1] or [1,0])
            
            Returns:
                one_hot:float
                    A scalar of value 0.0 if row was [1,0] and 1.0 
                    if row was [0,1].
     """
     
    value = 0.0
    if row[1] == 1.0:
        value = 1.0
    return value


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
    """
    assert x_column in database.columns, "database does not contain image column named '{0}'".format(x_column)   
    assert y_column in database.columns, "database does not contain label column named '{0}'".format(y_column)

    database["one_hot_encoding"] = database[y_column].apply(to1hot)
    database["x_flatten"] = database[x_column].apply(lambda x: x.flatten())
    return database


def split_database(database, divisions):
    """ Split the database into 3 datasets: train, validation, test.

        Args:
        database : pandas.DataFrame
            The database to be split. Must contain at least 2 colummns (x, y).
            Each row is an example.
        divisions: dict
            Dictionary indicating the initial and final rows for each dataset.
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
                A pandas dataset with two columns:'x_flatten'and
                'one_hot_encoding' (the output of the 'encode_database' function)
            input_shape: tuple (int,int)
                A tuple specifying the shape of the input images in pixels. Example: (128,128)

        Results:
            stacked_dataset: dict (of numpy arrays)
            A dictionary containing the stacked versions of the input and labels, 
            respectively under the keys 'x' and 'y'
    """

    x = np.vstack(dataset.x_flatten).reshape(dataset.shape[0], input_shape[0], input_shape[1],1).astype(np.float32)
    y = np.vstack(dataset.one_hot_encoding)

    stacked_dataset = {'x': x,
                       'y': y}

    return stacked_dataset


def prepare_database(database, x_column, y_column, divisions, input_shape):
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
            input_shape: tuple (int,int)
                A tuple specifying the shape of the input images in pixels.
                Example: (128,128)

        Returns:
            stacked_datasets: dict
                A dictionary containing the stacked datasets.
                Keys are: train_x, train_y, validation_x, validation_y
                          test_x and test_y. Values are the respective
                                             stacked datasets (numpy arrays)
    """

    encoded_data = encode_database(database=database, x_column=x_column, y_column=y_column)
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