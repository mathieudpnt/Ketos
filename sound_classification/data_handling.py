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
    call(["ffmpeg","-loglevel", "quiet", "-i", file, "-ss", start, "-to", end, "-y", out_name])



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
    """ Encodes database so that it has flatten inputs and one hot labels.

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
    
    database["one_hot_encoding"] = database[y_column].apply(to1hot)
    database["x_flatten"] = database[x_column].apply(lambda x: x.flatten())
    return database


def split_database(database, boundaries):
    """ Split the database into 3 datasets: train, validation, test.

        Args:
        database : pandas.DataFrame
            The database to be split. Must contain at least 2 colummns (x, y).
            Each row is an example.
        boundaries: dict
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

    train_data = database[boundaries["train"][0]:boundaries["train"][1]]
    validation_data = database[boundaries["validation"][0]:boundaries["validation"][1]]
    test_data = database[boundaries["test"][0]:boundaries["test"][1]]

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


def prepare_database(database, x_column, y_column, boundaries, input_shape):
    """ Encode data base, split it into training, validation and test sets
        and stack those sets.

        This function is a wrap around the 'encode_database', 'split_databases'
        and 'stack_datasets'

        Args:
            database: pandas DataFrame
                A the database containing at least one column of input images
                and one column of labels
            x_column: str
                The name of the column to be used as input
            y_column: str
                The name of the column to be used as label.
                Must be binary (only have 1s or 0s).
            boundaries: dict
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
    datasets = split_database(database=encoded_data, boundaries=boundaries)
    
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