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
    value=0.0
    if row[1] == 1.0:
        value = 1.0
    return value