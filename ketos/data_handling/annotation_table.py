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

""" annotation_table module within the ketos library

    This module provides functions to create and modify annotation tables. 
"""

import os
import numpy as np
import pandas as pd
from ketos.utils import str_is_int


def standardize(table=None, filename=None, sep=',', mapper=None, signal_labels=None, backgr_labels=[]):
    """ Standardize the annotation table format.

        The table can be passed as a pandas DataFrame or as the filename of a csv file.

        The table headings are renamed to conform with the ketos standard naming convention: 
        'filename', 'start', 'stop', 'fmin', 'fmax', 'label'

        The signal labels are mapped to integers 1,2,3,...,N

        The background labels are mapped to 0.

        Any remaining labels are mapped to -1.        

        Args:
            table: pandas DataFrame
                Annotation table.
            filename: str
                Full path to csv file containing the annotation table. 
            sep: str
                Separator. Only relevant if filename is specified. Default is ",".
            mapper: dict
                Dictionary mapping the headings of the input table to the 
                standard ketos headings (filename, start, stop, fmin, fmax, 
                label).
            signal_labels: list
                Labels of interest. Will be mapped to 1,2,3,...
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).

        Returns:
            table_std: pandas DataFrame
                Standardized annotation table
            label_dict: dict
                Dictionary mapping new labels to old labels
    """
    assert table is not None or filename is not None, 'Either table or filename must be specified'

    # load input table
    if filename is None:
        df = table  
    else:
        assert os.path.exists(filename), 'Could not find input file: {0}'.format(filename)
        df = pd.read_csv(filename, sep=sep)

    # rename columns
    if mapper is not None:
        df = df.rename(columns=mapper)

    # cast label column to str
    df = df.astype({'label': 'str'})

    # keep only relevant columns
    df = trim(df)

    # check that dataframe has minimum required columns
    has_required_columns(df)    

    # create list of unique labels in input table
    labels = list(set(df['label'].values)) 

    if signal_labels is None:
        signal_labels = [x for x in labels if x not in backgr_labels]

    # cast to str
    backgr_labels = [str(x) for x in backgr_labels]
    signal_labels = [str(x) for x in signal_labels]

    # check for labeling conflict
    common_labels = [x for x in backgr_labels if x in signal_labels]
    assert len(common_labels) == 0, 'Duplication of labels in signal_labels and backgr_labels'

    # separate out background labels, if any
    for x in backgr_labels:
        assert x in labels, 'label {0} not found in input table'.format(x)
    
    # ignore remaining labels
    ignore_labels = [x for x in labels if x not in signal_labels and x not in backgr_labels]

    # create label dictionary and apply to label column in DataFrame
    _label_dict = create_label_dict(signal_labels, backgr_labels, ignore_labels)
    df['label'] = df['label'].apply(lambda x: _label_dict.get(x))

    # cast integer dict keys from str back to int
    label_dict = dict()
    for key, value in _label_dict.items():
        if str_is_int(key): key = int(key)
        label_dict[key] = value

    table_std = df
    return table_std, label_dict

def trim(table):
    keep_cols = ['filename', 'start', 'stop', 'label', 'fmin', 'fmax']
    drop_cols = [x for x in table.columns.values if x not in keep_cols]
    table = table.drop(drop_cols, axis=1)
    return table

def has_required_columns(table):
    required_cols = ['filename', 'start', 'stop', 'label']
    for x in required_cols:
        assert x in table.columns.values, 'Column {0} missing from input table'.format(x)

def create_label_dict(signal_labels, backgr_labels, ignore_labels):
    label_dict = dict()    
    for l in ignore_labels: label_dict[l] = -1
    for l in backgr_labels: label_dict[l] = 0
    num = 1
    for l in signal_labels:
        label_dict[l] = num
        num += 1

    return label_dict


##def create(path, seg_len, columns=None, balancing_method=None, labels=None, backgr_labels=0,\
##    rndm_backgr=None):
##    """ Create an annotation table
##    """
##    return None
