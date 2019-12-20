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

    A Ketos annotation table always contains the columns 'filename' and 'label'.

    For call-level annotations, the table also contains the columns 'time_start' 
    and 'time_stop', giving the start and end time of the call measured in seconds 
    since the beginning of the file. 
    
    The table may also contain the columns 'freq_min' and 'freq_max', giving the 
    minimum and maximum frequencies of the call in Hz, but this is not required.
    
    The user may add any number of additional columns.
"""

import os
import numpy as np
import pandas as pd
from ketos.utils import str_is_int


def unfold(table, sep=','):
    """ Unfolds rows containing multiple labels.

        Args:
            table: pandas DataFrame
                Annotation table.
            sep: str
                Character used to separate multiple labels.

        Returns:
            : pandas DataFrame
                Unfolded table
    """
    df = table    
    df = df.astype({'label': 'str'})
    s = df.label.str.split(",").apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'label'
    del df['label']
    df = df.join(s)
    return df

def rename_columns(table, mapper):
    """ Renames the table headings to conform with the ketos naming convention.

        Args:
            table: pandas DataFrame
                Annotation table.
            mapper: dict
                Dictionary mapping the headings of the input table to the 
                standard ketos headings.

        Returns:
            : pandas DataFrame
                Table with new headings
    """
    return table.rename(columns=mapper)

def standardize(table=None, filename=None, sep=',', mapper=None, signal_labels=None,\
    backgr_labels=[], unfold_labels=False, label_sep=',', trim_table=False):
    """ Standardize the annotation table format.

        The table can be passed as a pandas DataFrame or as the filename of a csv file.

        The table may have either a single label per row, in which case unfold_labels should be set 
        to False, or multiple labels per row (e.g. as a comma-separated list of values), in which 
        case unfold_labels should be set to True.

        The table headings are renamed to conform with the ketos standard naming convention, following the 
        name mapping specified by the user. 

        Signal labels are mapped to integers 1,2,3,... while background labels are mapped to 0, 
        and any remaining labels are mapped to -1.        

        Args:
            table: pandas DataFrame
                Annotation table.
            filename: str
                Full path to csv file containing the annotation table. 
            sep: str
                Separator. Only relevant if filename is specified. Default is ",".
            mapper: dict
                Dictionary mapping the headings of the input table to the 
                standard ketos headings.
            signal_labels: list
                Labels of interest. Will be mapped to 1,2,3,...
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).
            unfold_labels: bool
                Should be set to True if any of the rows have multiple labels. 
                Shoudl be set to False otherwise (default).
            label_sep: str
                Character used to separate multiple labels. Only relevant if unfold_labels is set to True. Default is ",".
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format.

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

    if unfold_labels:
        df = unfold(df, sep=label_sep)

    # cast label column to str
    df = df.astype({'label': 'str'})

    # keep only relevant columns
    if trim_table:
        df = trim(df)

    # check that dataframe has minimum required columns
    mis = missing_columns(df)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)
    
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
    """ Keep only the columns prescribed by the Ketos annotation format.

        Args:
            table: pandas DataFrame
                Annotation table. 

        Returns:
            table: pandas DataFrame
                Annotation table, after removal of columns.
    """
    keep_cols = ['filename', 'label', 'time_start', 'time_stop', 'freq_min', 'freq_max']
    drop_cols = [x for x in table.columns.values if x not in keep_cols]
    table = table.drop(drop_cols, axis=1)
    return table

def missing_columns(table, has_time=False):
    """ Check if the table has the minimum required columns.

        Args:
            table: pandas DataFrame
                Annotation table. 
            has_time: bool
                Require time information for each annotation, i.e. start and stop times.

        Returns:
            mis: list
                List of missing columns, if any.
    """
    required_cols = ['filename', 'label']
    if has_time:
        required_cols = required_cols + ['time_start', 'time_stop']

    mis = [x for x in required_cols if x not in table.columns.values]
    return mis

def create_label_dict(signal_labels, backgr_labels, ignore_labels):
    """ Create label dictionary, following the convetion:

            * signal_labels are mapped to 1,2,3,...
            * backgr_labels are mapped to 0
            * ignore_labels are mapped to -1

        Args:
            signal_labels: list
                Labels of interest. Will be mapped to 1,2,3,...
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).
            ignore_labels: list
                Labels will be grouped into a common "ignore" class (-1).

        Returns:
            label_dict: dict
                Dict that maps old labels to new labels.
    """
    label_dict = dict()    
    for l in ignore_labels: label_dict[l] = -1
    for l in backgr_labels: label_dict[l] = 0
    num = 1
    for l in signal_labels:
        label_dict[l] = num
        num += 1

    return label_dict

def label_occurrence(table):
    """ Identify the unique labels occurring in the table and determine how often 
        each label occurs.

        The input table must have the standardized Ketos format, see 
        :func:`data_handling.annotation_table.standardize`. In particular, each 
        annotation should have only a single label value.

        Args:
            table: pandas DataFrame
                Input table.

        Results:
            occurrence: dict
                Dictionary where the labels are the keys and the values are the occurrences.
    """
    occurrence = table.groupby('label').size().to_dict()
    return occurrence


def trainify(table, seg_len, balance=None, map_orig=False):
    """ Generate an annotation table suitable for training a machine-learning model.

        The input table must have the standardized Ketos format and contain call-level 
        annotations, see :func:`data_handling.annotation_table.standardize`.

        option to query by string ...

        Args:
            table: pandas DataFrame
                Input table with call-level annotations.
            seg_len: float
                Segment length in seconds.
            balance: str
                Class balancing method. Options are: None, 'down', 'up'.   
            map_orig: bool
                Include the original time-frequency bounding box in the 
                output table. This makes it possible to map the new annotations 
                to the original ones. Default is False.

        Results:
            table_train: pandas DataFrame
                Output annotation table.
    """
    df = table

    # check that input table has expected format
    mis = missing_columns(df, has_time=True)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)

    #x = label_occurrence(df)

    table_train = table
    return table_train
