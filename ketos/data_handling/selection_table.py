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

""" selection_table module within the ketos library.

    This module provides functions for handling annotation tables and creating 
    selection tables. 
    A Ketos annotation table always has the column 'label'. 
    For call-level annotations, the table also contains the columns 'start' 
    and 'end', giving the start and end time of the call measured in seconds 
    since the beginning of the file. 
    The table may also contain the columns 'freq_min' and 'freq_max', giving the 
    minimum and maximum frequencies of the call in Hz, but this is not required.    
    The user may add any number of additional columns.
    Note that the table uses two levels of indices, the first index being the 
    filename and the second index an annotation identifier. 

    
    Here is a minimum example:

            +----------------------+-------+
            |                      | label |
            +-----------+----------+-------+
            | filename  | annot_id |       |
            +-----------+----------+-------+
            | file1.wav | 0        | 2     |
            +-----------+----------+-------+
            |           | 1        | 1     |
            +-----------+----------+-------+
            |           | 2        | 2     |
            +-----------+----------+-------+
            | file2.wav | 0        | 2     |
            +-----------+----------+-------+
            |           | 1        | 2     |
            +-----------+----------+-------+
            |           | 2        | 1     |
            +-----------+----------+-------+


    And here is a table with time information (call-level annotations) and a few extra columns ('min_freq', 'max_freq' and 'file_time_stamp')

            +----------------------+-------+------+-------+----------+----------+---------------------+
            |                      | start | end  | label | min_freq | max_freq | file_time_stamp     |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
            | filename  | annot_id |                                                                  |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
            | file1.wav | 0        | 7.0   | 8.1  | 2     | 180.6    | 294.3    | 2019-02-24 13:15:00 |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
            |           | 1        | 8.5   | 12.5 | 1     | 174.2    | 258.7    | 2019-02-24 13:15:00 |
            |           +----------+-------+------+-------+----------+----------+---------------------+
            |           | 2        | 13.1  | 14.0 | 2     | 183.4    | 292.3    | 2019-02-24 13:15:00 |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
            | file2.wav | 0        | 2.2   | 3.1  | 2     | 148.8    | 286.6    | 2019-02-24 13:30:00 |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
            |           | 1        | 5.8   | 6.8  | 2     | 156.6    | 278.3    | 2019-02-24 13:30:00 |
            |           +----------+-------+------+-------+----------+----------+---------------------+
            |           | 2        | 9.0   | 13.0 | 1     | 178.2    | 304.5    | 2019-02-24 13:30:00 |
            +-----------+----------+-------+------+-------+----------+----------+---------------------+
    
"""

import os
import librosa
import warnings
import numpy as np
import pandas as pd
from ketos.utils import str_is_int, fractional_overlap
from ketos.data_handling.data_handling import find_wave_files, parse_datetime


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


def empty_annot_table():
    """ Create an empty call-level annotation table

        Returns:
            df: pandas DataFrame
                Empty annotation table
    """
    df = pd.DataFrame(columns=['filename','label','start','end'])
    df = use_multi_indexing(df, 'annot_id')

    return df


def empty_selection_table():
    """ Create an empty selection table

        Returns:
            df: pandas DataFrame
                Empty selection table
    """
    df = pd.DataFrame(columns=['filename','label','start','end', 'annot_id'])
    df = use_multi_indexing(df, 'sel_id')
    return df


def standardize(table=None, path=None, sep=',', mapper=None, labels=None, 
    start_labels_at_1=False, unfold_labels=False, label_sep=',', trim_table=False, datetime_format=None):
    """ Standardize the annotation table format.

        The input table can be passed as a pandas DataFrame or as the filename of a csv file.
        The table may have either a single label per row, in which case unfold_labels should be set 
        to False, or multiple labels per row (e.g. as a comma-separated list of values), in which 
        case unfold_labels should be set to True and label_sep should be specified.

        The table headings are renamed to conform with the ketos standard naming convention, following the 
        name mapping specified by the user. 

        Labels specified by the `labels` argument are mapped to integers 0,1,2,... and any remaining 
        labels are mapped to -1. 

        Note that the labels can be mapped to 1,2,3,.... instead using the `start_labels_at_1` argument.
        This can be useful if you want to reserve the label 0 for background/negative samples.

        Note that the standardized output table has two levels of indices, the first index being the 
        filename and the second index the annotation identifier. 

        The label mapping is stored as a class attribute named 'label_dict' within the output table 
        and may be retrieved with `df.attrs['label_dict']`.

        Args:
            table: pandas DataFrame
                Annotation table.
            path: str
                Full path to csv file containing the annotation table. 
            sep: str
                Separator. Only relevant if filename is specified. Default is ",".
            mapper: dict
                Dictionary mapping the standard ketos headings to the headings of the input table.
                It is also possible to specify mappings that involve mathematical/logical operations 
                on the headings of the input table. For example, `{"end": "x['Start'] + x['Duration']"}`.
            labels: list, or list of lists
                Labels of interest. Will be mapped to 0,1,2,...
                Several labels can be mapped to the same integer by using nested lists. For example, 
                signal_labels=[A,[B,C]] would result in A being mapped to 0 and B and C both being mapped 
                to 1. Any remaining labels not specified by the `labels` argument are mapped to -1.
            start_labels_at_1: bool
                Map labels to 1,2,3,... instead of 0,1,2,... Default is False.
                Useful if you want to reserve the label 0 for background/negative samples.
            unfold_labels: bool
                Should be set to True if any of the rows have multiple labels and False otherwise (default).
            label_sep: str
                Character used to separate multiple labels. Only relevant if unfold_labels is set to True. Default is ",".
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format and any additional columns specified 
                in the mapper dictionary.
            datetime_format: str
                String defining the date-time format. 
                Example: %d_%m_%Y* would capture "14_3_1999.txt".
                See https://pypi.org/project/datetime-glob/ for a list of valid directives.
                If specified, the method will look for a column named 'datetime' and, if found, attempt to parse the 
                values in this column. If your datetime column has a different name, use the `mapper` argument to change 
                its name to 'datetime'. If the method does not find a column named 'datetime' it will attempt to 
                parse the datetime information from the filename column.

        Returns:
            df: pandas DataFrame
                Standardized annotation table
    """
    assert table is not None or path is not None, 'Either table or path must be specified'

    # load input table
    if path is None:
        df = table  
    else:
        assert os.path.exists(path), 'Could not find input file: {0}'.format(path)
        df = pd.read_csv(path, sep=sep)

    # map columns
    if mapper is not None:
        for key,value in mapper.items():
            if value not in df.columns.values:
                df[key] = df.apply(lambda x: eval(value), axis=1) #first, map mathematical/logical operations
        for key,value in mapper.items():
            if value in df.columns.values:
                df = df.rename(columns={value:key}) #second, map 1-to-1 mappings

    # if user has provided duration instead of end time, compute end time
    if 'start' in df.columns.values and 'duration' in df.columns.values and 'end' not in df.columns.values:
        df['end'] = df['start'] + df['duration']

    # keep only relevant columns
    if trim_table:
        df = trim(df, list(mapper.keys()))

    # check that dataframe has minimum required columns
    mis = missing_columns(df)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)

    if unfold_labels:
        df = unfold(df, sep=label_sep)

    # cast label column to str
    df = df.astype({'label': 'str'})

    # create list of unique labels in input table
    unique_labels = np.sort(np.unique(df['label'].values)).tolist()

    if labels is None:
        labels = unique_labels
        discard_labels = []
        
    else:
        assert isinstance(labels, list), 'labels is not a list or list of lists. Found {0}'.format(type(labels))

        labels, labels_flat = cast_to_str(labels, nested=True)

        # Check if the mapped labels are in the unique labels
        for x in labels_flat:
            assert x in unique_labels, 'label {0} not found in input table'.format(x)

        # discard remaining labels
        discard_labels = [x for x in unique_labels if x not in labels_flat]


    # create label dictionary and apply to label column in DataFrame
    _label_dict = _create_label_dict(labels, discard_labels, start_labels_at_1)

    # convert labels to standardized form 
    df['label'] = df['label'].apply(lambda x: _label_dict.get(x))

    # cast integer dict keys from str back to int
    label_dict = dict()
    for key, value in _label_dict.items():
        if str_is_int(key): key = int(key)
        label_dict[key] = value

    # always sort by filename (first) and start time (second)
    by = ['filename']
    if 'start' in df.columns.values: 
        by += ['start']
    df.sort_values(by=by, inplace=True, ignore_index=True)

    # convert path to format suitable for the operating system that is being used
    df['filename'] = df['filename'].apply(lambda x: x.replace("\\","/") if os.name == "posix" else x.replace("/","\\"))

    # parse datetime field
    if datetime_format is not None:
        if 'datetime' in df.columns.values:
            df['datetime'] = df['datetime'].apply(lambda x: parse_datetime(x, fmt=datetime_format))
        else:
            df['datetime'] = df.apply(lambda x: parse_datetime(os.path.basename(x.filename), fmt=datetime_format), axis=1)
    
    # transform to multi-indexing
    df = use_multi_indexing(df, 'annot_id')

    # store label dictionary as class attribute
    df.attrs["label_dict"] = label_dict

    return df


def use_multi_indexing(df, level_1_name):
    """ Change from single-level indexing to double-level indexing. 
        
        The first index level is the filename while the second 
        index level is a cumulative integer.

        Args:
            table: pandas DataFrame
                Singly-indexed table. Must contain a column named 'filename'. 

        Returns:
            table: pandas DataFrame
                Multi-indexed table.
    """
    df = df.set_index([df.filename, df.index])
    df = df.drop(['filename'], axis=1)
    df = df.sort_index()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), df.groupby(level=0).cumcount()],
        names=['filename', level_1_name])

    return df


def trim(table, extra_cols=None):
    """ Keep only the columns prescribed by the Ketos annotation format.

        Args:
            table: pandas DataFrame
                Annotation table. 
            extra_cols: list(str)
                Any additional columns that we wish to keep

        Returns:
            table: pandas DataFrame
                Annotation table, after removal of columns.
    """
    keep_cols = ['filename', 'label', 'start', 'end', 'freq_min', 'freq_max']
    if extra_cols is not None:
        keep_cols += extra_cols
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
        required_cols = required_cols + ['start', 'end']

    mis = [x for x in required_cols if x not in table.columns.values]
    return mis


def is_standardized(table, has_time=False, verbose=True):
    """ Check if the table has the correct indices and the minimum required columns.

        Args:
            table: pandas DataFrame
                Annotation table. 
            has_time: bool
                Require time information for each annotation, i.e. start and stop times.
            verbose: bool
                If True and the table is not standardized, print a message with an example table in the standard format.

        Returns:
            res: bool
                True if the table has the standardized Ketos format. False otherwise.
    """
    required_indices = ['filename', 'annot_id']
    required_cols = ['label']
    if has_time:
        required_cols = required_cols + ['start', 'end']

    mis_cols = [x for x in required_cols if x not in table.columns.values]
    res = (table.index.names == required_indices) and (len(mis_cols) == 0)

    message = """ Your table is not in the Ketos format.

            It should have two levels of indices: filename and annot_id.
            It should also contain at least the 'label' column.
            If your annotations have time information, these should appear in the 'start' and 'end' columns

            extra columns are allowed.

            Here is a minimum example:

                                 label
            filename  annot_id                    
            file1.wav 0          2
                      1          1
                      2          2
            file2.wav 0          2
                      1          2
                      2          1


            And here is a table with time information and a few extra columns ('min_freq', 'max_freq' and 'file_time_stamp')

                                 start   end  label  min_freq  max_freq  file_time_stamp
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2    180.6     294.3    2019-02-24 13:15:00
                      1           8.5  12.5      1    174.2     258.7    2019-02-24 13:15:00
                      2          13.1  14.0      2    183.4     292.3    2019-02-24 13:15:00
            file2.wav 0           2.2   3.1      2    148.8     286.6    2019-02-24 13:30:00
                      1           5.8   6.8      2    156.6     278.3    2019-02-24 13:30:00
                      2           9.0  13.0      1    178.2     304.5    2019-02-24 13:30:00

    
    """
    if res == False and verbose == True:
        print(message)            

    return res


def _create_label_dict(labels, discard_labels=None, start_labels_at_1=False):
    """ Create label dictionary, following the convetion:

            * signal_labels are mapped to 1,2,3,...
            * backgr_labels are mapped to 0
            * discard_labels are mapped to -1

        Args:
            signal_labels: list, or list of lists
                Labels of interest. Will be mapped to 1,2,3,...
                Several labels can be mapped to the same integer by using nested lists. For example, 
                signal_labels=[A,[B,C]] would result in A being mapped to 1 and B and C both being mapped 
                to 2.
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).
            discard_labels: list
                Labels will be grouped into a common "discard" class (-1).
            start_labels_at_1: bool
                Map labels to 1,2,3,... instead of 0,1,2,... Default is False.

        Returns:
            label_dict: dict
                Dict that maps old labels to new labels.
    """
    label_dict = dict()  
    if discard_labels is not None:  
        for l in discard_labels: label_dict[l] = -1
    
    num = 1 if start_labels_at_1 else 0
    for l in labels:
        if isinstance(l, list):
            for ll in l:
                label_dict[ll] = num

        else:
            label_dict[l] = num

        num += 1

    return label_dict


def label_occurrence(table):
    """ Identify the unique labels occurring in the table and determine how often 
        each label occurs.

        The input table must have the standardized Ketos format, see 
        :func:`data_handling.selection_table.standardize`. In particular, each 
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

def cast_to_str(labels, nested=False):
    """ Convert every label to str format. 

        If nested is set to True, a flattened version of the input 
        list is also returned.

        Args:
            labels: list
                Input labels
            nested: bool
                Indicate if the input list contains (or may contain) sublists.
                False by default. If True, a flattened version of the 
                list is also returned.

        Results:
            labels_str: list
                Labels converted to str format
            labels_str_flat: list
                Flattened list of labels. Only returned if nested is set to True.
    """
    if not nested:
        labels_str = [str(x) for x in labels]
        return labels_str

    else:
        labels_str = []
        labels_str_flat = []
        for x in labels:
            if isinstance(x, list):
                sublist = []
                for xx in x:
                    labels_str_flat.append(str(xx))
                    sublist.append(str(xx))

                labels_str.append(sublist)

            else:
                labels_str_flat.append(str(x))
                labels_str.append(str(x))

        return labels_str, labels_str_flat

def select(annotations, length, step=0, min_overlap=0, center=False, discard_long=False, 
    keep_id=False, keep_freq=False, label=None, avoid_label=None, discard_outside=False, files=None):
    """ Generate a selection table by defining intervals of fixed length around 
        annotated sections of the audio data. Each selection created in this 
        way is characterized by a single, integer-valued, label.

        This approach to generating selections lends itself well to cases in which 
        the annotated sections are well separated and rarely overlap. If this is not 
        the case, you may find the related function :func:`data_handling.selection_table.select_by_segmenting` 
        more useful.

        By default all annotated sections are used for generating selections except 
        those with label -1 which are ignored. Use the `label` argument to only 
        generate selections for specific labels. 

        Conversely, the argument `avoid_label` can be used to ensure that the generated 
        selections do not overlap with annotated sections with specific labels. For example, 
        if `label=[1,2]` and `avoid_label=[4]`, selections will be generated for every 
        annotated section with label 1 or 2, but any selection that happens to overlap with 
        an annotated sections with label 4 will be discarded. 

        The input table must have the standardized Ketos format and contain call-level 
        annotations, see :func:`data_handling.selection_table.standardize`.

        The output table uses two levels of indexing, the first level being the 
        filename and the second level being a selection id.

        The generated selections have uniform length given by the `length` argument. Annotated 
        sections longer than the specified length will be cropped (unless discard_long=True) 
        whereas shorter sections will be extended to achieve the specified length. 

        The `step` and `min_overlap` arguments may be used to generate multiple, time-shifted 
        selections for every annotated sections.

        Note that the selections may have negative start times and/or end times 
        that exceed the file duration, unless discard_outside=True in which case only 
        selections with start times and end times within the file duration are returned.

        Args:
            annotations: pandas DataFrame
                Input table with call-level annotations.
            length: float
                Selection length in seconds.
            step: float
                Produce multiple selections for each annotated  section by shifting the selection 
                window in steps of length step (in seconds) both forward and backward in 
                time. The default value is 0.
            min_overlap: float
                Minimum required overlap between the selection and the annotated section, expressed 
                as a fraction of whichever of the two is shorter. Only used if step > 0. 
            center: bool
                Center annotations. Default is False.
            discard_long: bool
                Discard all annotations longer than the output length. Default is False.
            keep_id: bool
                For each generated selection, include the id of the annotation from which 
                the selection was generated.
            keep_freq: bool
                For each generated selection, include the min and max frequency, if known.
            label: int or list(int)
                Only create selections for annotated sections with these labels.
            avoid_label: int, list(int) or str
                Avoid overlap with annotated sections with these labels. If overlap is to be avoided 
                with all other labels but the labels specified by the `label` argument, set `avoid_label="ALL"`.
            discard_outside: bool
                Discard selections that extend beyond file duration. Requires that a file duration 
                table is specified via the `files` argument.
            files: pandas DataFrame
                Table with file durations in seconds. Must contain columns named 'filename' and 'duration'.
                Only required if `discard_outside=True`.

        Results:
            df: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> df = standardize(df, start_labels_at_1=True)
            >>> print(df)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>> 
            >>> #Create a selection table by defining intervals of fixed 
            >>> #length around every annotation.
            >>> #Set the length to 3.0 sec and require a minimum overlap of 16%
            >>> #between selection and annotations.
            >>> #Also, create multiple time-shifted versions of the same selection
            >>> #using a step size of 1.0 sec.     
            >>> df_sel = select(df, length=3.0, step=1.0, min_overlap=0.16, center=True, keep_id=True) 
            >>> print(df_sel.round(2))
                              label  start    end  annot_id
            filename  sel_id                               
            file1.wav 0           2   5.05   8.05         0
                      1           1   6.00   9.00         1
                      2           2   6.05   9.05         0
                      3           1   7.00  10.00         1
                      4           2   7.05  10.05         0
                      5           1   8.00  11.00         1
                      6           1   9.00  12.00         1
                      7           1  10.00  13.00         1
                      8           1  11.00  14.00         1
                      9           2  11.05  14.05         2
                      10          1  12.00  15.00         1
                      11          2  12.05  15.05         2
                      12          2  13.05  16.05         2
            file2.wav 0           2   0.15   3.15         0
                      1           2   1.15   4.15         0
                      2           2   2.15   5.15         0
                      3           2   3.80   6.80         1
                      4           2   4.80   7.80         1
                      5           2   5.80   8.80         1
                      6           1   6.50   9.50         2
                      7           1   7.50  10.50         2
                      8           1   8.50  11.50         2
                      9           1   9.50  12.50         2
                      10          1  10.50  13.50         2
                      11          1  11.50  14.50         2
                      12          1  12.50  15.50         2
    """
    if len(annotations) == 0:
        return empty_selection_table()

    df = annotations.copy()
    df['annot_id'] = df.index.get_level_values(1)

    # check that input table has expected format
    assert is_standardized(df, has_time=True), 'Annotation table appears not to have the expected structure.'

    # select labels
    if label is not None:
        if isinstance(label, int): label = [label]
        df = df[df['label'].isin(label)]

    # discard annotations with label -1
    df = df[df['label'] != -1]

    # number of annotations
    N = len(df)

    # compute length of every annotation
    df['length'] = df['end'] - df['start']

    # discard annotations longer than the requested length
    if discard_long:
        df = df[df['length'] <= length]

    # alignment of new annotations relative to original ones
    if center:
        df['start_new'] = df['start'] + 0.5 * (df['length'] - length)
    else:
        df['start_new'] = df['start'] + np.random.random_sample(N) * (df['length'] - length)

    # create multiple time-shited instances of every annotation
    if step > 0:
        df_new = None
        for idx,row in df.iterrows():
            t = row['start_new']
            df_shift = time_shift(annot=row, time_ref=t, length=length, min_overlap=min_overlap, step=step)
            df_shift['filename'] = idx[0]

            if df_new is None:
                df_new = df_shift
            else:
                df_new = pd.concat([df_new, df_shift])

        # sort by filename and offset
        df = df_new.sort_values(by=['filename','start_new'], axis=0, ascending=[True,True]).reset_index(drop=True)

        # transform to multi-indexing
        df = use_multi_indexing(df, 'sel_id')

    # rename index
    df.index.rename('sel_id', level=1, inplace=True) 

    # drop old/temporary columns, and rename others
    df = df.drop(['start', 'end', 'length'], axis=1)
    df = df.rename(columns={"start_new": "start"})
    df['end'] = df['start'] + length

    # keep annotation id
    if not keep_id:
        df = df.drop(columns=['annot_id'])
    else:
        # re-order columns so annot_it appears last
        cols = df.columns.values.tolist()
        p = cols.index('annot_id')
        cols_new = cols[:p] + cols[p+1:] + ['annot_id']
        df = df[cols_new]
        df = df.astype({'annot_id': int}) #ensure annot_id is int

    # ensure label is integer
    df = df.astype({'label':int})

    # discard selections that overlap with unwanted annotations
    if avoid_label is not None:

        if isinstance(avoid_label, int): 
            avoid_label = [avoid_label]

        elif isinstance(avoid_label, str) and avoid_label.lower() == "all" and label is not None:
            labels = pd.unique(annotations.label)
            avoid_label = labels[~np.isin(labels,label)]
            avoid_label = avoid_label.tolist()

        if isinstance(avoid_label, list):
            def func(y, start, end, label):
                y = y[(y.label.isin(label)) & (y.end>=start) & (y.start<=end)]
                return len(y)
            
            df['overlap'] = df.apply(lambda x: func(annotations.loc[x.name[0]], label=avoid_label, start=x.start, end=x.end), axis=1)
            df = df[df['overlap']==0]
            df = df.drop(columns=['overlap'])
        
    # discard selections that extend beyond duration of file
    if discard_outside:
        if files is None:
            warnings.warn("discard_outside=True requires files to be specified")
        else:
            files = files.set_index('filename')
            df['outside'] = df.apply(lambda x: (x.start < 0) or (x.end > files.loc[x.name[0]].duration), axis=1)
            df = df[df['outside']==False]
            df = df.drop(columns=['outside'])

    if not keep_freq:
        df = df.drop(columns=['freq_min','freq_max'], errors='ignore')

    return df

def time_shift(annot, time_ref, length, step, min_overlap):
    """ Create multiple instances of the same selection by stepping in time, both 
        forward and backward.

        The time-shifted instances are returned in a pandas DataFrame with the same columns as the 
        input annotation, plus a column named 'start_new' containing the start times 
        of the shifted instances.

        Args:
            annot: pandas Series or dict
                Reference annotation. Must contain the labels/keys 'start' and 'end'.
            time_ref: float
                Reference time used as starting point for the stepping.
            length: float
                Output annotation length in seconds.
            step: float
                Produce multiple instances of the same selection by shifting the annotation 
                window in steps of length step (in seconds) both forward and backward in 
                time. The default value is 0.
            min_overlap: float
                Minimum required overlap between the selection intervals and the original 
                annotation, expressed as a fraction of whichever is smaller, the annotation 
                duration or the selection length.   

        Results:
            df: pandas DataFrame
                Output annotation table. The start times of the time-shifted annotations are 
                stored in the column 'start_new'.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import time_shift
            >>> 
            >>> #Create a single 2-s long annotation
            >>> annot = {'filename':'file1.wav', 'label':1, 'start':12.0, 'end':14.0}
            >>>
            >>> #Step across this annotation with a step size of 0.2 s, creating 1-s long annotations that 
            >>> #overlap by at least 50% with the original 2-s annotation 
            >>> df = time_shift(annot, time_ref=13.0, length=1.0, step=0.2, min_overlap=0.5)
            >>> print(df.round(2))
                filename  label  start   end  start_new
            0  file1.wav      1   12.0  14.0       11.6
            1  file1.wav      1   12.0  14.0       11.8
            2  file1.wav      1   12.0  14.0       12.0
            3  file1.wav      1   12.0  14.0       12.2
            4  file1.wav      1   12.0  14.0       12.4
            5  file1.wav      1   12.0  14.0       12.6
            6  file1.wav      1   12.0  14.0       12.8
            7  file1.wav      1   12.0  14.0       13.0
            8  file1.wav      1   12.0  14.0       13.2
            9  file1.wav      1   12.0  14.0       13.4
    """
    if isinstance(annot, dict):
        row = pd.Series(annot)
    elif isinstance(annot, pd.Series):
        row = annot.copy()
    
    row['start_new'] = time_ref
    rows_new = [row]

    # step backwards and forwards
    for sign in [-1, 1]:
        counter = 1
        while True:
            t0 = time_ref + sign * counter * step
            o = fractional_overlap(a=(t0,t0+length), b=(row['start'], row['end']))
            if o < min_overlap: 
                break
            ri = row.copy()
            ri['start_new'] = t0
            rows_new.append(ri)
            counter += 1

    # create DataFrame
    df = pd.DataFrame(rows_new)

    # sort according to new start time
    df = df.sort_values(by=['start_new'], axis=0, ascending=[True]).reset_index(drop=True)

    return df

def file_duration_table(path, search_subdirs=False, datetime_format=None):
    """ Create file duration table.

        Args:
            path: str
                Path to folder with audio files (\*.wav)
            search_subdirs: bool
                If True, search include also any audio files in subdirectories.
                Default is False.
            datetime_format: str
                String defining the date-time format. 
                Example: %d_%m_%Y* would capture "14_3_1999.txt".
                See https://pypi.org/project/datetime-glob/ for a list of valid directives.
                If specified, the method will attempt to parse the datetime information from the filename.

        Returns:
            df: pandas DataFrame
                File duration table. Columns: filename, duration, (datetime)
    """
    paths = find_wave_files(path=path, return_path=True, search_subdirs=search_subdirs)
    durations = [librosa.get_duration(filename=os.path.join(path,p)) for p in paths]
    df = pd.DataFrame({'filename':paths, 'duration':durations})
    if datetime_format is None:
        return df

    df['datetime'] = df.apply(lambda x: parse_datetime(os.path.basename(x.filename), fmt=datetime_format), axis=1)
    return df

def create_rndm_selections(files, length, num, label=0, annotations=None, no_overlap=False, trim_table=False, buffer=0):
    """ Create selections of uniform length, randomly distributed across the 
        data set and not overlapping with any annotations. The created selections
        will have a label value defined by the 'label' parameter.

        The random sampling is performed without regard to already created 
        selections. Therefore, it is in principle possible that some of the created 
        selections will overlap, although in practice this will only occur with very 
        small probability, unless the number of requested selections (num) is very 
        large and/or the (annotation-free part of) the data set is small in size.

        To avoid any overlap, set the 'no_overlap' to True, but note that this can 
        lead to longer execution times.

        Use the 'buffer' argument to ensure a minimum separation between
        selections and the annotated segments. This can be useful if the annotation 
        start and end times are not always fully accurate.

        Args:
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            num: int
                Number of selections to be created.
            label: int
                Value to be assigned to the created selections.
            annotations: pandas DataFrame
                Annotation table. Optional.
            no_overlap: bool
                If True, randomly selected segments will have no overlap.
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format.
            buffer: float
                Minimum separation in seconds between the background selections and the annotated segments.
                The default value is zero.

        Returns:
            table_backgr: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ketos.data_handling.selection_table import select
            >>> 
            >>> #Ensure reproducible results by fixing the random number generator seed.
            >>> np.random.seed(3)
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>> print(df)
                filename  start   end  label
            0  file1.wav    7.0   8.1      1
            1  file1.wav    8.5  12.5      0
            2  file1.wav   13.1  14.0      1
            3  file2.wav    2.2   3.1      1
            4  file2.wav    5.8   6.8      1
            5  file2.wav    9.0  13.0      0
            >>>
            >>> #Standardize annotation table format
            >>> df = standardize(df, start_labels_at_1=True)
            >>> print(df)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>>
            >>> #Enter file durations into a pandas DataFrame
            >>> file_dur = pd.DataFrame({'filename':['file1.wav','file2.wav','file3.wav',], 'duration':[18.,20.,15.]})
            >>> 
            >>> #Create randomly sampled background selection with fixed 3.0-s length.
            >>> df_bgr = create_rndm_selections(annotations=df, files=file_dur, length=3.0, num=12, trim_table=True) 
            >>> print(df_bgr.round(2))
                              start    end  label
            filename  sel_id                     
            file1.wav 0        1.06   4.06      0
                      1        1.31   4.31      0
                      2        2.26   5.26      0
            file2.wav 0       13.56  16.56      0
                      1       14.76  17.76      0
                      2       15.50  18.50      0
                      3       16.16  19.16      0
            file3.wav 0        2.33   5.33      0
                      1        7.29  10.29      0
                      2        7.44  10.44      0
                      3        9.20  12.20      0
                      4       10.94  13.94      0
    """
    if len(files) == 0:
        return empty_selection_table()

    assert isinstance(label, int), 'label is not int. Found {0}'.format(type(label))

    # compute lengths, and discard segments shorter than requested length
    c = files[['filename','duration']]

    if 'offset' in files.columns.names: c['offset'] = files['offset']
    else: c['offset'] = 0

    c.reset_index(drop=True, inplace=True)
    c['length'] = c['duration'] - length
    c = c[c['length'] >= 0]

    # cumulative length 
    cs = c['length'].cumsum().values.astype(float)
    cs = np.concatenate(([0],cs))

    # output
    filename, start, end = [], [], []

    # randomply sample
    df = pd.DataFrame()
    while (len(df) < num):
        times = np.random.random_sample(num) * cs[-1]
        for t in times:
            idx = np.argmax(t < cs) - 1
            row = c.iloc[idx]
            fname = row['filename']
            start = t - cs[idx] + row['offset']
            end   = start + length

            if annotations is not None:
                q = query(annotations, filename=fname, start=start-buffer, end=end+buffer)
                if len(q) > 0: continue

            if no_overlap and len(df) > 0:
                q = query(df.set_index(df.filename), filename=fname, start=start, end=end)
                if len(q) > 0: continue

            x = {'start':start, 'end':end}
            y = files[files['filename']==fname].iloc[0].to_dict()
            z = {**x, **y}
            df = df.append(z, ignore_index=True)

            if len(df) == num: break

    # sort by filename and offset
    df = df.sort_values(by=['filename','start'], axis=0, ascending=[True,True]).reset_index(drop=True)

    # re-order columns
    col_names = ['filename','start','end']
    if not trim_table:
        names = df.columns.values.tolist()
        for name in col_names: names.remove(name)
        col_names += names

    df = df[col_names]

    df['label'] = label #add label

    # transform to multi-indexing
    df = use_multi_indexing(df, 'sel_id')

    return df


def create_rndm_backgr_selections(files, length, num, annotations=None, no_overlap=False, trim_table=False, buffer=0):
    """ Deprecation warning: create_rndm_backgr_selections is deprecated, use create_rndm_selections instead.
        Create background selections of uniform length, randomly distributed across the 
        data set and not overlapping with any annotations, including those labelled 0.

        The random sampling is performed without regard to already created background 
        selections. Therefore, it is in principle possible that some of the created 
        selections will overlap, although in practice this will only occur with very 
        small probability, unless the number of requested selections (num) is very 
        large and/or the (annotation-free part of) the data set is small in size.

        To avoid any overlap, set the 'no_overlap' to True, but note that this can 
        lead to longer execution times.

        Use the 'buffer' argument to ensure a minimum separation between the background 
        selections and the annotated segments. This can be useful if the annotation 
        start and end times are not always fully accurate.

        Args:
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            num: int
                Number of selections to be created.
            annotations: pandas DataFrame
                Annotation table. Optional.
            no_overlap: bool
                If True, randomly selected segments will have no overlap.
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format.
            buffer: float
                Minimum separation in seconds between the background selections and the annotated segments.
                The default value is zero.

        Returns:
            table_backgr: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ketos.data_handling.selection_table import select
            >>> 
            >>> #Ensure reproducible results by fixing the random number generator seed.
            >>> np.random.seed(3)
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>> print(df)
                filename  start   end  label
            0  file1.wav    7.0   8.1      1
            1  file1.wav    8.5  12.5      0
            2  file1.wav   13.1  14.0      1
            3  file2.wav    2.2   3.1      1
            4  file2.wav    5.8   6.8      1
            5  file2.wav    9.0  13.0      0
            >>>
            >>> #Standardize annotation table format
            >>> df = standardize(df, start_labels_at_1=True)
            >>> print(df)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>>
            >>> #Enter file durations into a pandas DataFrame
            >>> file_dur = pd.DataFrame({'filename':['file1.wav','file2.wav','file3.wav',], 'duration':[18.,20.,15.]})
            >>> 
            >>> #Create randomly sampled background selection with fixed 3.0-s length.
            >>> df_bgr = create_rndm_backgr_selections(annotations=df, files=file_dur, length=3.0, num=12, trim_table=True) 
            >>> print(df_bgr.round(2))
                              start    end  label
            filename  sel_id                     
            file1.wav 0        1.06   4.06      0
                      1        1.31   4.31      0
                      2        2.26   5.26      0
            file2.wav 0       13.56  16.56      0
                      1       14.76  17.76      0
                      2       15.50  18.50      0
                      3       16.16  19.16      0
            file3.wav 0        2.33   5.33      0
                      1        7.29  10.29      0
                      2        7.44  10.44      0
                      3        9.20  12.20      0
                      4       10.94  13.94      0
    """
    warnings.warn("create_rndm_backgr_selections is deprecated and will be removed in a future version. Use create_rndm_selections() instead")
    return create_rndm_selections(files, length, num, label=0, annotations=annotations, no_overlap=no_overlap, trim_table=trim_table, buffer=buffer)

def random_choice(df, siz):
    """ Randomly select a specified number of elements from a table.

        Args:
            df: pandas DataFrame
                Selection table or annotation table
            siz: int
                Number of elements to be selected

        Returns:
            sel: pandas DataFrame
                Reduced table
    """

    name_id = 'sel_id' if 'sel_id' in df.index.names else 'annot_id'
    n = min(siz, len(df))
    idx = np.sort(np.random.choice(np.arange(len(df), dtype=int), size=n, replace=False)).tolist()
    df = df.reset_index()
    df = df.loc[idx]
    df = df.set_index([df.filename, df[name_id]])
    df = df.drop(['filename', name_id], axis=1)
    df = df.sort_index()
    return df

def select_by_segmenting(files, length, annotations=None, step=None,
    pad=True, discard_empty=False, keep_only_empty=False, label_empty=0, 
    avoid_label=None):
    """ Generate a selection table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 
        
        Unlike the :func:`data_handling.selection_table.select` method, selections 
        created by this method are not characterized by a single, integer-valued 
        label, but rather a list of annotations (which can have any length, including zero).

        Therefore, the method returns not one, but two tables: A selection table indexed by 
        filename and segment id, and an annotation table indexed by filename, segment id, 
        and annotation id.

        However, if `keep_only_empty=True` only a selection table is returned. This table 
        has a column named `label` with all entries having the same value, as specified via
        the `label_empty` argument.

        Args:
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            annotations: pandas DataFrame
                Annotation table.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            pad: bool
                If True (default), the last selection window is allowed to extend 
                beyond the endpoint of the audio file.
            discard_empty: bool
                If True, only selection that contain annotations will be used. 
                If False (default), all selections are used.
            keep_only_empty: bool
                If True, only selections *without* any annotations are used, and 
                only the selections table is returned. Default is False. 
            label_empty: int
                Only relevant if keep_only_empty is True. Value to be assigned to
                selections without annotations. Default is 0.
            avoid_label: int or list(int)
                If specified, only selections without annotations with these labels 
                are used.

        Returns:
            sel: pandas DataFrame
                Selection table
            annot: pandas DataFrame
                Annotations table. Only returned if annotations is specified 
                and keep_only_empty is False.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select_by_segmenting, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> annot = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> annot = standardize(annot, start_labels_at_1=True)
            >>> print(annot)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>>
            >>> #Create file table
            >>> files = pd.DataFrame({'filename':['file1.wav', 'file2.wav', 'file3.wav'], 'duration':[11.0, 19.2, 15.1]})
            >>> print(files)
                filename  duration
            0  file1.wav      11.0
            1  file2.wav      19.2
            2  file3.wav      15.1
            >>>
            >>> #Create a selection table by splitting the audio data into segments of 
            >>> #uniform length. The length is set to 10.0 sec and the step size to 5.0 sec.
            >>> sel = select_by_segmenting(files=files, length=10.0, annotations=annot, step=5.0) 
            >>> #Inspect the selection table
            >>> print(sel[0].round(2))
                              start   end
            filename  sel_id             
            file1.wav 0         0.0  10.0
                      1         5.0  15.0
            file2.wav 0         0.0  10.0
                      1         5.0  15.0
                      2        10.0  20.0
            file3.wav 0         0.0  10.0
                      1         5.0  15.0
                      2        10.0  20.0
            >>> #Inspect the annotations
            >>> print(sel[1].round(2))
                                       start   end  label
            filename  sel_id annot_id                    
            file1.wav 0      0           7.0   8.1      2
                             1           8.5  12.5      1
                      1      0           2.0   3.1      2
                             1           3.5   7.5      1
                             2           8.1   9.0      2
                      2      1          -1.5   2.5      1
                             2           3.1   4.0      2
            file2.wav 0      0           2.2   3.1      2
                             1           5.8   6.8      2
                             2           9.0  13.0      1
                      1      1           0.8   1.8      2
                             2           4.0   8.0      1
                      2      2          -1.0   3.0      1
    """
    if step is None:
        step = length

    # check that the annotation table has expected format
    if annotations is not None:
        assert is_standardized(annotations, has_time=True), 'Annotation table appears not to have the expected structure.'
        
        annotations = annotations[annotations.label != -1] #discard annotations with label -1

    # create selections table by segmenting
    sel = segment_files(files, length=length, step=step, pad=pad)

    # max number of segments
    num_segs = sel.index.get_level_values(1).max() + 1

    # create annotation table by segmenting
    if annotations is not None:
        annot = segment_annotations(annotations, num=num_segs, length=length, step=step)

        # get the indices of those selections that have annotations associated with them
        indices = list(set([(a, b) for a, b, c in annot.index.tolist()]))

        if keep_only_empty: 
            sel = sel.loc[~sel.index.isin(indices)].sort_index()
            sel['label'] = label_empty
            return sel

        if discard_empty: 
            sel = sel.loc[indices].sort_index()

        if avoid_label is not None:
            if isinstance(avoid_label, int): avoid_label = [avoid_label]
            annot_avoid = annot[annot.label.isin(avoid_label)]
            indices_avoid = list(set([(a, b) for a, b, c in annot_avoid.index.tolist()]))
            sel = sel.loc[~sel.index.isin(indices_avoid)].sort_index()

        return sel, annot
            
    else:
        return sel

def segment_files(table, length, step=None, pad=True):
    """ Generate a selection table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 

        Args:
            table: pandas DataFrame
                File duration table.
            length: float
                Selection length in seconds.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            pad: bool
                If True (default), the last selection window is allowed to extend 
                beyond the endpoint of the audio file.

        Returns:
            df: pandas DataFrame
                Selection table
    """
    if step is None:
        step = length

    # compute number of segments for each file
    table['num'] = (table['duration'] - length) / step + 1
    table.num = table.num.apply(lambda x: np.maximum(x, 0))
    if pad: 
        table.num = table.num.apply(np.ceil).astype(int)
    else:
        table.num = table.num.apply(np.floor).astype(int)

    df = table.loc[table.index.repeat(table.num)]
    df.set_index(keys=['filename'], inplace=True, append=True)
    df = df.swaplevel()
    df = df.sort_index()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), df.groupby(level=0).cumcount()],
        names=['filename', 'sel_id'])

    df['start'] = df.index.get_level_values(1) * step
    df['end'] = df['start'] + length
    df.drop(columns=['num','duration'], inplace=True)

    return df

def segment_annotations(table, num, length, step=None, compute_overlap=False):
    """ Generate a segmented annotation table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 
        
        Args:
            table: pandas DataFrame
                Annotation table.
            num: int
                Number of segments
            length: float
                Selection length in seconds.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            compute_overlap: bool
                If True, the fractional overlap between the selection window and the annotation 
                will be computed and added as an extra column in the output table. Default is False.

        Returns:
            df: pandas DataFrame
                Annotations table
    """
    if step is None:
        step = length

    segs = []
    for n in range(num):
        # select annotations that overlap with segment
        t1 = n * step
        t2 = t1 + length
        a = table[(table.start < t2) & (table.end > t1)].copy()
        if len(a) > 0:
            # shift and crop annotations
            if compute_overlap:
                a['overlap'] = a.apply(lambda r: fractional_overlap(a=(t1,t2), b=(r.start, r.end)), axis=1)

            a['start'] -= t1 
            a['end'] -= t1 
            a['sel_id'] = n #map to segment
            segs.append(a)

    df = pd.concat(segs)
    df.set_index(keys=['sel_id'], inplace=True, append=True)
    df = df.swaplevel()
    df = df.sort_index()
    return df

def query(selections, annotations=None, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            selections: pandas DataFrame
                Selections table
            annotations: pandas DataFrame
                Annotations table. Optional.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            : pandas DataFrame or tuple(pandas DataFrame, pandas DataFrame)
            Selection table, accompanied by an annotation table if an input 
            annotation table is provided.
    """
    if annotations is None:
        return query_labeled(selections, filename, label, start, end)
    else:
        return query_annotated(selections, annotations, filename, label, start, end)

def query_labeled(table, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            table: pandas DataFrame
                Annotations table or Selections table with a 'label' column.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            df: pandas DataFrame
            Selection table
    """
    df = table
    if filename is not None:
        if isinstance(filename, str): filename = [filename]

        filename = [f for f in filename if f in df.index]

        if len(filename) == 0: 
            return df.iloc[0:0]
        else: 
            df = df.loc[filename]

    if label is not None:
        if not isinstance(label, list):
            label = [label]

        df = df[df.label.isin(label)]

    if start is not None:
        df = df[df.end > start]

    if end is not None:
        df = df[df.start < end]

    return df

def query_annotated(selections, annotations, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            selections: pandas DataFrame
                Selections table.
            annotations: pandas DataFrame
                Annotations table.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            df1,df2: tuple(pandas DataFrame, pandas DataFrame)
                Selection table and annotation table
    """
    df1 = selections
    df2 = annotations

    df1 = query_labeled(df1, filename=filename, start=start, end=end)
    df2 = query_labeled(df2, filename=filename, label=label, start=start, end=end)

    indices = list(set([x[:-1] for x in df2.index.tolist()]))
    df1 = df1.loc[indices].sort_index()

    return df1, df2

def aggregate_duration(table, label=None):
    """ Compute the aggregate duration of the annotations.

        Overlapping segments are only counted once.

        Args:
            table: pandas DataFrame
                Annotations table or Selections table
            label: int or list(int)
                Label(s). Optional

        Returns:
            agg_dur: float
                Aggregate duration in seconds
    """
    df = query_labeled(table=table, label=label)
    df.sort_index(axis='index', level=[0, 1], inplace=True)

    agg_dur = 0
    
    for index, row in df.iterrows():
        filename = index[0]
        dur = row['end'] - row['start']

        if agg_dur == 0 or filename != filename_prev:
            agg_dur += dur
            end_prev = row['end']
        
        else:
            if row['start'] >= end_prev:
                agg_dur += dur
                end_prev = row['end']

            else:
                extend = max(0, row['end'] - end_prev)                
                agg_dur += extend
                end_prev += extend

        filename_prev = filename

    return agg_dur