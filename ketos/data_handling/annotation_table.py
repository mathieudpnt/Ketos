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
from ketos.utils import str_is_int, complement_intervals


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
            signal_labels: list, or list of lists
                Labels of interest. Will be mapped to 1,2,3,...
                Several labels can be mapped to the same integer by using nested lists. For example, 
                signal_labels=[A,[B,C]] would result in A being mapped to 1 and B and C both being mapped 
                to 2.
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
    backgr_labels = cast_to_str(backgr_labels)
    signal_labels, signal_labels_flat = cast_to_str(signal_labels, nested=True)

    # separate out background labels, if any
    for x in backgr_labels:
        assert x in labels, 'label {0} not found in input table'.format(x)
    
    # discard remaining labels
    discard_labels = [x for x in labels if x not in signal_labels_flat and x not in backgr_labels]

    # create label dictionary and apply to label column in DataFrame
    _label_dict = create_label_dict(signal_labels, backgr_labels, discard_labels)
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

def create_label_dict(signal_labels, backgr_labels, discard_labels):
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

        Returns:
            label_dict: dict
                Dict that maps old labels to new labels.
    """
    label_dict = dict()    
    for l in discard_labels: label_dict[l] = -1
    for l in backgr_labels: label_dict[l] = 0
    num = 1
    for l in signal_labels:
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

def create_ml_table(table, annot_len, step_size=0, overlap=0, center=False,\
    discard_long=False, keep_index=False):
    """ Generate an annotation table suitable for training/testing a machine-learning model.

        The input table must have the standardized Ketos format and contain call-level 
        annotations, see :func:`data_handling.annotation_table.standardize`.

        The generated annotations have uniform length given by the annot_len argument. 
        
        Note that the generated annotations may have negative start times and/or stop times 
        that exceed the file duration.

        Annotations longer than the specified length will be cropped, unless the step_size 
        is set to a value larger than 0.

        Annotations with label -1 are discarded.

        Args:
            table: pandas DataFrame
                Input table with call-level annotations.
            annot_len: float
                Output annotation length in seconds.
            step_size: float
                Produce multiple instances of the same annotation by shifting the annotation 
                window in steps of length step_size (in seconds) both forward and backward in 
                time. The default value is 0.
            overlap: float
                Minimum required overlap between the generated annotation and the original 
                annotation, expressed as a fraction of annot_len. Only used if step_size > 0. 
                The requirement is imposed on all annotations (labeled 1,2,3,...) except 
                background annotations (labeled 0) which are always required to have an 
                overlap of 1.0.
            center: bool
                Center annotations. Default is False.
            discard_long: bool
                Discard all annotations longer than the output length. Default is False.
            keep_index: bool
                For each generated annotation, include the index of the original annotation 
                in the input table from which the new annotation was generated.

        Results:
            table_ml: pandas DataFrame
                Output annotation table.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.annotation_table import create_ml_table
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>> print(df)
                filename  time_start  time_stop  label
            0  file1.wav         7.0        8.1      1
            1  file1.wav         8.5       12.5      0
            2  file1.wav        13.1       14.0      1
            3  file2.wav         2.2        3.1      1
            4  file2.wav         5.8        6.8      1
            5  file2.wav         9.0       13.0      0
            >>>
            >>> #Create a table with fixed-length annotations, suitable for 
            >>> #building training/test data for a Machine Learning model.
            >>> #Set the length to 3.0 sec and require a minimum overlap of 
            >>> #0.16*3.0=0.48 sec between generated and original annotations.
            >>> #Also, create multiple time-shifted versions of the same annotation
            >>> #using a step size of 1.0 sec both backward and forward in time.     
            >>> df_ml = create_ml_table(df, annot_len=3.0, step_size=1.0, overlap=0.16, center=True, keep_index=True) 
            >>> print(df_ml.round(2))
                index   filename label  time_start  time_stop orig_index
            0       0  file1.wav     1        5.05       8.05          0
            1       1  file1.wav     1        6.05       9.05          0
            2       2  file1.wav     1        7.05      10.05          0
            3       3  file1.wav     0        9.00      12.00          1
            4       4  file1.wav     1       11.05      14.05          2
            5       5  file1.wav     1       12.05      15.05          2
            6       6  file1.wav     1       13.05      16.05          2
            7       7  file2.wav     1        0.15       3.15          3
            8       8  file2.wav     1        1.15       4.15          3
            9       9  file2.wav     1        2.15       5.15          3
            10     10  file2.wav     1        3.80       6.80          4
            11     11  file2.wav     1        4.80       7.80          4
            12     12  file2.wav     1        5.80       8.80          4
            13     13  file2.wav     0        9.50      12.50          5
    """
    df = table.copy()
    df['orig_index'] = df.index.copy()
        
    # check that input table has expected format
    mis = missing_columns(df, has_time=True)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)

    # discard annotations with label -1
    df = df[df['label'] != -1]

    # number of annotations
    N = len(df)

    # annotation lengths
    df['length'] = df['time_stop'] - df['time_start']

    # discard annotations longer than the requested length
    if discard_long:
        df = df[df['length'] <= annot_len]

    # alignment of new annotations relative to original ones
    if center:
        df['time_start_new'] = df['time_start'] + 0.5 * (df['length'] - annot_len)
    else:
        df['time_start_new'] = df['time_start'] + np.random.random_sample(N) * (df['length'] - annot_len)

    # create multiple time-shited instances of every annotation
    if step_size > 0:
        df_tmp = df.copy()
        for _,row in df.iterrows():
            t = row['time_start_new']
            if row['label'] == 0:
                ovl = 1
            else:
                ovl = overlap
 
            df_shift = time_shift(annot=row, time_ref=t, annot_len=annot_len, overlap=ovl, step_size=step_size)
            df_tmp = pd.concat([df_tmp, df_shift])

        df = df_tmp.sort_values(by=['orig_index','time_start_new'], axis=0, ascending=[True,True]).reset_index(drop=True)

    # drop old/temporary columns, and rename others
    df = df.drop(['time_start', 'time_stop', 'length'], axis=1)
    df = df.rename(columns={"time_start_new": "time_start"})
    df['time_stop'] = df['time_start'] + annot_len

    # keep old index
    if not keep_index:
        df = df.drop(['orig_index'], axis=1)
    else:
        # re-order columns so orig_index appears last
        cols = df.columns.values.tolist()
        p = cols.index('orig_index')
        cols_new = cols[:p] + cols[p+1:] + ['orig_index']
        df = df[cols_new]

    df = df.reset_index()
    return df

def time_shift(annot, time_ref, annot_len, step_size, overlap):
    """ Create multiple instances of the same annotation by stepping in time, both 
        forward and backward.

        Args:
            annot: pandas Series
                Reference annotation. Must contain the labels 'time_start' and 'time_stop'.
            time_ref: float
                Reference time used as starting point for the stepping.
            annot_len: float
                Output annotation length in seconds.
            step_size: float
                Produce multiple instances of the same annotation by shifting the annotation 
                window in steps of length step_size (in seconds) both forward and backward in 
                time. The default value is 0.
            overlap: float
                Minimum required overlap between the generated annotation and the original 
                annotation, expressed as a fraction of annot_len.   

        Results:
            df: pandas DataFrame
                Output annotation table.
    """
    row = annot.copy()
    row['time_start_new'] = np.nan
    
    t = time_ref
    t1 = row['time_start']
    t2 = row['time_stop']

    t_min = t1 - (1 - overlap) * annot_len
    t_max = t2 - overlap * annot_len

    num_steps_back = int(np.floor((t - t_min) / step_size))
    num_steps_forw = int(np.floor((t_max - t) / step_size))

    num_steps = num_steps_back + num_steps_forw
    if num_steps == 0:
        return pd.DataFrame(columns=row.index) #return empty DataFrame

    rows_new = []

    # step backwards
    for i in range(num_steps_back):
        ri = row.copy()
        ri['time_start_new'] = t - (i + 1) * step_size
        rows_new.append(ri)

    # step forwards
    for i in range(num_steps_forw):
        ri = row.copy()
        ri['time_start_new'] = t + (i + 1) * step_size
        rows_new.append(ri)

    # create DataFrame
    df = pd.DataFrame(rows_new)

    return df

def complement(table, file_duration):
    """ Create a table listing all segments that have not been annotated (label 0,1,2,3,...) 
        or discarded (label -1).

        The annotation table must conform to the standard Ketos format and 
        contain call-level annotations, see :func:`data_handling.annotation_table.standardize`.

        Args:
            table: pandas DataFrame
                Annotation table.
            file_duration: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.

        Results:
            table_compl: pandas DataFrame
                Output table.
    """   
    df = table

    filename, time_start, time_stop = [], [], []

    for _, ri in file_duration.iterrows():
        fname = ri['filename']
        dur = ri['duration']
        dfi = df[df['filename']==fname]
        intervals = dfi[['time_start','time_stop']].values.tolist()
        c = complement_intervals([0, dur], intervals)

        for x in c:
            filename.append(fname)
            time_start.append(x[0])
            time_stop.append(x[1])

    df_out = pd.DataFrame({'filename':filename, 'time_start':time_start, 'time_stop':time_stop})
    return df_out

def create_rndm_backgr(table, file_duration, annot_len, num):
    """ Create background annotations of uniform length, randomly distributed across the 
        data set and not overlapping with any other annotations.

        The random sampling is performed without regard to already created background 
        annotations. Therefore, it is in principle possible that some of the created 
        annotations will overlap, although in practice this will only occur with very 
        small probability, unless the number of requested annotations (num) is very 
        large and/or the (annotation-free part of) the data set is small in size.

        Args:
            table: pandas DataFrame
                Annotation table.
            file_duration: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            annot_len: float
                Output annotation length in seconds.
            num: int
                Number of annotations to be created.

        Returns:
            table_backgr: pandas DataFrame
                Output annotation table.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ketos.data_handling.annotation_table import create_ml_table
            >>> 
            >>> #Ensure reproducible results by fixing the random number generator seed.
            >>> np.random.seed(3)
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>> print(df)
                filename  time_start  time_stop  label
            0  file1.wav         7.0        8.1      1
            1  file1.wav         8.5       12.5      0
            2  file1.wav        13.1       14.0      1
            3  file2.wav         2.2        3.1      1
            4  file2.wav         5.8        6.8      1
            5  file2.wav         9.0       13.0      0
            >>>
            >>> #Enter file durations into a pandas DataFrame
            >>> file_dur = pd.DataFrame({'filename':['file1.wav','file2.wav'], 'duration':[30.,20.]})
            >>> 
            >>> #Create randomly sampled background annotations with fixed 3.0-s length.
            >>> df_bgr = create_rndm_backgr(df, file_duration=file_dur, annot_len=3.0, num=5) 
            >>> print(df_bgr.round(2))
                filename  time_start  time_stop
            0  file1.wav       21.57      24.57
            1  file1.wav       24.87      27.87
            2  file1.wav       16.11      19.11
            3  file1.wav       20.73      23.73
            4  file2.wav       14.75      17.75
    """
    # create complement
    c = complement(table=table, file_duration=file_duration)

    # compute lengths, and discard segments shorter than requested length
    c['length'] = c['time_stop'] - c['time_start'] - annot_len
    c = c[c['length'] >= 0]

    # cumulative length 
    cs = c['length'].cumsum().values.astype(float)
    len_tot = cs[-1]
    cs = np.concatenate(([0],cs))

    # output
    filename, time_start, time_stop = [], [], []

    # randomply sample
    times = np.random.random_sample(num) * len_tot
    for t in times:
        idx = np.argmax(t < cs) - 1
        row = c.iloc[idx]
        filename.append(row['filename'])
        t1 = row['time_start'] + t - cs[idx]
        t2 = t1 + annot_len
        time_start.append(t1)
        time_stop.append(t2)

    # ensure that type is float
    time_start = np.array(time_start, dtype=float)
    time_stop = np.array(time_stop, dtype=float)

    df = pd.DataFrame({'filename':filename, 'time_start':time_start, 'time_stop':time_stop})    

    return df