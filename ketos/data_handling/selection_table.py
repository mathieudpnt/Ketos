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
    A Ketos annotation table always uses two levels of indices, the first index 
    being the filename and the second index an annotation identifier, and always 
    has the column 'label'. 
    For call-level annotations, the table also contains the columns 'start' 
    and 'end', giving the start and end time of the call measured in seconds 
    since the beginning of the file. 
    The table may also contain the columns 'freq_min' and 'freq_max', giving the 
    minimum and maximum frequencies of the call in Hz, but this is not required.    
    The user may add any number of additional columns.
    Note that the table uses two levels of indices, the first index being the 
    filename and the second index an annotation identifier. 
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

        The input table can be passed as a pandas DataFrame or as the filename of a csv file.
        The table may have either a single label per row, in which case unfold_labels should be set 
        to False, or multiple labels per row (e.g. as a comma-separated list of values), in which 
        case unfold_labels should be set to True and label_sep should be specified.

        The table headings are renamed to conform with the ketos standard naming convention, following the 
        name mapping specified by the user. 

        Signal labels are mapped to integers 1,2,3,... while background labels are mapped to 0, 
        and any remaining labels are mapped to -1.

        Note that the standardized output table has two levels of indices, the first index being the 
        filename and the second index the annotation identifier. 

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

    # keep only relevant columns
    if trim_table:
        df = trim(df)

    # check that dataframe has minimum required columns
    mis = missing_columns(df)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)

    if unfold_labels:
        df = unfold(df, sep=label_sep)

    # cast label column to str
    df = df.astype({'label': 'str'})
    # create list of unique labels in input table
    labels = np.sort(np.unique(df['label'].values)).tolist()

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

    # transform to multi-indexing
    df = use_multi_indexing(df, 'annot_id')

    table_std = df
    return table_std, label_dict

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

def trim(table):
    """ Keep only the columns prescribed by the Ketos annotation format.

        Args:
            table: pandas DataFrame
                Annotation table. 

        Returns:
            table: pandas DataFrame
                Annotation table, after removal of columns.
    """
    keep_cols = ['filename', 'label', 'start', 'end', 'freq_min', 'freq_max']
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

def is_standardized(table, has_time=False):
    """ Check if the table has the correct indices and the minimum required columns.

        Args:
            table: pandas DataFrame
                Annotation table. 
            has_time: bool
                Require time information for each annotation, i.e. start and stop times.

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
    return res

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

def select(annotations, length, step=0, min_overlap=0, center=False,\
    discard_long=False, keep_id=False):
    """ Generate a selection table by defining intervals of fixed length around 
        every annotated section of the audio data. Each selection created in this 
        way is chracterized by a single, integer-valued, label.

        The input table must have the standardized Ketos format and contain call-level 
        annotations, see :func:`data_handling.selection_table.standardize`.

        The output table uses two levels of indexing, the first level being the 
        filename and the second level being a selection id.

        The generated selections have uniform length given by the length argument. 
        
        Note that the selections may have negative start times and/or stop times 
        that exceed the file duration.

        Annotations longer than the specified selection length will be cropped, unless the 
        step is set to a value larger than 0.

        Annotations with label -1 are discarded.

        Args:
            annotations: pandas DataFrame
                Input table with call-level annotations.
            length: float
                Selection length in seconds.
            step: float
                Produce multiple instances of the same annotation by shifting the annotation 
                window in steps of length step (in seconds) both forward and backward in 
                time. The default value is 0.
            min_overlap: float
                Minimum required overlap between the generated annotation and the original 
                annotation, expressed as a fraction of length. Only used if step > 0. 
                The requirement is imposed on all annotations (labeled 1,2,3,...) except 
                background annotations (labeled 0) which are always required to have an 
                overlap of 1.0.
            center: bool
                Center annotations. Default is False.
            discard_long: bool
                Discard all annotations longer than the output length. Default is False.
            keep_id: bool
                For each generated selection, include the id of the annotation from which 
                the selection was generated.

        Results:
            table_sel: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> df, label_dict = standardize(df)
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
            >>> #Set the length to 3.0 sec and require a minimum overlap of 
            >>> #0.16*3.0=0.48 sec between selection and annotations.
            >>> #Also, create multiple time-shifted versions of the same selection
            >>> #using a step size of 1.0 sec.     
            >>> df_sel = select(df, length=3.0, step=1.0, min_overlap=0.16, center=True, keep_id=True) 
            >>> print(df_sel.round(2))
                              label  start    end  annot_id
            filename  sel_id                               
            file1.wav 0         2.0   5.05   8.05         0
                      1         1.0   6.00   9.00         1
                      2         2.0   6.05   9.05         0
                      3         1.0   7.00  10.00         1
                      4         2.0   7.05  10.05         0
                      5         1.0   8.00  11.00         1
                      6         1.0   9.00  12.00         1
                      7         1.0  10.00  13.00         1
                      8         1.0  11.00  14.00         1
                      9         2.0  11.05  14.05         2
                      10        1.0  12.00  15.00         1
                      11        2.0  12.05  15.05         2
                      12        2.0  13.05  16.05         2
            file2.wav 0         2.0   0.15   3.15         0
                      1         2.0   1.15   4.15         0
                      2         2.0   2.15   5.15         0
                      3         2.0   3.80   6.80         1
                      4         2.0   4.80   7.80         1
                      5         2.0   5.80   8.80         1
                      6         1.0   6.50   9.50         2
                      7         1.0   7.50  10.50         2
                      8         1.0   8.50  11.50         2
                      9         1.0   9.50  12.50         2
                      10        1.0  10.50  13.50         2
                      11        1.0  11.50  14.50         2
                      12        1.0  12.50  15.50         2
    """
    df = annotations.copy()
    df['annot_id'] = df.index.get_level_values(1)

    # check that input table has expected format
    assert is_standardized(df, has_time=True), 'Annotation table appears not to have the expected structure.'

    # discard annotations with label -1
    df = df[df['label'] != -1]

    # number of annotations
    N = len(df)

    # annotation lengths
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

            if row['label'] == 0:
                ovl = 1
            else:
                ovl = min_overlap
 
            df_shift = time_shift(annot=row, time_ref=t, length=length, min_overlap=ovl, step=step)
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

    table_sel = df
    return table_sel

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
                annotation, expressed as a fraction of the selection length.   

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
    
    row['start_new'] = np.nan
    
    t = time_ref
    t1 = row['start']
    t2 = row['end']

    t_min = t1 - (1 - min_overlap) * length
    t_max = t2 - min_overlap * length

    num_steps_back = int(np.floor((t - t_min) / step))
    num_steps_forw = int(np.floor((t_max - t) / step))

    num_steps = num_steps_back + num_steps_forw
    if num_steps == 0:
        return pd.DataFrame(columns=row.index) #return empty DataFrame

    row['start_new'] = time_ref
    rows_new = [row]

    # step backwards
    for i in range(num_steps_back):
        ri = row.copy()
        ri['start_new'] = t - (i + 1) * step
        rows_new.append(ri)

    # step forwards
    for i in range(num_steps_forw):
        ri = row.copy()
        ri['start_new'] = t + (i + 1) * step
        rows_new.append(ri)

    # create DataFrame
    df = pd.DataFrame(rows_new)

    # sort according to new start time
    df = df.sort_values(by=['start_new'], axis=0, ascending=[True]).reset_index(drop=True)

    return df

def complement(annotations, files):
    """ Create a table listing all segments that have not been annotated (label 0,1,2,3,...) 
        or discarded (label -1).

        The annotation table must conform to the standard Ketos format and 
        contain call-level annotations, see :func:`data_handling.selection_table.standardize`.

        Args:
            annotations: pandas DataFrame
                Annotation table.
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.

        Results:
            table_compl: pandas DataFrame
                Output table.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import complement, standardize
            >>> #Create annotation table and standardize
            >>> df = pd.DataFrame({'filename':['file1.wav', 'file1.wav'], 'label':[1, 2], 'start':[2.0, 7.5], 'end':[3.1, 9.0]})
            >>> df, label_dict = standardize(df)
            >>> #Create file duration table
            >>> dur = pd.DataFrame({'filename':['file1.wav', 'file2.wav', 'file3.wav'], 'duration':[10.0, 20.0, 15.0]})
            >>> #Create complement table
            >>> df_c = complement(df, dur)
            >>> print(df_c.round(2))
                                start   end
            filename  annot_id             
            file1.wav 0           0.0   2.0
                      1           3.1   7.5
                      2           9.0  10.0
            file2.wav 0           0.0  20.0
            file3.wav 0           0.0  15.0
    """   
    df = annotations

    filename, start, end = [], [], []

    for _, ri in files.iterrows():
        fname = ri['filename']
        dur = ri['duration']
        if fname in df.index:
            dfi = df.loc[fname]
            intervals = dfi[['start','end']].values.tolist()
            c = complement_intervals([0, dur], intervals)
        else:
            c = [[0, dur]]

        for x in c:
            filename.append(fname)
            start.append(x[0])
            end.append(x[1])

    # ensure that type is float
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    # fill output DataFrame
    df_out = pd.DataFrame({'filename':filename, 'start':start, 'end':end})

    # use multi-indexing  
    df_out = use_multi_indexing(df_out, 'annot_id')

    return df_out

def create_rndm_backgr_selections(annotations, files, length, num):
    """ Create background selections of uniform length, randomly distributed across the 
        data set and not overlapping with any annotations, including those labelled 0.

        The random sampling is performed without regard to already created background 
        selections. Therefore, it is in principle possible that some of the created 
        selections will overlap, although in practice this will only occur with very 
        small probability, unless the number of requested selections (num) is very 
        large and/or the (annotation-free part of) the data set is small in size.

        Args:
            annotations: pandas DataFrame
                Annotation table.
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            num: int
                Number of selections to be created.

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
            >>> df, label_dict = standardize(df)
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
            >>> file_dur = pd.DataFrame({'filename':['file1.wav','file2.wav','file3.wav',], 'duration':[30.,20.,15.]})
            >>> 
            >>> #Create randomly sampled background selection with fixed 3.0-s length.
            >>> df_bgr = create_rndm_backgr_selections(df, files=file_dur, length=3.0, num=10) 
            >>> print(df_bgr.round(2))
                              start    end
            filename  sel_id              
            file1.wav 0        1.70   4.70
                      1       14.14  17.14
                      2       16.84  19.84
                      3       19.60  22.60
                      4       24.55  27.55
                      5       26.86  29.86
            file2.wav 0       14.18  17.18
            file3.wav 0        2.37   5.37
                      1        8.47  11.47
                      2        8.58  11.58
    """
    # create complement
    c = complement(annotations=annotations, files=files)

    # reset index
    c = c.reset_index()

    # compute lengths, and discard segments shorter than requested length
    c['length'] = c['end'] - c['start'] - length
    c = c[c['length'] >= 0]

    # cumulative length 
    cs = c['length'].cumsum().values.astype(float)
    len_tot = cs[-1]
    cs = np.concatenate(([0],cs))

    # output
    filename, start, end = [], [], []

    # randomply sample
    times = np.random.random_sample(num) * len_tot
    for t in times:
        idx = np.argmax(t < cs) - 1
        row = c.iloc[idx]
        filename.append(row['filename'])
        t1 = row['start'] + t - cs[idx]
        start.append(t1)
        end.append(t1 + length)

    # ensure that type is float
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    # fill DataFrame
    df = pd.DataFrame({'filename':filename, 'start':start, 'end':end})    

    # sort by filename and offset
    df = df.sort_values(by=['filename','start'], axis=0, ascending=[True,True]).reset_index(drop=True)

    # re-order columns
    df = df[['filename','start','end']]

    # transform to multi-indexing
    df = use_multi_indexing(df, 'sel_id')

    return df

def select_by_segmenting(annotations, files, length, step=None,\
    discard_empty=False, pad=True):
    """ Generate a selection table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 
        
        Unlike the :func:`data_handling.selection_table.select` method, selections 
        created by this method are not characterized by a single, integer-valued 
        label, but rather a list of annotations (which can have any length, including zero).

        Therefore, the method returns not one, but two tables: A selection table indexed by 
        filename and segment id, and an annotation table indexed by filename, segment id, 
        and annotation id.

        Args:
            table: pandas DataFrame
                Annotation table.
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            discard_empty: bool
                If True, only selection that contain annotations will be used. 
                If False (default), all selections are used.
            pad: bool
                If True (default), the last selection window is allowed to extend 
                beyond the endpoint of the audio file.

        Returns:
            : tuple(pandas DataFrame, pandas DataFrame)
                Selection table and accompanying annotations table

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select_by_segmenting, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> annot = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> annot, label_dict = standardize(annot)
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
            >>> sel = select_by_segmenting(annotations=annot, files=files, length=10.0, step=5.0) 
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
                             1           8.5  10.0      1
                      1      0           2.0   3.1      2
                             1           3.5   7.5      1
                             2           8.1   9.0      2
                      2      1           0.0   2.5      1
                             2           3.1   4.0      2
            file2.wav 0      0           2.2   3.1      2
                             1           5.8   6.8      2
                             2           9.0  10.0      1
                      1      1           0.8   1.8      2
                             2           4.0   8.0      1
                      2      2           0.0   3.0      1
    """
    if step is None:
        step = length

    # check that the annotation table has expected format
    assert is_standardized(annotations, has_time=True), 'Annotation table appears not to have the expected structure.'

    # discard annotations with label -1
    annotations = annotations[annotations.label != -1]

    # create selections table by segmenting
    sel = segment_files(files, length=length, step=step, pad=pad)

    # max number of segments
    num_segs = sel.index.get_level_values(1).max() + 1

    # create annotation table by segmenting
    annot = segment_annotations(annotations, num=num_segs, length=length, step=step)

    # discard empties
    if discard_empty:
        indices = list(set([(a, b) for a, b, c in annot.index.tolist()]))
        sel = sel.loc[indices].sort_index()

    return (sel, annot)

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

def segment_annotations(table, num, length, step=None):
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
            a['start'] = a['start'].apply(lambda x: max(0, x - t1))
            a['end'] = a['end'].apply(lambda x: min(length, x - t1))
            a['sel_id'] = n #map to segment
            segs.append(a)

    df = pd.concat(segs)
    df.set_index(keys=['sel_id'], inplace=True, append=True)
    df = df.swaplevel()
    df = df.sort_index()
    return df