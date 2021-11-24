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

""" 'audio.annotation' module within the ketos library

    This module provides utilities to handle annotations associated 
    with waveform and spectrogram objects.

    Contents:
        AnnotationHandler class
"""

import numpy as np
import pandas as pd
from pint import UnitRegistry # SI units

# ignore 'chained assignment' warnings issued by pandas
pd.set_option('mode.chained_assignment', None)

# handling of SI units
ureg = UnitRegistry()
Q_ = ureg.Quantity


def convert_to_sec(x):
    """ Convert a time duration specified as a string with SI units, 
        e.g. "22min" to a float with units of seconds.

        Args:
            x: str
                Time duration specified as a string with SI units, e.g. "22min"

        Returns:
            : float
                Time duration in seconds.
    """
    return convert(x, 's')

def convert_to_Hz(x):
    """ Convert a frequency specified as a string with SI units, 
        e.g. "11kHz" to a float with units of Hz.

        Args:
            x: str
                Frequency specified as a string with SI units, e.g. "11kHz"

        Returns:
            : float
                Frequency in Hz.
    """
    return convert(x, 'Hz')

def convert(x, unit):
    """ Convert a quantity specified as a string with SI units, 
        e.g. "7kg" to a float with the specified unit, e.g. 'g'.

        If the input is not a string, the output will be the same 
        as the input.

        Args:
            x: str
                Value given as a string with SI units, e.g. "11kHz"
            unit: str
                Desired conversion unit "Hz"

        Returns:
            y : float
                Value in specified unit.
    """
    if isinstance(x, str):
        x = Q_(x).m_as(unit)
    
    return x

def add_index_level(df, key=0):
    """ Ensure the DataFrame has at least two indexing levels.

        Args: 
            df: pandas DataFrame
                Input DataFrame

        Returns: 
            df: pandas DataFrame
                Output DataFrame
    """
    df = pd.concat([df], axis=1, keys=[key]).stack(0).swaplevel(0,1)
    return df

def stack_annotations(handlers, keys=None, level=0):
    """ Create a handler to manage a stack of annotation sets.

        The annotation sets will be indexed in the order they 
        are provided.

        Args:
            handlers: list(AnnotationHandler)
                Annotation handlers
            keys: list
                Keys for indexing the sets. If None is specified, 
                the keys are set to 0,1,2,...
            level: int
                Set index level. Default is 0.

        Returns: 
            handler: AnnotationHandler
                Stacked annotation handler
    """
    dfs = []
    squeeze = (level==0)
    N = len(handlers)

    # collect pandas DataFrames from input handlers
    for h in handlers:
        dfs.append(h.get(squeeze=squeeze))

    if keys is None:
        keys = np.arange(N, dtype=int)

    # concatenate and stack
    df = pd.concat(dfs, sort=False, axis=1, keys=keys)
    df = df.stack(0)

    # specify order of indexing levels
    num_lev = df.index.nlevels
    order = np.arange(num_lev - 1, dtype=int)
    order = np.insert(order, level, num_lev - 1)

    # reorder levels and sort indices for faster slicing
    df = df.reorder_levels(order)
    df = df.sort_index()

    # create stacked annotation handler
    handler = AnnotationHandler(df)
    return handler

class AnnotationHandler():
    """ Class for handling annotations of acoustic data.
    
        An annotation is characterized by 
        
         * start and end time in seconds 
         * minimum and maximum frequency in Hz (optional)
         * label (integer)
         
        The AnnotationHandler stores annotations in a pandas DataFrame and offers 
        methods to add/get annotations and perform various manipulations such as 
        cropping, shifting, and segmenting.

        Multiple levels of indexing is used for handling several, stacked annotation sets:

            * level 0: annotation set
            * level 1: individual annotation

        Args:
            df: pandas DataFrame
                Annotations to be passed on to the handler.
                Must contain the columns 'label', 'start', and 'end', and 
                optionally also 'freq_min' and 'freq_max'.
    """
    def __init__(self, df=None):
        
        if df is None:
            # initialize empty DataFrame
            self._df = pd.DataFrame(columns=['label', 'start', 'end', 'freq_min', 'freq_max'], dtype='float')
            self._df['label'] = pd.Series(dtype='int')
        
        else:
            self._df = df
            self._df = self._df.astype({'label': int})

        # ensure multi-index
        if self._df.index.nlevels == 1:
            self._df = add_index_level(self._df)

    def copy(self):
        handler = self.__class__(self._df.copy())
        return handler

    def set_ids(self):
        """ Get the IDs of the annotation subsets managed by the handler.

            Returns:
                : numpy array
                    IDs of the annotation sets
        """
        return np.unique(self._df.index.get_level_values(0).values)

    def num_sets(self):
        """ Get number of annotation subsets managed by the handler.

            Returns:
                num: int
                    Number of annotation sets
        """
        num = len(self.set_ids())
        return num

    def num_annotations(self, id=None):
        """ Get number of annotations managed by the handler.

            Returns:
                num: int or tuple
                    Unique identifier of the annotation set. If None is specified, 
                    the total number of annotations is returned.
        """
        num = len(self.get(id=id))
        return num

    def get(self, label=None, id=None, squeeze=True, drop_freq=False, key_error=False, trim_table=False):
        """ Get annotations managed by the handler module.
        
            Note: This returns a view (not a copy) of the pandas DataFrame used by 
            the handler module to manage the annotations.

            Args:
                label: int or list(int)
                    Get only annotations with this label
                id: int or tuple
                    Unique identifier of the annotation subset. If None is specified, 
                    all annotations are returned.
                squeeze: bool
                    If the handler is managing a single annotation set, drop the 0th-level 
                    index. Default is True. 
                drop_freq: bool
                    Drop the frequency columns.
                key_error: bool
                    If set to True, return error if the specified annotation set does not 
                    exist. If set to False, return None. Default is False.  
                trim_table: bool
                    Keep only the columns prescribed by the Ketos annotation format: filename,start,end,label,freq_min,freq_max
                    Discard any other columns. Default is False.

            Returns:
                ans: pandas DataFrame
                    Annotations 

            Example:
                >>> from ketos.audio.annotation import AnnotationHandler
                >>> # Initialize an empty instance of the annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, start='1min', end='2min')
                >>> handler.add(label=2, start='11min', end='12min')
                >>> # Retrieve the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  start    end  freq_min  freq_max
                0      1   60.0  120.0       NaN       NaN
                1      2  660.0  720.0       NaN       NaN
                >>> # Retrieve only annotations with label 2
                >>> annot = handler.get(label=2)
                >>> print(annot)
                   label  start    end  freq_min  freq_max
                1      2  660.0  720.0       NaN       NaN
        """
        ans = self._df

        if self.num_sets() == 1 and squeeze:
            ans = ans.loc[self.set_ids()[0]]

        if id is not None:
            if not key_error and id not in ans.index: return None
            if len(ans) > 1: ans = ans.loc[id]

        # select label(s)
        if label is not None:
            if not isinstance(label, list):
                label = [label]

            ans = ans[ans.label.isin(label)]

        # ensure correct ordering of columns
        cols = ['label', 'start', 'end', 'freq_min', 'freq_max']

        # keep extra columns
        if not trim_table:
            extra_cols = np.copy(ans.columns.values)
            extra_cols = list(extra_cols[~np.isin(extra_cols, cols)])
            cols += extra_cols

        ans = ans[cols]

        # drop freq columns
        if drop_freq: 
            ans.drop(columns=['freq_min', 'freq_max'], inplace=True, errors='ignore')

        return ans

    def _next_index(self, id=0):
        """ Get the next available index for the selected annotation set.

            Args:
                id: int or tuple
                    Unique identifier of the annotation subset.

            Returns:
                idx, int
                    Next available index.
        """
        if len(self._df) == 0:
            idx = 0

        else:
            if id in self._df.index:
                idx = self._df.loc[id].index.values[-1] + 1
            else:
                idx = 0

        return idx

    def _add(self, df, id=0):
        """ Add annotations to the handler module.
        
            Args:
                df: pandas DataFrame or dict
                    Annotations stored in a pandas DataFrame or dict. Must have columns/keys 
                    'label', 'start', 'end', and optionally also 'freq_min' 
                    and 'freq_max'.
                id: int or tuple
                    Unique identifier of the annotation subset.

            Returns: 
                None
        """
        if isinstance(df, dict):
            if isinstance(df['label'], list):
                df = pd.DataFrame(df)
            else:
                df = pd.DataFrame(df, index=pd.Index([0]))
        
        next_index = self._next_index(id)
        new_indices = pd.Index(np.arange(next_index, next_index + len(df), dtype=int))
        df = df.set_index(new_indices)

        if df.index.nlevels == 1:
            df = add_index_level(df, key=id)

        self._df = pd.concat([self._df, df], sort=False)

        self._df = self._df.astype({'label': 'int'}) #cast label column to int

    def add(self, label=None, start=None, end=None, freq_min=None, freq_max=None, df=None, id=0):
        """ Add an annotation or a collection of annotations to the handler module.
        
            Individual annotations may be added using the arguments start, end, freq_min, 
            and freq_max.
            
            Groups of annotations may be added by first collecting them in a pandas 
            DataFrame or dictionary and then adding them using the 'df' argument.
        
            Args:
                label: int
                    Integer label.
                start: str or float
                    Start time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                start: str or float
                    Stop time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                freq_min: str or float
                    Lower frequency. Can be specified either as a float, in which case the 
                    unit will be assumed to be Hz, or as a string with an SI unit, 
                    for example, '3.1kHz'.
                freq_max: str or float
                    Upper frequency. Can be specified either as a float, in which case the 
                    unit will be assumed to be Hz, or as a string with an SI unit, 
                    for example, '3.1kHz'.
                df: pandas DataFrame or dict
                    Annotations stored in a pandas DataFrame or dict. Must have columns/keys 
                    'label', 'start', 'end', and optionally also 'freq_min' 
                    and 'freq_max'.
                id: int or tuple
                    Unique identifier of the annotation subset.

            Returns: 
                None

            Example:
                >>> from ketos.audio.annotation import AnnotationHandler
                >>> # Create an annotation table containing two annotations
                >>> annots = pd.DataFrame({'label':[1,2], 'start':[4.,8.], 'end':[6.,12.]})
                >>> # Initialize the annotation handler
                >>> handler = AnnotationHandler(annots)
                >>> # Add a couple of more annotations
                >>> handler.add(label=1, start='1min', end='2min')
                >>> handler.add(label=3, start='11min', end='12min')
                >>> # Inspect the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  start    end  freq_min  freq_max
                0      1    4.0    6.0       NaN       NaN
                1      2    8.0   12.0       NaN       NaN
                2      1   60.0  120.0       NaN       NaN
                3      3  660.0  720.0       NaN       NaN
        """   
        assert label is not None or df is not None, "At least one of the arguments 'label' and 'df' must be specified."

        if label is not None:
            assert start is not None and end is not None, 'time range must be specified'         
            
            start = convert_to_sec(start)
            end = convert_to_sec(end)
            
            freq_min = convert_to_Hz(freq_min)
            freq_max = convert_to_Hz(freq_max)
            if freq_min is None:
                freq_min = np.nan
            if freq_max is None:
                freq_max = np.nan

            df = {'label':[label], 'start':[start], 'end':[end], 'freq_min':[freq_min], 'freq_max':[freq_max]}

        self._add(df, id)
        
    def crop(self, start=0, end=None, freq_min=None, freq_max=None, make_copy=False):
        """ Crop annotations along the time and/or frequency dimension.

            Args:
                start: float or str
                    Lower edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                end: float or str
                    Upper edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                freq_min: float or str
                    Lower edge of frequency cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be Hz, 
                    or as a string with an SI unit, for example, '3.1kHz'
                freq_max: float or str
                    Upper edge of frequency cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be Hz, 
                    or as a string with an SI unit, for example, '3.1kHz'
        
            Returns: 
                None

            Example:
                >>> from ketos.audio.annotation import AnnotationHandler
                >>> # Initialize an empty annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, start='1min', end='2min', freq_min='20Hz', freq_max='200Hz')
                >>> handler.add(label=2, start='180s', end='300s', freq_min='60Hz', freq_max='1000Hz')
                >>> # Crop the annotations in time
                >>> handler.crop(start='30s', end='4min')
                >>> # Inspect the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  start    end  freq_min  freq_max
                0      1   30.0   90.0      20.0     200.0
                1      2  150.0  210.0      60.0    1000.0
                >>> # Note how all the start and stop times are shifted by -30 s due to the cropping operation.
                >>> # Crop the annotations in frequency
                >>> handler.crop(freq_min='50Hz')
                >>> annot = handler.get()
                >>> print(annot)
                   label  start    end  freq_min  freq_max
                0      1   30.0   90.0      50.0     200.0
                1      2  150.0  210.0      60.0    1000.0
        """
        # convert to desired units
        freq_min = convert_to_Hz(freq_min)
        freq_max = convert_to_Hz(freq_max)
        start = convert_to_sec(start)
        end = convert_to_sec(end)

        # crop min frequency
        if freq_min is not None:
            self._df['freq_min'][self._df['freq_min'] < freq_min] = freq_min

        # crop max frequency
        if freq_max is not None:
            self._df['freq_max'][self._df['freq_max'] > freq_max] = freq_max

        # crop stop time
        if end is not None:
            dr = -np.maximum(0, self._df['end'] - end)
            self._df['end'] = self._df['end'] + dr

        # crop start time
        if start is not None and start > 0:
            self.shift(-start)

        # remove annotations that were fully cropped along the time dimension
        if (start is not None and start > 0) or end is not None:
            self._df = self._df[self._df['end'] > self._df['start']]

        # remove annotations that were fully cropped along the frequency dimension
        if freq_min is not None or freq_max is not None:
            self._df = self._df[(self._df['freq_max'] > self._df['freq_min'])]
            
    def shift(self, delta_time=0):
        """ Shift all annotations by a fixed amount along the time dimension.

            If the shift places some of the annotations (partially) before time zero, 
            these annotations are removed or cropped.

            Args:
                delta_time: float or str
                    Amount by which annotations will be shifted. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'

            Example:
        """      
        delta_time = convert_to_sec(delta_time)
        
        self._df['start'] = self._df['start'] + delta_time
        self._df['start'][self._df['start'] < 0] = 0
        
        self._df['end'] = self._df['end'] + delta_time
        self._df['end'][self._df['end'] < 0] = 0

        self._df = self._df[self._df['end'] > self._df['start']]
        
    def segment(self, num_segs, window, step=None, offset=0):
        """ Divide the time axis into segments of uniform length, which may or may 
            not be overlapping.

            Args:
                num_segs: int
                    Number of segments
                window: float or str
                    Duration of each segment. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                step: float or str
                    Step size. Can be specified either as a float, in which 
                    case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'.
                    If no value is specified, the step size is set equal to 
                    the window size, implying non-overlapping segments.
                offset: float or str
                    Start time for the first segment. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'.
                    Negative times are permitted. 
                    
            Returns:
                ans: AnnotationHandler
                    Stacked annotation handler with three levels of indexing where 
                        * level 0: annotation set
                        * level 1: segment
                        * level 2: individual annotation

            Example:
                >>> from ketos.audio.annotation import AnnotationHandler
                >>> # Initialize an empty annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, start='1s', end='3s')
                >>> handler.add(label=2, start='5.2s', end='7.0s')
                >>> # Apply segmentation
                >>> handler = handler.segment(num_segs=10, window='1s', step='0.8s', offset='0.1s')
                >>> # Inspect the annotations
                >>> annots = handler.get(drop_freq=True)
                >>> print(annots)
                     label  start  end
                0 0      1    0.9  1.0
                1 0      1    0.1  1.0
                2 0      1    0.0  1.0
                3 0      1    0.0  0.5
                6 1      2    0.3  1.0
                7 1      2    0.0  1.0
                8 1      2    0.0  0.5
                >>> # Note the double index, where the first index refers to the segment 
                >>> # while the second index referes to the original annotation.
                >>> # We can get the annotations for a single segment like this,
                >>> annots3 = handler.get(id=3, drop_freq=True)
                >>> print(annots3)
                   label  start  end
                0      1    0.0  0.5
                >>> # If we attempt to retrieve annotations for a segment that does not 
                >>> # have any annotations, we get None,
                >>> annots4 = handler.get(id=4, drop_freq=True)
                >>> print(annots4)
                None
        """              
        if step is None:
            step = window
        
        # convert to seconds
        window = convert_to_sec(window)
        step = convert_to_sec(step)
        offset = convert_to_sec(offset)

        # crop times
        start = offset + step * np.arange(num_segs)
        end = start + window

        # loop over segments
        handlers, keys = [], []
        for i,(t1,t2) in enumerate(zip(start, end)):
            h = self.copy() # create a copy
            h.crop(t1, t2) # crop
            if h.num_annotations() > 0:
                handlers.append(h)
                keys.append(i)

        # stack handlers
        handler = stack_annotations(handlers, keys, level=1)

        return handler
