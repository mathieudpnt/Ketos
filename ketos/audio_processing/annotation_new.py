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

""" Annotation module within the ketos library

    This module provides utilities to handle annotations associated 
    with audio files and spectrograms.

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

def _ensure_multi_index(df):
    """ Ensure the DataFrame has at least two indexing levels.

        Args: 
            df: pandas DataFrame
                Input DataFrame

        Returns: 
            df: pandas DataFrame
                Output DataFrame
    """
    if df.index.nlevels == 1:
        df = pd.concat([df], axis=1, keys=[0]).stack(0).swaplevel(0,1)

    return df

def stack_handlers(handlers):
    """ Create a handler to manage a stack of annotation sets.

        The annotation sets will be indexed in the order they 
        are provided.

        Args:
            handlers: list(AnnotationHandler)
                Annotation handlers

        Returns: 
            handler: AnnotationHandler
                Stacked annotation handler
    """
    dfs = []
    N = len(handlers)
    for i,h in enumerate(handlers):
        df = h.get()
        dfs.append(df)

    df = pd.concat(dfs, sort=False, axis=1, keys=np.arange(N, dtype=int))
    df = df.stack(0)
    df = df.swaplevel(0,1)
    handler = AnnotationHandler(df)
    return handler

class AnnotationHandler():
    """ Class for handling annotations of acoustic data.
    
        An annotation is characterized by 
        
         * start and stop time in seconds 
         * minimum and maximum frequency in Hz (optional)
         * label (integer)
         
        The AnnotationHandler stores annotations in a pandas DataFrame and offers 
        methods to add/get annotations and perform various manipulations such as 
        cropping, shifting, and segmenting.

        Multi-indexing is used for handling several, stacked annotation sets.

        Args:
            df: pandas DataFrame
                Annotations to be passed on to the handler.
                Must contain the columns 'label', 'time_start', and 'time_stop', and 
                optionally also 'freq_min' and 'freq_max'.
    """
    def __init__(self, df=None):
        
        if df is None:
            # initialize empty DataFrame
            self._df = pd.DataFrame(columns=['label', 'time_start', 'time_stop', 'freq_min', 'freq_max'], dtype='float')
            self._df['label'] = pd.Series(dtype='int')
        
        else:
            self._df = df
            self._df = self._df.astype({'label': int})

        # ensure multi-index
        self._df = _ensure_multi_index(self._df)

    def copy(self):
        handler = self.__class__(self._df.copy())
        return handler

    def num_sets(self):
        """ Get number of seperate annotation sets managed by the handler.

            Returns:
                num: int
                    Number of annotation sets
        """
        ids = np.unique(self._df.index.get_level_values(0).values)
        num = len(ids)
        return num

    def num_annotations(self, set_id=None):
        """ Get number of annotations managed by the handler.

            Returns:
                num: int
                    Unique identifier of the annotation set. If None is specified, 
                    the total number of annotations is returned.
        """
        num = len(self.get(set_id=set_id))
        return num

    def get(self, set_id=None):
        """ Get annotations managed by the handler module.
        
            Note: This returns a view (not a copy) of the pandas DataFrame used by 
            the handler module to manage the annotations.

            Args:
                set_id: int
                    Unique identifier of the annotation set. If None is specified, 
                    all annotations are returned.

            Returns:
                ans: pandas DataFrame
                    Annotations 

            Example:
                >>> from ketos.audio_processing.annotation_new import AnnotationHandler
                >>> # Initialize an empty instance of the annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, time_start='1min', time_stop='2min')
                >>> handler.add(label=2, time_start='11min', time_stop='12min')
                >>> # Retrieve the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  time_start  time_stop  freq_min  freq_max
                0      1        60.0      120.0       NaN       NaN
                1      2       660.0      720.0       NaN       NaN
        """
        ans = self._df

        if self.num_sets() == 1:
            ans = ans.loc[0]
        else:
            if set_id is not None:
                ans = ans.loc[set_id]

        return ans

    def _next_index(self, set_id=0):
        """ Get the next available index for the selected annotation set.

            Args:
                set_id: int
                    Unique identifier of the annotation set.

            Returns:
                idx, int
                    Next available index.
        """
        if len(self._df) == 0:
            idx = 0

        else:
            idx = self._df.loc[set_id].index.values[-1] + 1

        return idx

    def _add(self, df, set_id=0):
        """ Add annotations to the handler module.
        
            Args:
                df: pandas DataFrame or dict
                    Annotations stored in a pandas DataFrame or dict. Must have columns/keys 
                    'label', 'time_start', 'time_stop', and optionally also 'freq_min' 
                    and 'freq_max'.
                set_id: int
                    Unique identifier of the annotation set.

            Returns: 
                None
        """
        if isinstance(df, dict):
            df = pd.DataFrame(df, index=pd.Index([0]))
        
        next_index = self._next_index()
        new_indices = pd.Index(np.arange(next_index, next_index + len(df), dtype=int))
        df = df.set_index(new_indices)
        df = _ensure_multi_index(df)
        self._df = pd.concat([self._df, df], sort=False)

        self._df = self._df.astype({'label': 'int'}) #cast label column to int

    def add(self, label=None, time_start=None, time_stop=None, freq_min=None, freq_max=None, df=None, set_id=0):
        """ Add an annotation or a collection of annotations to the handler module.
        
            Individual annotations may be added using the arguments time_range and 
            freq_range.
            
            Groups of annotations may be added by first collecting them in a pandas 
            DataFrame or dictionary and then adding them using the 'df' argument.
        
            Args:
                label: int
                    Integer label.
                time_start: str or float
                    Start time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                time_start: str or float
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
                    'label', 'time_start', 'time_stop', and optionally also 'freq_min' 
                    and 'freq_max'.
                set_id: int
                    Unique identifier of the annotation set.

            Returns: 
                None

            Example:
                >>> from ketos.audio_processing.annotation_new import AnnotationHandler
                >>> # Create an annotation table containing two annotations
                >>> annots = pd.DataFrame({'label':[1,2], 'time_start':[4.,8.], 'time_stop':[6.,12.]})
                >>> # Initialize the annotation handler
                >>> handler = AnnotationHandler(annots)
                >>> # Add a couple of more annotations
                >>> handler.add(label=1, time_start='1min', time_stop='2min')
                >>> handler.add(label=3, time_start='11min', time_stop='12min')
                >>> # Inspect the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  time_start  time_stop  freq_min  freq_max
                0      1         4.0        6.0       NaN       NaN
                1      2         8.0       12.0       NaN       NaN
                2      1        60.0      120.0       NaN       NaN
                3      3       660.0      720.0       NaN       NaN
        """        
        if label is not None:
            assert time_start is not None and time_stop is not None, 'time range must be specified'         
            
            time_start = convert_to_sec(time_start)
            time_stop = convert_to_sec(time_stop)
            
            freq_min = convert_to_Hz(freq_min)
            freq_max = convert_to_Hz(freq_max)
            if freq_min is None:
                freq_min = np.nan
            if freq_max is None:
                freq_max = np.nan

            df = {'label':[label], 'time_start':[time_start], 'time_stop':[time_stop], 'freq_min':[freq_min], 'freq_max':[freq_max]}

        self._add(df, set_id)
        
    def crop(self, time_start=0, time_stop=None, freq_min=None, freq_max=None):
        """ Crop annotations along the time and/or frequency dimension.

            Args:
                time_start: float or str
                    Lower edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                time_stop: float or str
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
                >>> from ketos.audio_processing.annotation_new import AnnotationHandler
                >>> # Initialize an empty annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, time_start='1min', time_stop='2min', freq_min='20Hz', freq_max='200Hz')
                >>> handler.add(label=2, time_start='180s', time_stop='300s', freq_min='60Hz', freq_max='1000Hz')
                >>> # Crop the annotations in time
                >>> handler.crop(time_start='30s', time_stop='4min')
                >>> # Inspect the annotations
                >>> annot = handler.get()
                >>> print(annot)
                   label  time_start  time_stop  freq_min  freq_max
                0      1        30.0       90.0      20.0     200.0
                1      2       150.0      210.0      60.0    1000.0
                >>> # Note how all the start and stop times are shifted by -30 s due to the cropping operation.
                >>> # Crop the annotations in frequency
                >>> handler.crop(freq_min='50Hz')
                >>> annot = handler.get()
                >>> print(annot)
                   label  time_start  time_stop  freq_min  freq_max
                0      1        30.0       90.0      50.0     200.0
                1      2       150.0      210.0      60.0    1000.0
        """
        # convert to desired units
        freq_min = convert_to_Hz(freq_min)
        freq_max = convert_to_Hz(freq_max)
        time_start = convert_to_sec(time_start)
        time_stop = convert_to_sec(time_stop)

        # crop min frequency
        if freq_min is not None:
            self._df['freq_min'][self._df['freq_min'] < freq_min] = freq_min

        # crop max frequency
        if freq_max is not None:
            self._df['freq_max'][self._df['freq_max'] < freq_max] = freq_max

        # crop stop time
        if time_stop is not None:
            dr = -np.maximum(0, self._df['time_stop'] - time_stop)
            self._df['time_stop'] = self._df['time_stop'] + dr

        # crop start time
        if time_start > 0:
            self.shift(-time_start)

        # remove annotations that were fully cropped along the time dimension
        if time_start > 0 or time_stop is not None:
            self._df = self._df[self._df['time_stop'] > self._df['time_start']]

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
        
        self._df['time_start'] = self._df['time_start'] + delta_time
        self._df['time_start'][self._df['time_start'] < 0] = 0
        
        self._df['time_stop'] = self._df['time_stop'] + delta_time
        self._df['time_stop'][self._df['time_stop'] < 0] = 0

        self._df = self._df[self._df['time_stop'] > self._df['time_start']]
        
    def segment(self, num_segs, window_size, step_size=None, offset=0):
        """ Divide the time axis into segments of uniform length, which may or may 
            not be overlapping.

            Args:
                num_segs: int
                    Number of segments
                window_size: float or str
                    Duration of each segment. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                step_size: float or str
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
                ans: dict
                    Dictionary in which the keys are the indices of those 
                    segments that have annotations, and the items are the 
                    annotation handlers.

            Example:
                >>> from ketos.audio_processing.annotation_new import AnnotationHandler
                >>> # Initialize an empty annotation handler
                >>> handler = AnnotationHandler()
                >>> # Add a couple of annotations
                >>> handler.add(label=1, time_start='1s', time_stop='3s')
                >>> handler.add(label=2, time_start='5.2s', time_stop='7.0s')
                >>> # Apply segmentation
                >>> annots = handler.segment(num_segs=10, window_size='1s', step_size='0.8s', time_start='0.1s')
                >>> # Inspect the annotations
        """              
        if step_size is None:
            step_size = window_size
        
        # convert to seconds
        window_size = convert_to_sec(window_size)
        step_size = convert_to_sec(step_size)
        offset = convert_to_sec(offset)

        # crop times
        time_start = offset + step_size * np.arange(num_segs)
        time_stop = time_start + window_size

        # loop over segments
        handlers = []
        for t1,t2 in zip(time_start, time_stop):
            h = self.copy() # create a copy
            h.crop(t1, t2) # crop 
            if h.num_annotations() > 0:
                h._df['offset'] = t1 # index with offset
                handlers.append(h)
        
        # stack handlers
        handler = stack_handlers(handlers)

        print()
        print(handler._df)

        df = handler._df
        df = df.swaplevel(0,1)
        df = df.swaplevel(1,2)

        print(df)

        return None
