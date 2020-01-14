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


class AnnotationHandler():
    """ Class for handling annotations of acoustic data.
    
        An annotation is characterized by 
        
         * start and stop time in seconds 
         * minimum and maximum frequency in Hz (optional)
         * label (integer)
         
        The AnnotationHandler stores annotations in a pandas DataFrame and offers 
        methods to add/get annotations and perform various manipulations such as 
        cropping, shifting, and segmenting.

        Args:
            df: pandas DataFrame
                Annotations to be passed on to the handler.
                Must contain the columns 'label', 'time_start', and 'time_stop', and 
                optionally also 'freq_min' and 'freq_max'.
    """
    def __init__(self, df=None):
        
        # columns that will be returned to the user via the 'get' method
        self._get_cols = ['label', 'time_start', 'time_stop', 'freq_min', 'freq_max']
        
        # initialize empty DataFrame
        self._df = pd.DataFrame(columns=['label', 'time_start', 'time_stop', 'freq_min', 'freq_max', '_dl', '_dr'], dtype='float')
        self._df['label'] = pd.Series(dtype='int')

        # note: the columns '_dl' and '_dr' are used to keep track of the amount of cropping 
        # applied to every annotation from the left (_dl) and right (_dr) in the most recent 
        # manipulation.

        # fill DataFrame
        if df is not None:
            self._add(df)
        
    def get(self):
        """ Get all annotations being managed by the handler module.
        
            Note: This returns a view (not a copy) of the pandas DataFrame used by 
            the handler module to manage the annotations.

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
        ans = self._df[self._get_cols]
        return ans
    
    def _add(self, df):
        """ Add annotations to the handler module.
        
            Args:
                df: pandas DataFrame
                    Must contain the columns 'label', 'time_start', and 'time_stop', and 
                    optionally also 'freq_min' and 'freq_max'.

            Returns: 
                None
        """
        self._df = self._df.append(df, sort=False, ignore_index=True)
        self._df['_dl'][np.isnan(self._df['_dl'])] = 0
        self._df['_dr'][np.isnan(self._df['_dr'])] = 0
        self._df = self._df.astype({'label': 'int'}) #cast label column to int

    def add(self, label=None, time_start=None, time_stop=None, freq_min=None, freq_max=None, df=None):
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
                    DataFrame with columns 'label', 'time_start', 'time_stop', and optionally 
                    also 'freq_min' and 'freq_max', containing one or several annotations. 

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

            df = {'label':label, 'time_start':time_start, 'time_stop':time_stop, 'freq_min':freq_min, 'freq_max':freq_max}

        self._add(df)
        
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
            self._df['_dr'] = dr
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
        self._df['_dl'] = np.maximum(0, -self._df['time_start'])
        self._df['time_start'][self._df['time_start'] < 0] = 0
        
        self._df['time_stop'] = self._df['time_stop'] + delta_time
        self._df['time_stop'][self._df['time_stop'] < 0] = 0

        self._df = self._df[self._df['time_stop'] > self._df['time_start']]
        
    def segment(self, num_segs, window_size, step_size=None, time_start=0):
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
                time_start: float or str
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
        time_start = convert_to_sec(time_start)

        # find overlap between annotations and segments
        iA = np.maximum(0, np.ceil((self._df['time_start'] - time_start - window_size) / step_size))
        iA = iA.astype(int, copy=False)
        iB = np.minimum(num_segs - 1, np.floor((self._df['time_stop'] - time_start) / step_size))
        iB = iB.astype(int, copy=False)

        # loop over annotations
        ans = {}
        for index, row in self._df.iterrows():
            
            # segments that overlap with this annotation
            i = np.arange(iA[index], iB[index] + 1)
            a = time_start + i * step_size
            b = a + window_size         
            n = len(i)

            # left and right cuts
            dl = np.maximum(0, a - row['time_start'])
            dr = -np.maximum(0, row['time_stop'] - b)

            # crop/shift
            b = np.minimum(b, row['time_stop']) - a
            a = np.maximum(a, row['time_start']) - a

            # create/fill annotation handlers for each segment
            for j in range(n):
                
                if b[j] <= a[j]:
                    continue

                r = row.copy()
                r['time_start'] = a[j]
                r['time_stop'] = b[j]
                r['_dl'] = dl[j]
                r['_dr'] = dr[j]
                idx = i[j]
                if idx in ans:
                    ans[idx].add(df=r)
                else:
                    ans[idx] = self.__class__(df=r)

        return ans
