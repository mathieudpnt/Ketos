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
        AnnotationHandler class:
        AudioSourceHandler class 
"""

import numpy as np
import pandas as pd
from pint import UnitRegistry # SI units

# ignore 'chained assignment' warnings issued by pandas
pd.set_option('mode.chained_assignment', None)

# handling of SI units
ureg = UnitRegistry()
Q_ = ureg.Quantity


def _convert_to_sec(x):
    """ Convert a time interval specified as a string with SI units, 
        e.g. "22min" to a float with units of seconds.
    """
    return _convert(x, 's')

def _convert_to_Hz(x):
    """ Convert a frequency specified as a string with SI units, 
        e.g. "11kHz" to a float with units of Hz.
    """
    return _convert(x, 'Hz')

def _convert(x, u):
    """ Convert a quantity specified as a string with SI units, 
        e.g. "7kg" to a float with the specified unit, e.g. 'g'.

        If the input is not a string, the output will be the same 
        as the input.
    """
    if isinstance(x, str):
        x = Q_(x).m_as(u)
    
    return x


class AnnotationHandler():
    """ Class for handling annotations of acoustic data.
    
        An annotation is characterized by 
        
         * start and stop time in seconds 
         * low and high frequency in Hz (optional)
         * label
         
        The AnnotationHandler stores annotations in a pandas DataFrame and offers 
        methods to add/get annotations and perform various manipulations such as 
        cropping, shifting, and segmenting.

        Args:
            label_type: str
                Label type. Default is int.
            df: pandas DataFrame
                DataFrame with columns 'start', 'stop', 'label' and optionally also 'low' 
                and 'high', containing annotations that will be handed over to 
                the annotation handler.
    """
    def __init__(self, label_type='int', df=None):
        
        # columns that will be returned to the user via the 'get' method
        self._get_cols = ['start', 'stop', 'low', 'high', 'label']
        
        # initialize empty DataFrame
        self._df = pd.DataFrame(columns=['start', 'stop', 'low', 'high', '_dl', '_dr'], dtype='float')
        self._df['label'] = pd.Series(dtype=label_type)

        # note: the columns '_dl' and '_dr' are used to keep track of the amount of cropping 
        # applied to every annotation from the left (_dl) and right (_dr) in the most recent 
        # manipulation.

        # fill DataFrame
        if df is not None:
            self._add(df)
        
    def get(self):
        """ Get all annotations being managed by the handler module.
        
            returns:
                ans: pandas DataFrame
                    Annotations 
        """        
        res = self._df[self._get_cols]
        return res
    
    def _add(self, df):
        """ Add annotations to the handler module.
        
            Args:
                df: pandas DataFrame
                    DataFrame with columns 'start', 'stop', 'label' and optionally also 'low' 
                    and 'high', containing one or several annotations. 
        """        
        self._df = self._df.append(df, sort=False, ignore_index=True)
        self._df['_dl'][np.isnan(self._df['_dl'])] = 0
        self._df['_dr'][np.isnan(self._df['_dr'])] = 0

    def add(self, start=None, stop=None, low=None, high=None, label=None, df=None):
        """ Add an annotation or a collection of annotations to the handler module.
        
            Individual annotations may be added using the arguments 'start', 'stop', 
            'low', 'high', 'label'.
            
            Groups of annotations may be added by first collecting them in a pandas 
            DataFrame or dictionary and then adding them using the 'df' argument.
        
            Args:
                start: str or float
                    Start time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                start: str or float
                    Stop time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                low: str or float
                    Lower frequency. Can be specified either as a float, in which case the 
                    unit will be assumed to be Hz, or as a string with an SI unit, 
                    for example, '3.1kHz'.
                high: str or float
                    Upper frequency. Can be specified either as a float, in which case the 
                    unit will be assumed to be Hz, or as a string with an SI unit, 
                    for example, '3.1kHz'.
                label: int
                    Integer label.
                df: pandas DataFrame or dict
                    DataFrame with columns 'start', 'stop', 'label' and optionally also 'low' 
                    and 'high', containing one or several annotations. 
        """        
        if label is not None:        
            start = _convert_to_sec(start)
            stop = _convert_to_sec(stop)
            low = _convert_to_Hz(low)
            high = _convert_to_Hz(high)
            df = {'start':start, 'stop':stop, 'low':low, 'high':high, 'label':label}

        self._add(df)
        
    def crop(self, start=0, stop=None, low=None, high=None):
        """ Crop annotations along the time and/or frequency dimension.
        
            Args:
                start: float or str
                    Lower edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                stop: float or str
                    Upper edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                low: float or str
                    Lower edge of frequency cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be Hz, 
                    or as a string with an SI unit, for example, '3.1kHz'
                high: float or str
                    Upper edge of frequency cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be Hz, 
                    or as a string with an SI unit, for example, '3.1kHz'
        """
        # convert to desired units
        low = _convert_to_Hz(low)
        high = _convert_to_Hz(high)
        start = _convert_to_sec(start)
        stop = _convert_to_sec(stop)

        # crop min frequency
        if low is not None:
            self._df['low'][self._df['low'] < low] = low

        # crop max frequency
        if high is not None:
            self._df['high'][self._df['high'] < high] = high

        # crop stop time
        if stop is not None:
            dr = -np.maximum(0, self._df['stop'] - stop)
            self._df['_dr'] = dr
            self._df['stop'] = self._df['stop'] + dr

        # crop start time
        if start > 0:
            self.shift(-start)

        # remove annotations that were fully cropped along the time dimension
        if start > 0 or stop is not None:
            self._df = self._df[self._df['stop'] > self._df['start']]

        # remove annotations that were fully cropped along the frequency dimension
        if low is not None or high is not None:
            self._df = self._df[(self._df['high'] > self._df['low'])]
            
    def shift(self, delta_time=0):
        """ Shift all annotations by a fixed amount along the time dimension.

            If the shift places some of the annotations (partially) before time zero, 
            these annotations are removed or cropped.

            Args:
                delta_time: float or str
                    Amount by which annotations will be shifted. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
        """      
        delta_time = _convert_to_sec(delta_time)
        
        self._df['start'] = self._df['start'] + delta_time
        self._df['_dl'] = np.maximum(0, -self._df['start'])
        self._df['start'][self._df['start'] < 0] = 0
        
        self._df['stop'] = self._df['stop'] + delta_time
        self._df['stop'][self._df['stop'] < 0] = 0

        self._df = self._df[self._df['stop'] > self._df['start']]
        
    def segment(self, num_segs, window_size, step_size=None, start=0):
        """ Divide the time axis into (potentially overlapping) segments of fixed duration.

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
                start: float or str
                    Start time for the first segment. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'.
                    Negative times are permitted. 
                    
            Returns:
                ans: dict
                    Dictionary in which the keys are the indices of those 
                    segments that have annotations, and the items are the 
                    annotation handlers.
        """              
        if step_size is None:
            step_size = window_size
        
        # convert to seconds
        window_size = _convert_to_sec(window_size)
        step_size = _convert_to_sec(step_size)
        start = _convert_to_sec(start)

        # find overlap between annotations and segments
        iA = np.maximum(0, np.ceil((self._df['start'] - start - window_size) / step_size))
        iA = iA.astype(int, copy=False)
        iB = np.minimum(num_segs - 1, np.floor((self._df['stop'] - start) / step_size))
        iB = iB.astype(int, copy=False)

        # loop over annotations
        ans = {}
        for index, row in self._df.iterrows():
            
            # segments that overlap with this annotation
            i = np.arange(iA[index], iB[index] + 1)
            a = start + i * step_size
            b = a + window_size         
            n = len(i)

            # left and right cuts
            dl = np.maximum(0, a - row['start'])
            dr = -np.maximum(0, row['stop'] - b)

            # crop/shift
            b = np.minimum(b, row['stop']) - a
            a = np.maximum(a, row['start']) - a

            # create/fill annotation handlers for each segment
            for j in range(n):
                r = row.copy()
                r['start'] = a[j]
                r['stop'] = b[j]
                r['_dl'] = dl[j]
                r['_dr'] = dr[j]
                idx = i[j]
                if idx in ans:
                    ans[idx].add(df=r)
                else:
                    ans[idx] = self.__class__(df=r)

        return ans
    

class AudioSourceHandler(AnnotationHandler):
    """ Class for handling audio source tags.
    
        An audio source tag is characterized by 
        
         * start and stop time in seconds 
         * label (typically the filename)
         * offset in seconds
         
        The AnnotationHandler stores audio source tags in a pandas DataFrame 
        and offers methods to add/get tags and perform various manipulations 
        such as cropping, shifting, and segmenting.

        Args:
            df: pandas DataFrame
                DataFrame with columns 'start', 'stop', 'label', and 'offset' containing 
                one or several audio source tags, to be passed on to the handler module.
    """
    def __init__(self, df=None):

        super().__init__(label_type='str', df=df)
        
        if 'offset' not in self._df.columns:
            self._df['offset'] = pd.Series(dtype='float')
            
        self._get_cols.append('offset')
        self._get_cols.remove('low')
        self._get_cols.remove('high')
        
    def add(self, start=0, stop=None, label=None, offset=0, df=None):
        """ Add an audio source tag or a collection of such tags to the handler module.
        
            Individual tags may be added using the arguments 'start', 'stop', 
            'offset', 'label'.

            Groups of tags may be added by first collecting them in a pandas 
            DataFrame or dictionary and then adding them using the 'df' argument.

            Args:
                start: str or float
                    Start time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                start: str or float
                    Stop time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                offset: str or float
                    Start time within the original audio source. Can be specified either 
                    as a float, in which case the unit will be assumed to be seconds, or 
                    as a string with an SI unit, for example, '22min'.
                label: str
                    Unique identifier of the audio source, typically a filename.
                df: pandas DataFrame or dict
                    DataFrame with columns 'start', 'stop', 'label' and optionally also 'low' 
                    and 'high', containing one or several annotations. 
        """        
        super().add(start=start, stop=stop, label=label, df=df)
        
        if df is None:
            offset = _convert_to_sec(offset)
            self._df['offset'].iloc[-1] = offset

    def crop(self, start=0, stop=None):
        """ Crop audio tags.
        
            Args:
                start: float or str
                    Lower edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
                stop: float or str
                    Upper edge of time cropping interval. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'
        """        
        super().crop(start, stop)        
        self._df['offset'] = self._df['offset'] + self._df['_dl']
        
    def segment(self, num_segs, window_size, step_size=None, start=0):
        """ Divide the time axis into (potentially overlapping) segments of fixed duration.

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
                start: float or str
                    Start time for the first segment. Can be specified either as 
                    a float, in which case the unit will be assumed to be seconds, 
                    or as a string with an SI unit, for example, '22min'.
                    Negative times are permitted. 
                    
            Returns:
                ans: dict
                    Dictionary in which the keys are the indices of those 
                    segments that have annotations, and the items are the 
                    annotation handlers.
        """
        ans = super().segment(num_segs, window_size, step_size, start)

        for _, an in ans.items():
            an._df['offset'] = an._df['offset'] + an._df['_dl']

        return ans