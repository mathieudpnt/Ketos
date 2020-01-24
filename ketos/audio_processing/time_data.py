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

""" time_data module within the ketos library

    This module provides utilities to work with time series data.

    Contents:
        TimeData class
"""
import os
import copy
import numpy as np
import ketos.audio_processing.audio_processing as ap
from ketos.audio_processing.annotation import AnnotationHandler
from ketos.audio_processing.axis import LinearAxis

def stack_attrs(filename, offset, label, mul):
    """ Ensure that data attributes have expected multiplicity.

        If the attribute is specified as a list or an array-like object, 
        assert that the length equals the data multiplicity.

        Args:
            filename: str or list(str)
                Filename attribute.
            offset: float or array-like
                Offset attribute.
            label: int or array-like
                Label attribute.
            mul: int
                Audio/spectrogram multiplicity

        Returns:
            filename: list(str)
                Filename attribute
            offset: array-like
                Offset attribute
            label: array-like
                Label attribute
    """
    if filename:
        if isinstance(filename, str):
            filename = [filename for _ in range(mul)]

        assert len(filename) == mul, 'Number of filenames ({0}) does not match data multiplicity ({1})'.format(len(filename), mul)

    if offset:
        if isinstance(offset, float) or isinstance(offset, int):
            offset = np.ones(mul, dtype=float) * float(offset)

        assert len(offset) == mul, 'Number of offsets ({0}) does not match data multiplicity ({1})'.format(len(offset), mul)

    if label:
        if isinstance(label, float) or isinstance(label, int):
            label = np.ones(mul, dtype=int) * int(label)

        assert len(label) == mul, 'Number of labels ({0}) does not match data multiplicity ({1})'.format(len(label), mul)

    return filename, offset, label

def segment_data(x, window, step=None):
    """ Divide the time axis into segments of uniform length, which may or may 
        not be overlapping.

        Window length and step size are converted to the nearest integer number 
        of time steps.

        If necessary, the data array will be padded with zeros at the end to 
        ensure that all segments have an equal number of samples. 

        Args:
            x: TimeData
                Data to be segmented
            window: float
                Length of each segment in seconds.
            step: float
                Step size in seconds.

        Returns:
            segs: TimeData
                Stacked data segments
            offset: array-like
                Offsets in seconds
    """              
    if step_size is None:
        step_size = window_size

    time_res = x.time_res()
    win_len = ap.num_samples(window, 1. / time_res)
    step_len = ap.num_samples(step, 1. / time_res)

    # segment data array
    segs = ap.segment(x=x.data, win_len=win_len, step_len=step_len, pad_mode='zero')

    window = win_len * time_res
    step = step_len * time_res
    num_segs = segs.shape[0]

    # segment annotations
    if x.annot:
        annots = x.annot.segment(num_segs=num_segs, window=window, step=step)
    else:
        annots = None

    # compute offsets
    offset = np.arange(num_segs) * step

    return segs, offset

class TimeData():
    """ Parent class for time-series data classes such as
        :class:`audio_processing.audio_signal.AudioSignal` 
        and :class:`audio_processing.spectrogram.Spectrogram`.

        Args:
            data: numpy array
                Data
            time_res: float
                Time resolution in seconds
            ndim: int
                Dimensionality of data.
            filename: str
                Filename of the original data file, if available (optional)
            offset: float
                Position within the original data file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional

        Attributes:
            data: numpy array
                Data 
            ndim: int
                Dimensionality of data.
            time_ax: LinearAxis
                Axis object for the time dimension
            filename: str
                Filename of the original data file, if available (optional)
            offset: float
                Position within the original data file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Data label.
            annot: AnnotationHandler
                AnnotationHandler object.
    """
    def __init__(self, data, time_res, ndim, filename='', offset=0, label=None, annot=None):
        self.ndim = ndim
        self.data = data
        length = data.shape[0] * time_res
        self.time_ax = LinearAxis(bins=data.shape[0], extent=(0., length), label='Time (s)') #initialize time axis

        if np.ndim(data) == ndim + 1: #stacked data arrays
            mul = data.shape[ndim]
            filename, offset, label = stack_attrs(filename, offset, label, mul)
            if annot:
                assert annot.num_sets() == mul, 'Number of annotation sets ({0}) does not match number of data sets ({1})'.format(annot.num_sets(), mul)

        self.filename = filename
        self.offset = offset
        self.label = label
        self.annot = annot

    def get_data(self, id=0):
        """ Get the underlying data numpy array.

            Args:
                id: int
                    Data set ID. Only relevant if the object 
                    contains multiple, stacked data sets.

            Returns:
                d: numpy array
                    Data
        """
        if id is None or np.ndim(self.data) == self.ndim:
            d = self.data
        else:
            d = self.data[:,id]

        return d

    def annotations(self, id=None):
        """ Get annotations.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                ans: pandas DataFrame
                    Annotations 
        """
        if self.annot:
            ans = self.annot.get(id)
        else:
            ans = None

        return ans


    def time_res():
        """ Get the time resolution.

            Returns:
                : float
                    Time resolution in seconds
        """
        return self.time_ax.bin_width()

    def deepcopy(self):
        """ Make a deep copy of the present instance

            See https://docs.python.org/2/library/copy.html

            Returns:
                : TimeData
                    Deep copy.
        """
        return copy.deepcopy(self)

    def length(self):
        """ Data array duration in seconds

            Returns:
                : float
                   Duration in seconds
        """    
        return self.time_ax.max()

    def max(self):
        """ Maximum data value

            Returns:
                : array-like
                   Maximum value of the data array
        """    
        return np.max(self.data, axis=0)

    def min(self):
        """ Minimum dta value

            Returns:
                : array-like
                   Minimum value of the data array
        """    
        return np.min(self.data, axis=0)

    def std(self):
        """ Standard deviation

            Returns:
                : array-like
                   Standard deviation of the data array
        """   
        return np.std(self.data, axis=0) 

    def average(self):
        """ Average value

            Returns:
                : array-like
                   Average value of the data array
        """   
        return np.average(self.data, axis=0)

    def median(self):
        """ Median value

            Returns:
                : array-like
                   Median value of the data array
        """   
        return np.median(self.data, axis=0)

    def normalize(self):
        """ Normalize the data array so that values range from 0 to 1
        """
        x_min = self.min()
        x_max = self.max()
        self.data = (self.data - x_min) / (x_max - x_min)

    def annotate(self, **kwargs):
        """ Add an annotation or a collection of annotations.

            Input arguments are described in :method:`audio_processing.annotation.AnnotationHandler.add`
        """
        assert self.annot is not None, "Attempting to add annotations to an AudioSignal without an AnnotationHandler object" 

        self.annot.add(**kwargs)

    def label_array(self, label):
        """ Get an array indicating presence/absence (1/0) 
            of the specified annotation label for each time bin.

            Args:
                label: int
                    Label of interest.

            Returns:
                y: numpy.array
                    Label array
        """
        assert self.annot is not None, "An AnnotationHandler object is required for computing the label vector" 

        y = np.zeros(self.time_ax.bins)
        ans = self.annot.get(label=label)
        for _,an in ans.iterrows():
            b1 = self.time_ax.bin(an.start, truncate=True)
            b2 = self.time_ax.bin(an.end, truncate=True, closed_right=True)
            y[b1:b2+1] = 1

        return y

    def segment(self, window, step=None):
        """ Divide the time axis into segments of uniform length, which may or may 
            not be overlapping.

            Window length and step size are converted to the nearest integer number 
            of time steps.

            If necessary, the data array will be padded with zeros at the end to 
            ensure that all segments have an equal number of samples. 

            Args:
                window: float
                    Length of each segment in seconds.
                step: float
                    Step size in seconds.

            Returns:
                d: TimeData
                    Stacked data segments
        """   
        segs, offset = segment_data(self, window, step)           

        d = self.__class__(data=segs, time_res=self.time_res(), ndim=self.ndim, filename=self.filename,\
            offset=offset, label=self.label, annot=annots)

        return d

    def crop(self, start=None, end=None, length=None, make_copy=False):
        """ Crop audio signal.
            
            Args:
                start: float
                    Start time in seconds, measured from the left edge of spectrogram.
                end: float
                    End time in seconds, measured from the left edge of spectrogram.
                length: int
                    Horizontal size of the cropped image (number of pixels). If provided, 
                    the `end` argument is ignored. 
                make_copy: bool
                    Return a cropped copy of the spectrogra. Leaves the present instance 
                    unaffected. Default is False.

            Returns:
                a: TimeData
                    Cropped data array
        """
        if make_copy:
            d = self.deepcopy()
        else:
            d = self

        # crop axis
        b1, b2 = d.time_ax.cut(x_min=start, x_max=end, bins=length)

        # crop audio signal
        d.data = d.data[b1:b2+1]

        # crop annotations, if any
        if d.annot:
            d.annot.crop(start=start, end=end)

        d.offset += d.time_ax.low_edge(0) #update time offset
        d.time_ax.zero_offset() #shift time axis to start at t=0 

        return d
