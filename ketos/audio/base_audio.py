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

""" 'audio.base_audio' module within the ketos library

    This module contains the base class for the Waveform and Spectrogram classes.

    Contents:
        BaseAudio class
"""
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ketos.audio.utils.misc as aum
from ketos.audio.annotation import AnnotationHandler, stack_annotations
from ketos.audio.utils.axis import LinearAxis

def stack_attr(value, shape, dtype):
    """ Ensure that data attribute has the requested shape.

        Args:
            value: array-like
                Attribute values.
            shape: tuple
                Requested shape.
            dtype: str
                Type

        Returns:
            value_stacked: numpy array
                Array containing the stacked attribute values
    """
    if value is None:
        return None

    value_stacked = value

    if isinstance(value_stacked, list):
        value_stacked = np.array(value, dtype=dtype)
    
    if np.ndim(value_stacked) == 0:
        if dtype is str or dtype is 'str':
            dtype = 'S{0}'.format(len(value))

        value_stacked = np.empty(shape=shape, dtype=dtype)
        value_stacked[:] = value

    assert value_stacked.shape == shape, 'Attribute value shape ({0}) does not match requested shape ({1})'.format(value_stacked.shape, shape)

    return value_stacked

def segment_data(x, window, step=None):
    """ Divide the time axis into segments of uniform length, which may or may 
        not be overlapping.

        Window length and step size are converted to the nearest integer number 
        of time steps.

        If necessary, the data array will be padded with zeros at the end to 
        ensure that all segments have an equal number of samples. 

        Args:
            x: BaseAudio
                Data to be segmented
            window: float
                Length of each segment in seconds.
            step: float
                Step size in seconds.

        Returns:
            segs: BaseAudio
                Stacked data segments
            filename: array-like
                Filenames
            offset: array-like
                Offsets in seconds
            label: array-like
                Labels
            annot: AnnotationHandler
                Stacked annotation handlers, if any
    """              
    if step is None:
        step = window

    time_res = x.time_res()
    win_len = aum.num_samples(window, 1. / time_res)
    step_len = aum.num_samples(step, 1. / time_res)

    # segment data array
    segs = aum.segment(x=x.data, win_len=win_len, step_len=step_len, pad_mode='zero')

    window = win_len * time_res
    step = step_len * time_res
    num_segs = segs.shape[0]

    # segment annotations
    if x.annot:
        annot = x.annot.segment(num_segs=num_segs, window=window, step=step)
    else:
        annot = None

    # compute offsets
    offset = np.arange(num_segs) * step

    #permute axes so the segment no. becomes the last axis
    axes = np.concatenate([np.arange(1, len(segs.shape)), [0]]) 
    segs = np.transpose(segs, axes)

    # segments inherit filename and label from original instance
    filename = x.filename
    label = x.label

    #when segment method is applied to stacked objects, a little extra work is required:
    if len(segs.shape) > x.ndim + 1: 
        num_segs = segs.shape[-1]

        if x.filename is not None:
            filename = [[x for _ in range(num_segs)] for x in x.filename]

        if x.label is not None:
            label = np.array([[x for _ in range(num_segs)] for x in x.label], dtype=int)

        if x.offset is not None:
            offset = np.repeat(offset[np.newaxis, :], segs.shape[x.ndim], axis=0)

    return segs, filename, offset, label, annot

def get_slice(arr, axis=0, indices=None):
    """ Get a slice of an array.

        Args:
            arr: array-like
                Input array
            axis: int
                The axis over which to select values.
            indices: int or tuple
                The indices of the values to extract.

        Returns:
            ans: array-like
                Sliced array
    """
    ans = arr

    if indices is None or np.ndim(arr) == 0:
        return ans

    if np.ndim(indices) == 0:
        indices = [indices]

    num_dims = min(len(indices), len(arr.shape) - axis)
    for i in range(num_dims):
        ans = ans.take(indices=indices[i], axis=axis + i)
        
    return ans


class BaseAudio():
    """ Parent class for time-series data classes such as
        :class:`audio.waveform.Waveform` 
        and :class:`audio.spectrogram.Spectrogram`.

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
            annot: AnnotationHandler or pandas DataFrame
                AnnotationHandler object.
    """
    def __init__(self, data, time_res, ndim, filename='', offset=0, label=None, annot=None):
        self.ndim = ndim
        self.data = data
        length = data.shape[0] * time_res
        self.time_ax = LinearAxis(bins=data.shape[0], extent=(0., length), label='Time (s)') #initialize time axis

        if isinstance(annot, pd.DataFrame): annot = AnnotationHandler(annot)

        if np.ndim(data) > ndim: #stacked arrays
            filename = stack_attr(filename, shape=data.shape[ndim:], dtype=str)
            offset = stack_attr(offset, shape=data.shape[ndim:], dtype=float)
            label = stack_attr(label, shape=data.shape[ndim:], dtype=int)

        self.filename = filename
        self.offset = offset
        self.label = label
        self.annot = annot

        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        n = np.ndim(self.data) - self.ndim
        if n == 0:   idx = 0
        elif n == 1: idx = self.counter
        else: idx = np.unravel_index(self.counter, shape=self.data.shape[n:])
        self.counter += 1
        return idx

    @classmethod
    def stack(cls, objects):
        """ Stack objects 

            Args:
                objects: list(BaseAudio)
                    List of objects to be stacked.

            Returns:
                : BaseAudio
                    Stacked objects        
        """
        assert len(objects) > 0, 'at least one object required'

        kwargs = objects[0].get_attrs()

        filename = [a.filename for a in objects]
        offset   = [a.offset for a in objects]

        label = [a.label for a in objects if a.label is not None]
        if len(label) == 0: label = None

        annot    = [a.annot for a in objects if a.annot is not None]
        if len(annot) == 0: annot = None
        else: annot = stack_annotations(annot)

        data = np.moveaxis(np.concatenate([a.data[np.newaxis,:] for a in objects], axis=0), 0, -1)

        return cls(data=data, filename=filename, offset=offset, label=label, annot=annot, **kwargs)

    @classmethod
    def get(cls, id):
        return cls(data=self.get_data(id), filename=self.get_filename(id), 
            offset=self.get_offset(id), label=get_label(id), annot=get_annotations(id), **self.get_attrs())

    def get_attrs(self):
        return {'time_res':self.time_res(), 'ndim':self.ndim}

    def num_objects(self):
        num = 1
        n = np.ndim(self.data) - self.ndim
        if n > 0:
            dims = self.data.shape[n:]
            for d in dims: num *= d
        
        return num

    def get_data(self, id=None):
        """ Get underlying data.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                : numpy array
                    Data array 
        """
        return get_slice(self.data, axis=self.ndim, indices=id)

    def get_filename(self, id=None):
        """ Get filename.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                : array-like
                    Filename
        """
        ans = get_slice(self.filename, axis=0, indices=id)
        if ans is not None and not isinstance(ans, str) and np.ndim(ans) == 0:
            ans = ans.decode()

        if isinstance(ans, np.ndarray):
            ans = ans.astype(str)
        
        return ans

    def get_offset(self, id=None):
        """ Get offset.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                : array-like
                    Offset
        """
        return get_slice(self.offset, axis=0, indices=id)

    def get_label(self, id=None):
        """ Get label.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                : array-like
                    Label
        """
        return get_slice(self.label, axis=0, indices=id)

    def get_annotations(self, id=None):
        """ Get annotations.

            Args:
                id: int
                    Data array ID. Only relevant if the object 
                    contains multiple, stacked arrays.

            Returns:
                : pandas DataFrame
                    Annotations 
        """
        if self.annot is None: 
            return None
        else: 
            return self.annot.get(id=id)

    def time_res(self):
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
                : BaseAudio
                    Deep copy.
        """
        return copy.deepcopy(self)

    def duration(self):
        """ Data array duration in seconds

            Returns:
                : float
                   Duration in seconds
        """    
        return self.time_ax.max()

    def max(self):
        """ Maximum data value along time axis

            Returns:
                : array-like
                   Maximum value of the data array
        """    
        return np.max(self.data, axis=0)

    def min(self):
        """ Minimum data value along time axis

            Returns:
                : array-like
                   Minimum value of the data array
        """    
        return np.min(self.data, axis=0)

    def std(self):
        """ Standard deviation along time axis

            Returns:
                : array-like
                   Standard deviation of the data array
        """   
        return np.std(self.data, axis=0) 

    def average(self):
        """ Average value along time axis

            Returns:
                : array-like
                   Average value of the data array
        """   
        return np.average(self.data, axis=0)

    def median(self):
        """ Median value along time axis

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

            Input arguments are described in :meth:`ketos.audio.annotation.AnnotationHandler.add`
        """
        if self.annot is None: #if the object does not have an annotation handler, create one!
            self.annot = AnnotationHandler() 

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
                d: BaseAudio
                    Stacked data segments
        """   
        segs, filename, offset, label, annot = segment_data(self, window, step)

        # create stacked object
        d = self.__class__(data=segs, time_res=self.time_res(), ndim=self.ndim, filename=filename,\
            offset=offset, label=label, annot=annot)

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
                a: BaseAudio
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

    def plot(self, id=0, figsize=(5,4), label_in_title=True):
        """ Plot the data with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the data.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                id: int
                    ID of data array to be plotted. Only relevant if the object 
                    contains multiple, stacked data arrays.
                figsize: tuple
                    Figure size
                label_in_title: bool
                    Include label (if available) in figure title
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
                ax: matplotlib.axes.Axes
                    Axes object
        """
        # create canvas and axes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True)

        # select the data array and attributes
        x = self.get_data(id)
        filename = self.get_filename(id)
        offset = self.get_offset(id)
        label = self.get_label(id)

        # axis labels
        ax.set_xlabel(self.time_ax.label)

        # title
        title = ""
        if filename is not None: title += "{0}".format(filename)       
        if label is not None and label_in_title:
            if len(title) > 0: title += ", "
            title += "{0}".format(label)

        plt.title(title)

        # if offset is non-zero, add a second time axis at the top 
        # showing the `absolute` time
        if offset != 0:
            axt = ax.twiny()
            axt.set_xlim(offset, offset + self.duration())

        #fig.tight_layout()
        return fig, ax

