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
        BaseAudio class;
        BaseAudioTimeAxis class
"""
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ketos.audio.utils.misc as aum
from ketos.audio.annotation import AnnotationHandler, stack_annotations
from ketos.audio.utils.axis import LinearAxis


def segment_data(x, window, step=None):
    """ Divide the time axis into segments of uniform length, which may or may 
        not be overlapping.

        Window length and step size are converted to the nearest integer number 
        of time steps.

        If necessary, the data array will be padded with zeros at the end to 
        ensure that all segments have an equal number of samples. 

        Args:
            x: BaseAudioTime
                Data to be segmented
            window: float
                Length of each segment in seconds.
            step: float
                Step size in seconds.

        Returns:
            audio_objects: list(BaseAudioTime)
                Data segments
    """              
    if step is None: step = window

    time_res = x.time_res()
    win_len = aum.num_samples(window, 1. / time_res)
    step_len = aum.num_samples(step, 1. / time_res)

    # segment data array
    segs = aum.segment(x=x.data, win_len=win_len, step_len=step_len, pad_mode='zero')

    window = win_len * time_res
    step = step_len * time_res
    num_segs = segs.shape[0]

    # segment annotations
    if x.annot is not None:
        annots = x.annot.segment(num_segs=num_segs, window=window, step=step)    
    else: 
        annots = None

    # compute offsets
    offsets = np.arange(num_segs) * step

    # add global offset
    offsets += x.offset

    # create audio objects
    audio_objects = []    
    for i in range(segs.shape[0]):
        if annots is not None: annot = annots.get(id=i)
        else: annot = None
        kwargs = x.get_kwargs()
        kwargs.pop('offset', None)
        audio_objects.append(x.__class__(data=segs[i], annot=annot, offset=offsets[i], **kwargs))

    return audio_objects


class BaseAudio():
    """ Parent class for all audio classes.

        While the underlying data array can be accessed via the :attr:`data` attribute,
        it is recommended to always use the :func:`get_data` function to access the data 
        array, i.e., 

        >>> from ketos.audio.base_audio import BaseAudio
        >>> x = np.ones(6)
        >>> audio_sample = BaseAudio(data=x)
        >>> audio_sample.get_data()
        array([1., 1., 1., 1., 1., 1.])
        
        Args:
            data: numpy array
                Data
            filename: str
                Filename of the original data file, if available (optional)
            offset: float
                Position within the original data file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            duration: float
                Duration in seconds.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation and its arguments, if any. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}

        Attributes:
            data: numpy array
                Data 
            ndim: int
                Dimensionality of data.
            filename: str
                Filename of the original data file, if available (optional)
            offset: float
                Position within the original data file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Data label.
            annot: AnnotationHandler or pandas DataFrame
                AnnotationHandler object.
            allowed_transforms: dict
                Transforms that can be applied via the apply_transform method
            transform_log: list
                List of transforms that have been applied to this object
    """
    def __init__(self, data, filename='', offset=0, duration=None, label=None, annot=None, 
                    transforms=None, transform_log=None, **kwargs):

        if transform_log is None: transform_log = []
        if isinstance(annot, pd.DataFrame): annot = AnnotationHandler(annot)

        self.ndim = np.ndim(data)
        self.data = data

        self.filename = filename
        self.offset = offset
        self._duration = duration
        self.label = label

        self.annot = annot

        self.allowed_transforms = {'normalize': self.normalize, 
                                   'adjust_range': self.adjust_range}

        self.transform_log = transform_log        
        self.apply_transforms(transforms)

        self.kwargs = kwargs

    @staticmethod
    def infer_shape(**kwargs):
        """ Infers the data shape that would result if the class were 
            instantiated with a specific set of parameter values.

            Returns a None value if `duration` or `rate` are not specified.

            Args:
                duration: float
                    Duration in seconds
                rate: float
                    Sampling rate in Hz

            Returns:
                : tuple
                    Inferred shape. If the parameter value do not allow 
                    the shape be inferred, a None value is returned.
        """
        if 'duration' in kwargs.keys() and 'rate' in kwargs.keys():
            num_samples = int(kwargs['duration'] * kwargs['rate'])
            return (num_samples,)
        else:
            return None

    def get(self):
        """ Get a copy of this instance """ 
        return self.__class__(data=self.get_data(), annot=self.get_annotations(), **self.get_kwargs())

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = {}
        kwargs.update(self.get_repres_attrs())
        kwargs.update(self.get_instance_attrs())
        return kwargs

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = {'transform_log':self.transform_log}
        return attrs

    def get_instance_attrs(self):
        """ Get instance attributes """ 
        attrs = {'filename':self.filename, 'offset':self.offset, 'duration':self._duration, 'label':self.label}
        attrs.update(self.kwargs)
        return attrs

    def get_data(self):
        """ Get underlying data.

            Returns:
                : numpy array
                    Data array 
        """
        return self.data

    def get_filename(self):
        """ Get filename.

            Returns:
                : string
                    Filename
        """
        return self.filename

    def get_offset(self):
        """ Get offset.

            Returns:
                : float
                    Offset
        """
        return self.offset

    def duration(self):
        """ Data array duration in seconds

            TODO: rename to get_duration()

            Returns:
                : float
                   Duration in seconds
        """    
        return self._duration

    def get_label(self, id=None):
        """ Get label.

            Returns:
                : int
                    Label
        """
        return self.label

    def get_annotations(self):
        """ Get annotations.

            Returns:
                : pandas DataFrame
                    Annotations 
        """
        if self.annot is None: return None
        else: return self.annot.get()

    def deepcopy(self):
        """ Make a deep copy of the present instance

            See https://docs.python.org/2/library/copy.html

            Returns:
                : BaseAudio
                    Deep copy.
        """
        return copy.deepcopy(self)

    def max(self, axis=0):
        """ Maximum data value along selected axis

            Args:
                axis: int
                    Axis along which metric is computed

            Returns:
                : array-like
                   Maximum value of the data array
        """    
        return np.max(self.data, axis=axis)

    def min(self, axis=0):
        """ Minimum data value along selected axis

            Args:
                axis: int
                    Axis along which metric is computed

            Returns:
                : array-like
                   Minimum value of the data array
        """    
        return np.min(self.data, axis=axis)

    def std(self, axis=0):
        """ Standard deviation along selected axis

            Args:
                axis: int
                    Axis along which metric is computed

            Returns:
                : array-like
                   Standard deviation of the data array
        """   
        return np.std(self.data, axis=axis) 

    def average(self, axis=0):
        """ Average value along selected axis

            Args:
                axis: int
                    Axis along which metric is computed

            Returns:
                : array-like
                   Average value of the data array
        """   
        return np.average(self.data, axis=axis)

    def median(self, axis=0):
        """ Median value along selected axis

            Args:
                axis: int
                    Axis along which metric is computed

            Returns:
                : array-like
                   Median value of the data array
        """   
        return np.median(self.data, axis=axis)

    def normalize(self, mean=0, std=1):
        """ Normalize the data array to specified mean and standard deviation.

            For the data array to be normalizable, it must have non-zero standard 
            deviation. If this is not the case, the array is unchanged by calling 
            this method. 

            Args:
                mean: float
                    Mean value of the normalized array. The default is 0.
                std: float
                    Standard deviation of the normalized array. The default is 1.
        """
        std_orig = np.std(self.data)
        if std_orig > 0:
            self.data = std * (self.data - np.mean(self.data)) / std_orig + mean
            self.transform_log.append({'name':'normalize', 'mean':mean, 'std':std})

    def adjust_range(self, range=(0,1)):
        """ Applies a linear transformation to the data array that puts the values
            within the specified range. 

            Args:
                range: tuple(float,float)
                    Minimum and maximum value of the desired range. Default is (0,1)
        """
        x_min = self.min()
        x_max = self.max()
        self.data = (range[1] - range[0]) * (self.data - x_min) / (x_max - x_min) + range[0]
        self.transform_log.append({'name':'adjust_range', 'range':range})

    def view_allowed_transforms(self):
        """ View allowed transformations for this audio object.

            Returns:
                : list
                    List of allowed transformations
        """
        return list(self.allowed_transforms.keys())

    def apply_transforms(self, transforms):
        """ Apply specified transforms to the audio object.

            Args:
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation and its arguments, if any. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}

            Returns:
                None

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # read audio signal from wav file
                >>> wf = Waveform.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # print allowed transforms
                >>> wf.view_allowed_transforms()
                ['normalize', 'adjust_range', 'crop', 'add_gaussian_noise', 'bandpass_filter']
                >>> # apply gaussian normalization followed by cropping
                >>> transforms = [{'name':'normalize','mean':0.5,'std':1.0},{'name':'crop','start':0.2,'end':0.7}]
                >>> wf.apply_transforms(transforms)
                >>> # inspect record of applied transforms 
                >>> wf.transform_log
                [{'name': 'normalize', 'mean': 0.5, 'std': 1.0}, {'name': 'crop', 'start': 0.2, 'end': 0.7, 'length': None}]
        """
        if transforms is None: return

        t = copy.deepcopy(transforms)
        for kwargs in t:
            name = kwargs.pop('name')
            if name in self.view_allowed_transforms():
                self.allowed_transforms[name](**kwargs)

    def annotate(self, **kwargs):
        """ Add an annotation or a collection of annotations.

            Input arguments are described in :meth:`ketos.audio.annotation.AnnotationHandler.add`
        """
        if self.annot is None: self.annot = AnnotationHandler() #if the object does not have an annotation handler, create one!

        self.annot.add(**kwargs)


class BaseAudioTime(BaseAudio):
    """ Parent class for time-series audio classes such as :class:`audio.waveform.Waveform` 
        and :class:`audio.spectrogram.Spectrogram`.

        Args:
            data: numpy array
                Data
            time_res: float
                Time resolution in seconds
            filename: str
                Filename of the original data file, if available (optional)
            offset: float
                Position within the original data file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation and its arguments, if any. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}

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
            allowed_transforms: dict
                Transforms that can be applied via the apply_transform method
            transform_log: list
                List of transforms that have been applied to this object
    """
    def __init__(self, data, time_res, filename='', offset=0, label=None, annot=None, 
                    transforms=None, transform_log=None, **kwargs):

        bins = max(1, data.shape[0])
        length = data.shape[0] * time_res
        self.time_ax = LinearAxis(bins=bins, extent=(0., length), label='Time (s)') #initialize time axis

        super().__init__(data=data, filename=filename, offset=offset, duration=self.duration(),
            label=label, annot=annot, transforms=transforms, transform_log=transform_log, **kwargs)

        self.allowed_transforms.update({'crop': self.crop})

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'time_res':self.time_res()})
        return attrs

    def get_instance_attrs(self):
        """ Get instance attributes """ 
        attrs = super().get_instance_attrs()
        attrs.pop('duration', None)
        return attrs

    def time_res(self):
        """ Get the time resolution.

            Returns:
                : float
                    Time resolution in seconds
        """
        return self.time_ax.bin_width()

    def duration(self):
        """ Data array duration in seconds

            Returns:
                : float
                   Duration in seconds
        """    
        return self.time_ax.max()

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
                : list(BaseAudioTime)
                    Stacked data segments
        """   
        return segment_data(self, window, step)

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

        if make_copy is False:
            self.transform_log.append({'name':'crop', 'start':start, 'end':end, 'length':length})

        return d

    def plot(self, figsize=(5,4), label_in_title=True, append_title=''):
        """ Plot the data with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the data.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                figsize: tuple
                    Figure size
                label_in_title: bool
                    Include label (if available) in figure title
                append_title: str
                    Append this string to the title
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
                ax: matplotlib.axes.Axes
                    Axes object
        """
        # create canvas and axes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True)

        # select the data array and attributes
        x = self.get_data()
        filename = self.get_filename()
        offset = self.get_offset()
        label = self.get_label()

        # axis labels
        ax.set_xlabel(self.time_ax.label)

        # title
        title = ""
        if filename is not None: title += "{0}".format(filename)       
        if label is not None and label_in_title:
            if len(title) > 0: title += ", "
            title += "{0}".format(label)

        title += append_title
        plt.title(title)

        # if offset is non-zero, add a second time axis at the top 
        # showing the `absolute` time
        if offset != 0:
            axt = ax.twiny()
            axt.set_xlim(offset, offset + self.duration())

        #fig.tight_layout()
        return fig, ax

