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

""" Spectrogram module within the ketos library

    This module provides utilities to work with spectrograms.

    Spectrograms are two-dimensional visual representations of 
    sound waves, in which time is shown along the horizontal 
    axis, frequency along the vertical axis, and color is used 
    to indicate the sound amplitude. Read more on Wikipedia:
    https://en.wikipedia.org/wiki/Spectrogram

    The module contains the parent class Spectrogram, and four
    child classes (MagSpectrogram, PowerSpectrogram, MelSpectrogram, 
    CQTSpectrogram), which inherit methods and attributes from the 
    parent class.

    Note, however, that not all methods (e.g. crop) work for all 
    child classes. See the documentation of the individual methods 
    for further details.

    Contents:
        Spectrogram class:
        MagSpectrogram class:
        PowerSpectrogram class:
        MelSpectrogram class:
        CQTSpectrogram class
"""

import os
import numpy as np
from scipy.signal import get_window
from scipy.fftpack import dct
from scipy import ndimage
from skimage.transform import rescale
import matplotlib.pyplot as plt
import time
import datetime
import math
from ketos.audio_processing.audio_processing import make_frames, to_decibel, from_decibel, estimate_audio_signal, enhance_image
from ketos.audio_processing.audio import AudioSignal
from ketos.data_handling.parsing import WinFun
from ketos.utils import random_floats, factors
from tqdm import tqdm
from librosa.core import cqt
import librosa


import copy
from ketos.audio_processing.audio_processing import stft, make_frames
from ketos.audio_processing.axis import LinearAxis, Log2Axis
from ketos.audio_processing.annotation import AnnotationHandler
import ketos.audio_processing.augmentation as aug


def ensure_same_length(specs, pad=False):
    """ Ensure that all spectrograms have the same length.

        Note that all spectrograms must have the same time resolution.
        If this is not the case, an assertion error is thrown.
        
        Args:
            specs: list(Spectrogram)
                Input spectrograms
            pad: bool
                If True, the shorter spectrograms will be padded with zeros 
                on the right (i.e. at late times). If False, the longer 
                spectrograms will be cropped on the right.
    
        Returns:   
            specs: list(Spectrogram)
                List of same-length spectrograms

        Example:
            >>> from ketos.audio_processing.audio import AudioSignal
            >>> # Create two audio signals with different lengths
            >>> audio1 = AudioSignal.morlet(rate=100, frequency=5, width=1)   
            >>> audio2 = AudioSignal.morlet(rate=100, frequency=5, width=1.5)
            >>>
            >>> # Compute spectrograms
            >>> spec1 = MagSpectrogram(audio1, winlen=0.2, winstep=0.05)
            >>> spec2 = MagSpectrogram(audio2, winlen=0.2, winstep=0.05)
            >>>
            >>> # Print the lengths
            >>> print('{0:.2f}, {1:.2f}'.format(spec1.duration(), spec2.duration()))
            5.85, 8.85
            >>>
            >>> # Ensure all spectrograms have same length as the shortest spectrogram
            >>> specs = ensure_same_length([spec1, spec2])
            >>> print('{0:.2f}, {1:.2f}'.format(specs[0].duration(), specs[1].duration()))
            5.85, 5.85
    """
    if len(specs) == 0:
        return specs

    tres = specs[0].tres # time resolution of 1st spectrogram

    nt = list()
    for s in specs:
        assert s.tres == tres, 'Spectrograms must have the same time resolution' 
        nt.append(s.tbins())

    nt = np.array(nt)
    if pad: 
        n = np.max(nt)
        for s in specs:
            ns = s.tbins()
            if n-ns > 0:
                s.image = np.pad(s.image, pad_width=((0,n-ns),(0,0)), mode='constant')
                s.time_vector = np.append(s.time_vector, np.zeros(n-ns))
                s.file_vector = np.append(s.file_vector, np.zeros(n-ns))
    else: 
        n = np.min(nt)
        for s in specs:
            s.image = s.image[:n]
            s.time_vector = s.time_vector[:n]
            s.file_vector = s.file_vector[:n]

    return specs

def stack_spec_attrs(filename, offset, label, mul):
    """ Ensure that spectrogram attributes have expected multiplicity.

        If the attribute is specified as a list or an array-like object, 
        assert that the length equals the spectrogram multiplicity.

        Args:
            filename: str or list(str)
                Filename attribute.
            offset: float or array-like
                Offset attribute.
            label: int or array-like
                Label attribute.
            mul: int
                Spectrogram multiplicity

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

        assert len(filename) == mul, 'Number of filenames ({0}) does not match spectrogram multiplicity ({1})'.format(len(filename), mul)

    if offset:
        if isinstance(offset, float) or isinstance(offset, int):
            offset = np.ones(mul, dtype=float) * float(offset)

        assert len(offset) == mul, 'Number of offsets ({0}) does not match spectrogram multiplicity ({1})'.format(len(offset), mul)

    if label:
        if isinstance(label, float) or isinstance(label, int):
            label = np.ones(mul, dtype=int) * int(label)

        assert len(label) == mul, 'Number of labels ({0}) does not match spectrogram multiplicity ({1})'.format(len(label), mul)

    return filename, offset, label

class Spectrogram():
    """ Spectrogram.

        Parent class for MagSpectrogram, PowerSpectrogram, MelSpectrogram, 
        and CQTSpectrogram.

        The Spectrogram class stores the spectrogram pixel values in a 2d 
        numpy array, where the first axis (0) is the time dimension and 
        the second axis (1) is the frequency dimensions.

        The Spectrogram class can also store a stack of multiple, identical-size, 
        spectrograms in a 3d numpy array with the last axis (3) representing the 
        multiple instances.

        Args:
            image: 2d or 3d numpy array
                Spectrogram pixel values. 
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            spec_type: str
                Spectrogram type. Options include,
                    * 'Mag': Magnitude spectrogram
                    * 'Pow': Power spectrogram
                    * 'Mel': Mel spectrogram
                    * 'CQT': CQT spectrogram
            freq_ax: LinearAxis or Log2Axis
                Axis object for the frequency dimension
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            
        Attributes:
            image: 2d or 3d numpy array
                Spectrogram pixel values. 
            time_ax: LinearAxis
                Axis object for the time dimension
            freq_ax: LinearAxis or Log2Axis
                Axis object for the frequency dimension
            type: str
                Spectrogram type. Options include,
                    * 'Mag': Magnitude spectrogram
                    * 'Pow': Power spectrogram
                    * 'Mel': Mel spectrogram
                    * 'CQT': CQT spectrogram
            filename: str or list(str)
                Name of the source audio file.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file.
            label: int
                Spectrogram label.
            annot: AnnotationHandler
                AnnotationHandler object.
"""
    def __init__(self, image, time_res, spec_type, freq_ax, filename=None, offset=0, label=None, annot=None):
        self.image = image
        length = time_res * image.shape[0]
        self.time_ax = LinearAxis(bins=image.shape[0], extent=(0., length), label='Time (s)') #initialize time axis
        assert freq_ax.bins == image.shape[1], 'image and freq_ax have incompatible shapes'
        self.freq_ax = freq_ax
        self.type = spec_type

        if np.ndim(image) == 3:
            mul = image.shape[2]
            filename, offset, label = stack_spec_attrs(filename, offset, label, mul)
            if annot:
                assert annot.num_sets() == mul, 'Number of annotation sets ({0}) does not match spectrogram multiplicity ({1})'.format(annot.num_sets(), mul)

        self.offset = offset
        self.filename = filename
        self.label = label
        self.annot = annot

    def deepcopy(self):
        """ Make a deep copy of the spectrogram.

            See https://docs.python.org/2/library/copy.html

            Returns:
                spec: Spectrogram
                    Deep copy.
        """
        spec = copy.deepcopy(self)
        return spec

    def data(self):
        """ Get the underlying data numpy array

            Returns:
                self.image: numpy array
                    Image
        """
        return self.image

    def time_res(self):
        """ Get the time resolution of the spectrogram.

            Returns:
                : float
                    Time resolution in seconds.
        """
        return self.time_ax.bin_width()

    def length(self):
        """ Get spectrogram length in seconds.

            Returns:
                : float
                    Length in seconds
        """
        return self.time_ax.max()

    def freq_min(self):
        """ Get spectrogram minimum frequency in Hz.

            Returns:
                : float
                    Frequency in Hz
        """
        return self.freq_ax.min()

    def freq_max(self):
        """ Get spectrogram maximum frequency in Hz.

            Returns:
                : float
                    Frequency in Hz
        """
        return self.freq_ax.max()

    def annotate(self, label=None, start=None, end=None, freq_min=None, freq_max=None, df=None, spec_id=0):
        """ Add an annotation or a collection of annotations.
        
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
                spec_id: int or tuple
                    Unique identifier of the spectrogram. Only relevant for stacked spectrograms.
        """
        assert self.annot is not None, "Attempting to add annotations to a Spectrogram without an AnnotationHandler object" 

        self.annot.add(label, start, end, freq_min, freq_max, df, spec_id)

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

    def normalize(self):
        """ Normalize spectogram so that values range from 0 to 1
        """
        self.image = self.image - np.min(self.image)
        self.image = self.image / np.max(self.image)

    def crop(self, start=None, end=None, length=None,\
        freq_min=None, freq_max=None, height=None, make_copy=False, **kwargs):
        """ Crop spectogram along time axis, frequency axis, or both.
            
            Args:
                start: float
                    Start time in seconds, measured from the left edge of spectrogram.
                end: float
                    End time in seconds, measured from the left edge of spectrogram.
                length: int
                    Horizontal size of the cropped image (number of pixels). If provided, 
                    the `end` argument is ignored. 
                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                height: int
                    Vertical size of the cropped image (number of pixels). If provided, 
                    the `freq_max` argument is ignored. 
                make_copy: bool
                    Return a cropped copy of the spectrogra. Leaves the present instance 
                    unaffected. Default is False.

            Returns:
                spec: Spectrogram
                    Cropped spectrogram

            Examples: 
                >>> import matplotlib.pyplot as plt
                >>> from.ketos.audio_processing.spectrogram import Spectrogram
                >>> from.ketos.audio_processing.axis import LinearAxis
                >>>
                >>> # Create a spectrogram with shape (20,30), time resolution of 
                >>> # 0.5 s, random pixel values, and a linear frequency axis from 
                >>> # 0 to 300 Hz,
                >>> ax = LinearAxis(bins=30, extent=(0.,300.), label='Frequency (Hz)')
                >>> img = np.random.rand((20,30))
                >>> spec = Spectrogram(image=img, time_res=0.5, spec_type='Mag', freq_ax=ax)
                >>>
                >>> # Draw the spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_orig.png")
                >>> plt.close(fig)
                
                .. image:: ../../../../ketos/tests/assets/tmp/spec_orig.png

                >>> # Crop the spectrogram along time axis
                >>> spec1 = spec.crop(start=2.0, end=4.2, make_copy=True)
        """
        if make_copy:
            spec = self.copy()
        else:
            spec = self

        # crop axes
        bx1, bx2 = self.time_ax.cut(x_min=start, x_max=end, bins=length)
        by1, by2 = self.freq_ax.cut(x_min=freq_min, x_max=freq_max, bins=height)

        # crop image
        spec.image = self.image[bx1:bx2+1, by1:by2+1]

        # crop annotations, if any
        if self.annot:
            self.annot.crop(start=start, end=end, freq_min=freq_min, freq_max=freq_max)

        self.offset += self.time_ax.low_edge(0) #update time offset
        self.time_ax.zero_offset() #shift time axis to start at t=0 

        return spec

    def segment(self, window, step=None):
        """ Divide the time axis into segments of uniform length, which may or may 
            not be overlapping.

            Window length and step size are converted to the nearest integer number 
            of time steps.

            If necessary, the spectrogram will be padded with zeros at the end to 
            ensure that all segments have an equal number of samples. 

            Args:
                window: float
                    Length of each segment in seconds.
                step: float
                    Step size in seconds.

            Returns:
                segs: Spectrogram
                    Spectrogram segments
        """              
        if step_size is None:
            step_size = window_size

        time_res = self.time_res()

        win_len = int(round(window / time_res))
        step_len = int(round(step / time_res))

        # segment image
        segs = make_frames(x=self.image, win_len=win_len, step_len=step_len, pad=True, center=False)

        window = win_len * time_res
        step = step_len * time_res
        num_segs = segs.shape[0]

        # segment annotations
        if self.annot:
            annots = self.annot.segment(num_segs=num_segs, window=window, step=step)
        else:
            annots = None

        # compute offsets
        offset = np.arange(num_segs) * step

        ax = copy.deepcopy(self.freq_ax)
        specs = self.__class__(image=segs, time_res=time_res, spec_type=self.type, freq_ax=ax,\
            filename=self.filename, offset=offset, label=self.label, annot=annots)
        
        return specs
                
    def add(self, spec, offset=0, make_copy=False):
        """ Add another spectrogram on top of this spectrogram.

            The spectrograms must be of the same type, and share the same 
            time resolution. 
            
            The spectrograms must have consistent frequency axes. 
            For linear frequency axes, this implies having the same 
            resolution; for logarithmic axes with base 2, this implies having 
            the same number of bins per octave minimum values that differ by 
            a factor of :math:`2^{n/m}` where :math:`m` is the number of bins 
            per octave and :math:`n` is any integer. No check is made for the 
            consistency of the frequency axes.

            Note that the attributes filename, offset, and label of the spectrogram 
            that is being added are lost.

            The sum spectrogram has the same dimensions (time x frequency) as 
            the original spectrogram.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
                offset: float
                    Shift the spectrograms that is being added by this many seconds 
                    relative to the original spectrogram.
                make_copy: bool
                    Make copies of both spectrograms so as to leave the original 
                    instances unchanged.

            Returns:
                : Spectrogram
                    Sum spectrogram
        """
        return aug.add_specs(a=self, b=spec, offset=offset, make_copy=make_copy)

    def plot(self, label=None, pred=None, feat=None, conf=None):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display selected label, binary predictions, features, and confidence levels.

            All plotted quantities share the same time axis, and are assumed to span the 
            same period of time as the spectrogram.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                spec: Spectrogram
                    spectrogram to be plotted
                label: int
                    Label of interest
                pred: 1d array
                    Binary prediction for each time bin in the spectrogram
                feat: 2d array
                    Feature vector for each time bin in the spectrogram
                conf: 1d array
                    Confidence level of prediction for each time bin in the spectrogram
            
            Returns:
            fig: matplotlib.figure.Figure
            A figure object.

            Example:
                >>> # extract saved spectrogram from database file
                >>> import tables
                >>> import ketos.data_handling.database_interface as di
                >>> db = tables.open_file("ketos/tests/assets/cod.h5", "r") 
                >>> table = di.open_table(db, "/sig") 
                >>> spectrogram = di.load_specs(table)[0]
                >>> db.close()
                >>> 
                >>> # plot the spectrogram and label '1'
                >>> import matplotlib.pyplot as plt
                >>> fig = spectrogram.plot(label=1)
                >>> plt.show()
        """
        nrows = 1
        if (label is not None): 
            nrows += 1
        if (pred is not None): 
            nrows += 1
        if (feat is not None): 
            nrows += 1
        if (conf is not None): 
            nrows += 1

        hratio = 1.5/4.0
        figsize=(6, 4.0*(1.+hratio*(nrows-1)))    
        height_ratios = []
        for _ in range(1,nrows):
            height_ratios.append(hratio)

        height_ratios.append(1)
        
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': height_ratios})

        if nrows == 1:
            ax0 = ax
        else:
            ax0 = ax[-1]

        t1 = self.tmin
        t2 = self.tmin+self.duration()

        # spectrogram
        x = self.image
        img_plot = ax0.imshow(x.T, aspect='auto', origin='lower', extent=(t1, t2, self.fmin, self.fmax()))
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Frequency (Hz)')
        fig.colorbar(img_plot, ax=ax0, format='%+2.0f dB')

        row = -2

        # labels
        if label is not None:
            labels = self.get_label_vector(label)
            n = len(labels)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt + self.tmin
            ax[row].plot(t_axis, labels, color='C1')
            ax[row].set_xlim(t1, t2)
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('label')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)
            row -= 1

        # predictions
        if pred is not None:
            n = len(pred)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt + self.tmin
            ax[row].plot(t_axis, pred, color='C2')
            ax[row].set_xlim(t1, t2)
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('prediction')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  
            row -= 1

        # feat
        if feat is not None:
            m = np.mean(feat, axis=0)
            idx = np.argwhere(m != 0)
            idx = np.squeeze(idx)
            x = feat[:,idx]
            x = x / np.max(x, axis=0)
            img_plot = ax[row].imshow(x.T, aspect='auto', origin='lower', extent=(t1, t2, 0, 1))
            ax[row].set_ylabel('feature #')
            fig.colorbar(img_plot, ax=ax[row])
            row -= 1

        # confidence
        if conf is not None:
            n = len(conf)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt + self.tmin
            ax[row].plot(t_axis, conf, color='C3')
            ax[row].set_xlim(t1, t2)
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('confidence')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  
            row -= 1

        return fig


class MagSpectrogram(Spectrogram):
    """ Magnitude Spectrogram computed from Short Time Fourier Transform (STFT)
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins.     
            compute_phase: bool
                Compute phase spectrogram in addition to magnitude spectrogram
            decibel: bool
                Use logarithmic (decibel) scale.
            tag: str
                Identifier, typically the name of the wave file used to generate the spectrogram.
                If no tag is provided, the tag from the audio_signal will be used.
            decibel: bool
                Use logarithmic z axis
            image: 2d numpy array
                Spectrogram matrix. If provided, audio_signal is ignored.
            tmin: float
                Spectrogram start time. Only used if image is provided.
            fres: float
                Spectrogram frequency resolution. Only used if image is provided.
    """
    def __init__(self, audio_signal=None, winlen=None, winstep=1, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, compute_phase=False, decibel=False, tag='',\
                 image=None, tmin=0, fres=1):

        super(MagSpectrogram, self).__init__(timestamp=timestamp, tres=winstep, flabels=flabels, tag=tag, decibel=decibel)

        if image is not None:
            super(MagSpectrogram, self).__init__(image=image, NFFT=NFFT, tres=winstep, tmin=tmin, fres=fres, tag=tag, timestamp=timestamp, flabels=flabels, decibel=decibel)

        elif audio_signal is not None:
            self.image, self.NFFT, self.fres, self.phase_change = self.make_mag_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
            if tag is '':
                self.tag = audio_signal.tag

            self.annotate(labels=audio_signal.labels, boxes=audio_signal.boxes)
            self.tmin = audio_signal.tmin

        self.file_dict, self.file_vector, self.time_vector = self._create_tracking_data(tag) 

    def make_mag_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):
        """ Create magnitude spectrogram from audio signal
        
            Args:
                signal: AudioSignal
                    Audio signal 
                winlen: float
                    Window size in seconds
                winstep: float
                    Step size in seconds 
                hamming: bool
                    Apply Hamming window
                NFFT: int
                    Number of points for the FFT. If None, set equal to the number of samples.
                timestamp: datetime
                    Spectrogram time stamp (default: None)
                compute_phase: bool
                    Compute phase spectrogram in addition to magnitude spectrogram
                decibel: bool
                    Use logarithmic (decibel) scale.
                res_type: str
                    Resampling method. Options: 'kaiser_best' (default), 'kaiser_fast', 'scipy', 'polyphase'.
                    See http://librosa.github.io/librosa/master/generated/librosa.core.resample.html for further details.

            Returns:
                (image, NFFT, fres):numpy.array,int, int
                A tuple with the resulting magnitude spectrogram, the NFFT, the frequency resolution
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self._make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        
        return image, NFFT, fres, phase_change

    @classmethod
    def from_wav(cls, path, spec_config=None, window_size=0.1, step_size=0.01, sampling_rate=None, offset=0, duration=None, channel=0,\
                    decibel=True, adjust_duration=False, fmin=None, fmax=None, window_function='HAMMING', res_type='kaiser_best'):
        """ Create magnitude spectrogram directly from wav file.

            The arguments offset and duration can be used to select a segment of the audio file.

            To ensure that the spectrogram has the desired duration and is centered correctly, the loaded 
            audio segment is slightly longer than the selection at both ends. If no or insufficient audio 
            is available beyond the ends of the selection (e.g. if the selection is the entire audio file), 
            the audio is padded with zeros.

            Note that the duration must be equal to an integer number of steps. If this is not the case, 
            an exception will be raised. Alternatively, you can set adjust_duration to True.

            Note that if spec_config is specified, the following arguments are ignored: 
            sampling_rate, window_size, step_size, duration, fmin, fmax.

            TODO: Modify implementation so that arguments are not ignored when spec_config is specified.

            TODO: Align implementation with the rest of the module.

            TODO: Abstract method to also handle Power, Mel, and CQT spectrograms.
        
            Args:
                path: str
                    Complete path to wav file 
                spec_config: SpectrogramConfiguration
                    Spectrogram configuration
                window_size: float
                    Window size in seconds
                step_size: float
                    Step size in seconds 
                sampling_rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used.
                offset: float
                    Start time of spectrogram in seconds.
                duration: float
                    Duration of spectrogrma in seconds.
                channel: int
                    Channel to read from (for stereo recordings).
                decibel: bool
                    Use logarithmic (decibel) scale.
                adjust_duration: bool
                    If True, the duration is adjusted (upwards) to ensure that the 
                    length corresponds to an integer number of steps.
                fmin: float
                    Minimum frequency in Hz
                fmax: float
                    Maximum frequency in Hz. If None, fmax is set equal to half the sampling rate.
                window_function: str
                    Window function. Ignored for CQT spectrograms.

            Returns:
                spec: MagSpectrogram
                    Magnitude spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio_processing.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window_size=0.2, step_size=0.01)
                >>> # crop frequency
                >>> spec.crop(flow=50, fhigh=800)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_grunt1.png")

                .. image:: ../../../../ketos/tests/assets/tmp/spec_grunt1.png
        """
        if spec_config is not None:
            window_size = spec_config.window_size
            step_size = spec_config.step_size
            fmin = spec_config.low_frequency_cut
            fmax = spec_config.high_frequency_cut
            sampling_rate=spec_config.rate
            duration = spec_config.length
            if spec_config.window_function is not None:
                window_function = WinFun(spec_config.window_function).name
        
        # ensure offset is non-negative
        offset = max(0, offset)

        # ensure selected segment does not exceed file duration
        file_duration = librosa.get_duration(filename=path)
        if duration is None:
            duration = file_duration - offset

        # assert that segment is non-empty
        assert offset < file_duration, 'Selected audio segment is empty'

        # sampling rate
        if sampling_rate is None:
            sr = librosa.get_samplerate(path)
        else:
            sr = sampling_rate

        # segment size 
        seg_siz = int(duration * sr)

        # ensure that window size is an even number of samples
        win_siz = int(round(window_size * sr))
        win_siz += win_siz%2

        # step size
        step_siz = int(step_size * sr)

        # ensure step size is a divisor of the segment size
        res = seg_siz % step_siz
        if not adjust_duration:
            assert res == 0, 'Step size must be a divisor of the audio duration. Consider setting adjust_duration=True.'
        else:
            if res > 0: 
                seg_siz = step_siz * int(np.ceil(seg_siz / step_siz))
                duration = float(seg_siz) / sr

        # number of steps
        num_steps = int(seg_siz / step_siz)

        # padding before / after        
        pad_zeros = [0, 0]

        # padding before
        pad = win_siz / 2
        pad_sec = pad / sr # convert to seconds
        pad_zeros_sec = max(0, pad_sec - offset) # amount of zero padding required
        pad_zeros[0] = round(pad_zeros_sec * sr) # convert to # samples

        # increment duration
        pad_sec -= pad_zeros_sec
        duration += pad_sec

        # reduce offset
        delta_offset = pad_sec

        # padding after        
        pad = max(0, win_siz / 2 - (seg_siz - num_steps * step_siz) - step_siz)
        pad_sec = pad / sr # convert to seconds
        resid = file_duration - (offset - delta_offset + duration)
        pad_zeros_sec = max(0, pad_sec - resid) # amount of zero padding required
        pad_zeros[1] = round(pad_zeros_sec * sr) # convert to # samples

        # increment duration
        pad_sec -= pad_zeros_sec
        duration += pad_sec

        # load audio segment
        x, sr = librosa.core.load(path=path, sr=sampling_rate, offset=offset-delta_offset, duration=duration, mono=False, res_type=res_type)

        # check that loaded audio segment has the expected length.
        # if this is not the case, load the entire audio file and 
        # select the segment of interest manually. 
        N = int(sr * duration)
        if len(x) != N:
            x, sr = librosa.core.load(path=path, sr=sampling_rate, mono=False)
            if np.ndim(x) == 2:
                x = x[channel]

            start = int((offset - delta_offset) * sr)
            num_samples = int(duration * sr)
            stop = min(len(x), start + num_samples)
            x = x[start:stop]

        # check again, pad with zeros to fix any remaining mismatch
        N = round(sr * duration)
        if len(x) < N:
            z = np.zeros(N-len(x))
            x = np.concatenate((z, x))        

        # parse file name
        fname = os.path.basename(path)

        # select channel
        if np.ndim(x) == 2:
            x = x[channel]

        # pad with zeros
        if pad_zeros[0] > 0:
            z = np.zeros(pad_zeros[0])
            x = np.concatenate((z, x))        
        if pad_zeros[1] > 0:
            z = np.zeros(pad_zeros[1])
            x = np.concatenate((x, z))        

        # make frames
        frames = make_frames(x, winlen=win_siz, winstep=step_siz)

        # Apply Hamming window    
        if window_function == 'HAMMING':
            frames *= np.hamming(frames.shape[1])

        # Compute fast fourier transform
        fft = np.fft.rfft(frames)

        # Compute magnitude
        image = np.abs(fft)

        # Number of points used for FFT
        NFFT = frames.shape[1]
        
        # Frequency resolution
        fres = sr / 2. / image.shape[1]

        # use logarithmic axis
        if decibel:
            image = to_decibel(image)

        spec = cls(image=image, NFFT=NFFT, winstep=step_siz/sr, tmin=offset, fres=fres, tag=fname, decibel=decibel)
        spec.hop = step_siz
        spec.hamming = True

        # crop frequencies
        spec.crop(flow=fmin, fhigh=fmax) 

        return spec


    def audio_signal(self, num_iters=25, phase_angle=0):
        """ Estimate audio signal from magnitude spectrogram.

            Args:
                num_iters: 
                    Number of iterations to perform.
                phase_angle: 
                    Initial condition for phase.

            Returns:
                audio: AudioSignal
                    Audio signal
        """
        mag = self.image
        if self.decibel:
            mag = from_decibel(mag)

        # if the frequency axis has been cropped, pad with zeros
        # along the 2nd axis to ensure that the spectrogram has 
        # the expected shape
        if self.fcroplow > 0 or self.fcrophigh > 0:
            mag = np.pad(mag, pad_width=((0,0),(self.fcroplow,self.fcrophigh)), mode='constant')

        n_fft = self.NFFT
        hop = self.hop

        if self.hamming:
            window = get_window('hamming', n_fft)
        else:
            window = np.ones(n_fft)

        audio = estimate_audio_signal(image=mag, phase_angle=phase_angle, n_fft=n_fft, hop=hop, num_iters=num_iters, window=window)

        # sampling rate of estimated audio signal should equal the old rate
        N = len(audio)
        old_rate = self.fres * 2 * mag.shape[1]
        rate = N / (self.duration() + n_fft/old_rate - self.winstep)
        
        assert abs(old_rate - rate) < 0.1, 'The sampling rate of the estimated audio signal ({0:.1f} Hz) does not match the original signal ({1:.1f} Hz).'.format(rate, old_rate)

        audio = AudioSignal(rate=rate, data=audio)

        return audio


class PowerSpectrogram(Spectrogram):
    """ Creates a Power Spectrogram from an :class:`audio_signal.AudioSignal`
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels:list of strings
                List of labels for the frequency bins.
            compute_phase: bool
                Compute phase spectrogram in addition to power spectrogram                        
            decibel: bool
                Use logarithmic (decibel) scale.
            tag: str
                Identifier, typically the name of the wave file used to generate the spectrogram
            decibel: bool
                Use logarithmic z axis
    """
    def __init__(self, audio_signal, winlen, winstep,flabels=None,
                 hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False, tag=''):

        super(PowerSpectrogram, self).__init__(timestamp=timestamp, tres=winstep, flabels=flabels, tag=tag, decibel=decibel)

        if audio_signal is not None:
            self.image, self.NFFT, self.fres, self.phase_change = self.make_power_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
            if tag is '':
                tag = audio_signal.tag

        self.file_dict, self.file_vector, self.time_vector = self._create_tracking_data(tag) 

    def make_power_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):
        """ Create spectrogram from audio signal
        
            Args:
                signal: AudioSignal
                    Audio signal 
                winlen: float
                    Window size in seconds
                winstep: float
                    Step size in seconds 
                hamming: bool
                    Apply Hamming window
                NFFT: int
                    Number of points for the FFT. If None, set equal to the number of samples.
                timestamp: datetime
                    Spectrogram time stamp (default: None)
                compute_phase: bool
                    Compute phase spectrogram in addition to power spectrogram
                decibel: bool
                    Use logarithmic (decibel) scale.

            Returns:
                (power_spec, NFFT, fres, phase):numpy.array,int,int,numpy.array
                A tuple with the resulting power spectrogram, the NFFT, the frequency resolution, 
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self._make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        power_spec = (1.0/NFFT) * (image ** 2)
        
        return power_spec, NFFT, fres, phase_change

       
    
class MelSpectrogram(Spectrogram):
    """ Creates a Mel Spectrogram from an :class:`audio_signal.AudioSignal`
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins.
            tag: str
                Identifier, typically the name of the wave file used to generate the spectrogram
            decibel: bool
                Use logarithmic z axis
    """


    def __init__(self, audio_signal, winlen, winstep,flabels=None, hamming=True, 
                 NFFT=None, timestamp=None, tag='', decibel=False, **kwargs):

        super(MelSpectrogram, self).__init__(timestamp=timestamp, tres=winstep, flabels=flabels, tag=tag, decibel=decibel)

        if audio_signal is not None:
            self.image, self.filter_banks, self.NFFT, self.fres = self.make_mel_spec(audio_signal, winlen, winstep, hamming=hamming, NFFT=NFFT, timestamp=timestamp, **kwargs)
            if tag is '':
                tag = audio_signal.tag

        self.file_dict, self.file_vector, self.time_vector = self._create_tracking_data(tag) 


    def make_mel_spec(self, audio_signal, winlen, winstep, n_filters=40,
                         n_ceps=20, cep_lifter=20, hamming=True, NFFT=None, timestamp=None):
        """ Create a Mel spectrogram from audio signal
    
        Args:
            signal: AudioSignal
                Audio signal 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            n_filters: int
                The number of filters in the filter bank.
            n_ceps: int
                The number of Mel-frequency cepstrums.
            cep_lifters: int
                The number of cepstum filters.
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)

        Returns:
            mel_spec: numpy.array
                Array containing the Mel spectrogram
            filter_banks: numpy.array
                Array containing the filter banks
            NFFT: int
                The number of points used for creating the magnitude spectrogram
                (Calculated if not given)
            fres: int
                The calculated frequency resolution
           
        """

        image, NFFT, fres, _ = self._make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, decibel=False)
        power_spec = (1.0/NFFT) * (image ** 2)
        
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (audio_signal.rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / audio_signal.rate)

        fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, n_filters + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(power_spec, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        
        
        mel_spec = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_ceps + 1)] # Keep 2-13
                
        (nframes, ncoeff) = mel_spec.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mel_spec *= lift  
        
        return mel_spec, filter_banks, NFFT, fres

    def plot(self, filter_bank=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                filter_bank: bool
                    Plot the filter banks if True. If false (default) print the mel spectrogram.
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        if filter_bank:
            img = self.filter_banks
        else:
            img = self.image

        fig, ax = plt.subplots()
        img_plot = ax.imshow(img.T,aspect='auto',origin='lower',extent=(self.tmin,self.tmin+self.duration(),self.fmin,self.fmax()))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        if self.decibel:
            fig.colorbar(img_plot,format='%+2.0f dB')
        else:
            fig.colorbar(img_plot,format='%+2.0f')  
        return fig


class CQTSpectrogram(Spectrogram):
    """ Magnitude Spectrogram computed from Constant Q Transform (CQT) using the librosa implementation:

            https://librosa.github.io/librosa/generated/librosa.core.cqt.html

        The time axis (0th axis) is characterized by a 
        starting value, :math:`t_{min}`, and a bin size, :math:`t_{res}`, while the 
        frequency axis (1st axis) is characterized by a starting value, :math:`f_{min}`, 
        a maximum value, :math:`f_{max}`, and the number of bins per octave, 
        :math:`m`.
        The parameters :math:`t_{min}`, :math:`f_{min}`, :math:`m` are specified via the arguments 
        `tmin`, `fmin`, `bins_per_octave`. The parameters :math:`t_{res}` and :math:`f_{max}`, on the other hand, 
        are computed as detailed below, attempting to match the arguments `winstep` and `fmax` as closely as possible.

        The total number of bins is given by :math:`n = k \cdot m` where :math:`k` denotes 
        the number of octaves, computed as 

        .. math::
            k = ceil(log_{2}[f_{max}/f_{min}])

        For example, with :math:`f_{min}=10`, :math:`f_{max}=16000`, and :math:`m = 32` the number 
        of octaves is :math:`k = 11` and the total number of bins is :math:`n = 352`.  
        The frequency of a given bin, :math:`i`, is given by 

        .. math:: 
            f_{i} = 2^{i / m} \cdot f_{min}

        This implies that the maximum frequency is given by :math:`f_{max} = f_{n} = 2^{n/m} \cdot f_{min}`.
        For the above example, we find :math:`f_{max} = 20480` Hz, i.e., somewhat larger than the requested maximum value.

        Note that if :math:`f_{max}` exceeds the Nyquist frequency, :math:`f_{nyquist} = 0.5 \cdot s`, where :math:`s` is the sampling rate,  
        the number of octaves, :math:`k`, is reduced to ensure that :math:`f_{max} \leq f_{nyquist}`. 

        The CQT algorithm requires the step size to be an integer multiple :math:`2^k`.
        To ensure that this is the case, the step size is computed as follows,

        .. math::
            h = ceil(s \cdot x / 2^k ) \cdot 2^k

        where :math:`s` is the sampling rate in Hz, and :math:`x` is the step size 
        in seconds as specified via the argument `winstep`.
        For example, assuming a sampling rate of 32 kHz (:math:`s = 32000`) and a step 
        size of 0.02 seconds (:math:`x = 0.02`) and adopting the same frequency limits as 
        above (:math:`f_{min}=10` and :math:`f_{max}=16000`), the actual 
        step size is determined to be :math:`h = 2^{11} = 2048`, corresponding 
        to a physical bin size of :math:`t_{res} = 2048 / 32000 Hz = 0.064 s`, i.e., about three times as large 
        as the requested step size.

    
        Args:
            signal: AudioSignal
                And instance of the :class:`audio_signal.AudioSignal` class 
            image: 2d numpy array
                Spectrogram image. Only applicable if signal is None.
            fmin: float
                Minimum frequency in Hz
            fmax: float
                Maximum frequency in Hz. If None, fmax is set equal to half the sampling rate.
            winstep: float
                Step size in seconds 
            bins_per_octave: int
                Number of bins per octave
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins.     
            decibel: bool
                Use logarithmic (decibel) scale.
            tag: str
                Identifier, typically the name of the wave file used to generate the spectrogram.
                If no tag is provided, the tag from the audio_signal will be used. 
    """

    def __init__(self, audio_signal=None, image=np.zeros((2,2)), fmin=1, fmax=None, winstep=0.01, bins_per_octave=32, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, compute_phase=False, decibel=False, tag=''):

        if fmin is None:
            fmin = 1

        super(CQTSpectrogram, self).__init__(timestamp=timestamp, tres=winstep, flabels=flabels, tag=tag, decibel=decibel)
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave

        if audio_signal is not None:

            self.image, self.tres = self.make_cqt_spec(audio_signal, fmin, fmax, winstep, bins_per_octave, decibel)

            if tag is '':
                tag = audio_signal.tag

            self.annotate(labels=audio_signal.labels, boxes=audio_signal.boxes)
            self.tmin = audio_signal.tmin

        else:
            self.image = image
            self.tres = winstep

        self.file_dict, self.file_vector, self.time_vector = self._create_tracking_data(tag) 


    def make_cqt_spec(self, audio_signal, fmin, fmax, winstep, bins_per_octave, decibel):
        """ Create CQT spectrogram from audio signal
        
            Args:
                signal: AudioSignal
                    Audio signal 
                fmin: float
                    Minimum frequency in Hz
                fmax: float
                    Maximum frequency in Hz. If None, fmax is set equal to half the sampling rate.
                winstep: float
                    Step size in seconds 
                bins_per_octave: int
                    Number of bins per octave
                decibel: bool
                    Use logarithmic (decibel) scale.

            Returns:
                (image, tres):numpy.array,float
                A tuple with the resulting magnitude spectrogram, and the time resolution
        """
        f_nyquist = 0.5 * audio_signal.rate
        k_nyquist = int(np.floor(np.log2(f_nyquist / fmin)))

        if fmax is None:
            k = k_nyquist
        else:    
            k = int(np.ceil(np.log2(fmax/fmin)))
            k = min(k, k_nyquist)

        h0 = int(2**k)
        b = bins_per_octave
        fbins = k * b

        h = audio_signal.rate * winstep
        r = int(np.ceil(h / h0))
        h = int(r * h0)

        c = cqt(y=audio_signal.data, sr=audio_signal.rate, hop_length=h, fmin=fmin, n_bins=fbins, bins_per_octave=b)
        c = np.abs(c)
        if decibel:
            c = to_decibel(c)
    
        image = np.swapaxes(c, 0, 1)
        
        tres = h / audio_signal.rate

        return image, tres

    def copy(self):
        """ Make a deep copy of the spectrogram.

            Returns:
                spec: CQTSpectrogram
                    Spectrogram copy.
        """
        spec = super().copy()
        spec.bins_per_octave = self.bins_per_octave
        return spec

    def _find_fbin(self, f, truncate=False, roundup=True):
        """ Find bin corresponding to given frequency.

            Returns -1, if f < f_min.
            Returns N, if f > f_max, where N is the number of frequency bins.

            Args:
                f: float
                   Frequency in Hz 
                truncate: bool
                    Return 0 if below the lower range, and N-1 if above the upper range, where N is the number of bins
                roundup: bool
                    Return lower or higher bin number, if value coincides with a bin boundary

            Returns:
                bin: int
                     Bin number
        """
        bin = self.bins_per_octave * np.log2(f / self.fmin)
        bin = int(bin)

        if truncate:
            bin = max(bin, 0)
            bin = min(bin, self.fbins())

        return bin

    def _fbin_low(self, bin):
        """ Get the lower frequency value of the specified frequency bin.

            Args:
                bin: int
                    Bin number
        """
        f = 2**(bin / self.bins_per_octave) * self.fmin
        return f

    def fmax(self):
        """ Get upper range of frequency axis

            Returns:
                fmax: float
                    Maximum frequency in Hz
        """
        fmax = self._fbin_low(self.fbins())
        return fmax

    @classmethod
    def from_wav(cls, path, spec_config=None, step_size=0.01, fmin=1, fmax=None, bins_per_octave=32, sampling_rate=None, offset=0, duration=None, channel=0, decibel=True):
        """ Create CQT spectrogram directly from wav file.

            The arguments offset and duration can be used to select a segment of the audio file.

            To ensure that the spectrogram has the desired duration and is centered correctly, the loaded 
            audio segment is slightly longer than the selection at both ends. If no or insufficient audio 
            is available beyond the ends of the selection (e.g. if the selection is the entire audio file), 
            the audio is padded with zeros.

            Note that if spec_config is specified, the following arguments are ignored: 
            sampling_rate, bins_per_octave, step_size, duration, fmin, fmax, cqt.

            TODO: Modify implementation so that arguments are not ignored when spec_config is specified.

            TODO: Align implementation with the rest of the module.

            TODO: Abstract method to also handle Power, Mel, and CQT spectrograms.
        
            Args:
                path: str
                    Complete path to wav file 
                spec_config: SpectrogramConfiguration
                    Spectrogram configuration
                step_size: float
                    Step size in seconds 
                fmin: float
                    Minimum frequency in Hz
                fmax: float
                    Maximum frequency in Hz. If None, fmax is set equal to half the sampling rate.
                bins_per_octave: int
                    Number of bins per octave
                sampling_rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used.
                offset: float
                    Start time of spectrogram in seconds.
                duration: float
                    Duration of spectrogrma in seconds.
                channel: int
                    Channel to read from (for stereo recordings).
                decibel: bool
                    Use logarithmic (decibel) scale.

            Returns:
                spec: CQTSpectrogram
                    CQT spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio_processing.spectrogram import CQTSpectrogram
                >>> spec = CQTSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', step_size=0.01, fmin=10, fmax=800, bins_per_octave=16)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cqt_grunt1.png")

                .. image:: ../../../../ketos/tests/assets/tmp/cqt_grunt1.png
        """
        if spec_config is not None:
            step_size = spec_config.step_size
            fmin = spec_config.low_frequency_cut
            fmax = spec_config.high_frequency_cut
            bins_per_octave = spec_config.bins_per_octave
            sampling_rate=spec_config.rate
            duration = spec_config.length

        # ensure offset is non-negative
        offset = max(0, offset)

        # ensure selected segment does not exceed file duration
        file_duration = librosa.get_duration(filename=path)
        if duration is None:
            duration = file_duration - offset

        # assert that segment is non-empty
        assert offset < file_duration, 'Selected audio segment is empty'

        # load audio
        x, sr = librosa.core.load(path=path, sr=sampling_rate, offset=offset, duration=duration, mono=False)

        # select channel
        if np.ndim(x) == 2:
            x = x[channel]

        # check that loaded audio segment has the expected length.
        # if this is not the case, load the entire audio file and 
        # select the segment of interest manually. 
        N = int(sr * duration)
        if len(x) != N:
            x, sr = librosa.core.load(path=path, sr=sampling_rate, mono=False)
            if np.ndim(x) == 2:
                x = x[channel]

            start = int(offset * sr)
            num_samples = int(duration * sr)
            stop = min(len(x), start + num_samples)
            x = x[start:stop]

        # if the segment is shorted than expected, pad with zeros
        N = round(sr * duration)
        if len(x) < N:
            z = np.zeros(N-len(x))
            x = np.concatenate([x,z])

        # parse file name
        fname = os.path.basename(path)

        # create audio signal
        a = AudioSignal(rate=sr, data=x, tag=fname, tstart=offset)

        # create CQT spectrogram
        spec = cls(audio_signal=a, fmin=fmin, fmax=fmax, winstep=step_size, bins_per_octave=bins_per_octave,\
                hamming=True, decibel=decibel)

        return spec


    def plot(self, label=None, pred=None, feat=None, conf=None):
        """ Plot the CQT spectrogram with proper axes ranges and labels.

            Optionally, also display selected label, binary predictions, features, and confidence levels.

            All plotted quantities share the same time axis, and are assumed to span the 
            same period of time as the spectrogram.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                spec: Spectrogram
                    spectrogram to be plotted
                label: int
                    Label of interest
                pred: 1d array
                    Binary prediction for each time bin in the spectrogram
                feat: 2d array
                    Feature vector for each time bin in the spectrogram
                conf: 1d array
                    Confidence level of prediction for each time bin in the spectrogram
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        fig = super().plot(label, pred, feat, conf)

        i = np.arange(0, self.fbins(), self.bins_per_octave)
        if i[-1] != self.fbins():
            i = np.concatenate((i, [self.fbins()]))

        ticks = self.fmin + i * (self.fmax() - self.fmin) / self.fbins()
        labels = 2**(i / self.bins_per_octave) * self.fmin
        labels_str = list()
        for l in labels.tolist():
            labels_str.append('{0:.1f}'.format(l))            

        plt.yticks(ticks, labels_str)

        return fig
