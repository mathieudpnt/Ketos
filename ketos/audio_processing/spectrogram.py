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
from ketos.audio_processing.audio import AudioSignal
from ketos.data_handling.parsing import WinFun
from ketos.utils import random_floats, factors
from tqdm import tqdm
import librosa


import copy
import ketos.audio_processing.audio_processing as ap
from ketos.audio_processing.axis import LinearAxis, Log2Axis
from ketos.audio_processing.annotation import AnnotationHandler
from ketos.audio_processing.image import enhance_, reduce_tonal_noise


# TODO: This methods needs updating (or be removed)
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

def add_specs(a, b, offset=0, make_copy=False):
    """ Place two spectrograms on top of one another by adding their 
        pixel values.

        The spectrograms must be of the same type, and share the same 
        time resolution. 
        
        The spectrograms must have consistent frequency axes. 
        For linear frequency axes, this implies having the same 
        resolution; for logarithmic axes with base 2, this implies having 
        the same number of bins per octave minimum values that differ by 
        a factor of :math:`2^{n/m}` where :math:`m` is the number of bins 
        per octave and :math:`n` is any integer. No check is made for the 
        consistency of the frequency axes.

        Note that the attributes filename, offset, and label of spectrogram 
        `b` is being added are lost.

        The sum spectrogram has the same dimensions (time x frequency) as 
        spectrogram `a`.

        Args:
            a: Spectrogram
                Spectrogram
            b: Spectrogram
                Spectrogram to be added
            offset: float
                Shift spectrogram `b` by this many seconds relative to spectrogram `a`.
            make_copy: bool
                Make copies of both spectrograms, leaving the orignal instances 
                unchanged by the addition operation.

        Returns:
            ab: Spectrogram
                Sum spectrogram
    """
    assert a.type == b.type, "It is not possible to add spectrograms with different types"
    assert a.time_res() == b.time_res(), 'It is not possible to add spectrograms with different time resolutions'

    # make copy
    if make_copy:
        ab = a.deepcopy()
    else:
        ab = a

    # compute cropping boundaries for time axis
    start = -offset
    end = a.length() - offset

    # determine position of b within a
    pos_x = a.time_ax.bin(start, truncate=True) #lower left corner time bin
    pos_y = a.freq_ax.bin(b.freq_min(), truncate=True) #lower left corner frequency bin

    # crop spectrogram b
    b = b.crop(start=start, end=end, freq_min=a.freq_min(), freq_max=a.freq_max(), make_copy=make_copy)

    # add the two images
    bins_x = b.image.shape[0]
    bins_y = b.image.shape[1]
    ab.image[pos_x:pos_x+bins_x, pos_y:pos_y+bins_y] += b.image[pos_x:pos_x+bins_x, pos_y:pos_y+bins_y]

    return ab

def mag2pow(img, num_fft):
    """ Convert a Magnitude spectrogram to a Power spectrogram.

        Args:
            img: numpy.array
                Magnitude spectrogram image.
            num_fft: int
                Number of points used for the FFT.
        
        Returns:
            : numpy.array
                Power spectrogram image
    """
    return (1.0 / num_fft) * (img ** 2)

def mag2mel(img, num_fft, rate, num_filters, num_ceps, cep_lifter):
    """ Convert a Magnitude spectrogram to a Mel spectrogram.

        Args:
            img: numpy.array
                Magnitude spectrogram image.
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
            num_filters: int
                The number of filters in the filter bank.
            num_ceps: int
                The number of Mel-frequency cepstrums.
            cep_lifters: int
                The number of cepstum filters.
        
        Returns:
            mel_spec: numpy.array
                Mel spectrogram image
            filter_banks: numpy.array
                Filter banks
    """
    power_spec = mag2pow(img, num_fft)
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((num_fft + 1) * hz_points / rate)

    fbank = np.zeros((num_filters, int(np.floor(num_fft / 2 + 1))))
    for m in range(1, num_filters + 1):
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
    
    mel_spec = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
            
    (nframes, ncoeff) = mel_spec.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mel_spec *= lift  
    
    return mel_spec, filter_banks

def load_audio_for_spec(path, channel, rate, window, step,\
            offset, duration, resample_method):
    """ Load audio data from a wav file for the specific purpose of computing 
        the spectrogram.

        The loaded audio covers a time interval that extends slightly beyond 
        that specified, [offset, offset+duration], as needed to compute the 
        full spectrogram without zero padding at either end. If the lower/upper 
        boundary of the time interval coincidences with the start/end of the 
        audio file so that no more data is available, we pad with zeros to achieve
        the desired length. 

        Args:
            path: str
                Path to wav file
            channel: int
                Channel to read from. Only relevant for stereo recordings
            rate: float
                Desired sampling rate in Hz. If None, the original sampling rate will be used
            window: float
                Window size in seconds that will be used for computing the spectrogram
            step: float
                Step size in seconds that will be used for computing the spectrogram
            offset: float
                Start time of spectrogram in seconds, relative the start of the wav file.
            duration: float
                Length of spectrogrma in seconds.
            resample_method: str
                Resampling method. Only relevant if `rate` is specified. Options are
                    * kaiser_best
                    * kaiser_fast
                    * scipy (default)
                    * polyphase
                See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                for details on the individual methods.

        Returns:
            audio: AudioSignal
                The audio signal
            seg_args: tuple(int,int,int,int)
                Input arguments for :func:`audio_processing.audio_processing.make_segment`
    """
    if rate is None:
        rate = librosa.get_samplerate(path) #if not specified, use original sampling rate

    file_duration = librosa.get_duration(filename=path) #get file duration
    file_len = int(file_duration * rate) #file length (number of samples)
    
    assert offset < file_duration, 'Offset exceeds file duration'
    
    if duration is None:
        duration = file_duration - offset #if not specified, use file duration minus offset

    duration = min(duration, file_duration - offset) # cap duration at end of file minus offset

    # compute segmentation parameters
    num_frames, offset_len, win_len, step_len = ap.segment_args(rate=rate, duration=duration,\
        offset=offset, window=window, step=step)
    com_len = int(num_frames * step_len + win_len) #combined length of frames

    # convert back to seconds and compute required amount of zero padding
    pad_left = max(0, -offset_len)
    pad_right = max(0, (offset_len + com_len) - file_len)
    load_offset = max(0, offset_len) / rate
    audio_len = int(com_len - pad_left - pad_right)
    duration = audio_len / rate

    # load audio segment
    x, rate = librosa.core.load(path=path, sr=rate, offset=load_offset,\
        duration=duration, mono=False, res_type=resample_method)

    # select channel (for stereo only)
    if np.ndim(x) == 2:
        x = x[channel]

    # check that loaded audio segment has the expected length (give or take 1 sample).
    # if this is not the case, load the entire audio file into memory, then cut out the 
    # relevant section. 
    if abs(len(x) - audio_len) > 1:
        x, rate = librosa.core.load(path=path, sr=rate, mono=False)
        if np.ndim(x) == 2:
            x = x[channel]

        a = max(0, offset_len)
        b = a + audio_len
        x = x[a:b]

    # pad with own reflection
    pad_right += max(0, len(x) - com_len))
    x = ap.pad_reflect(x, pad_left=pad_left, pad_right=pad_right)

    # create AudioSignal object
    filename = os.path.basename(path) #parse file name
    audio = AudioSignal(data=x, rate=rate, filename=filename, offset=offset)

    seg_args = (num_segs, offset_len, win_len, step_len)

    return audio, seg_args

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

    def data(self, spec_id=None):
        """ Get the pixel values as a numpy array.

            Args:
                spec_id: int
                    Spectrogram ID. Only relevant if the spectrogram object 
                    contains multiple, stacked spectrograms.

            Returns:
                d: numpy array
                    Image
        """
        if spec_id is None or np.ndim(self.image) == 2:
            d = self.image
        else:
            d = self.image[:,:,spec_id]

        return d

    def annotations(self, spec_id=None):
        """ Get annotations.

            Args:
                spec_id: int
                    Spectrogram ID. Only relevant if the spectrogram object 
                    contains multiple, stacked spectrograms.

            Returns:
                ans: pandas DataFrame
                    Annotations 
        """
        if self.annot:
            ans = self.annot.get(set_id=spec_id)
        else:
            ans = None

        return ans

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
        segs = ap.segment(x=self.image, win_len=win_len, step_len=step_len, pad=True, center=False)

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
        return add_specs(a=self, b=spec, offset=offset, make_copy=make_copy)

    def blur(self, sigma_time, sigma_freq=0):
        """ Blur the spectrogram using a Gaussian filter.

            Note that the spectrogram frequency axis must be linear if sigma_freq > 0.

            This uses the Gaussian filter method from the scipy.ndimage package:
            
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

            Args:
                sigma_time: float
                    Gaussian kernel standard deviation along time axis in seconds. 
                    Must be strictly positive.
                sigma_freq: float
                    Gaussian kernel standard deviation along frequency axis in Hz.

            Example:        
                >>> from ketos.audio_processing.spectrogram import Spectrogram
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> import matplotlib.pyplot as plt
                >>> # create audio signal
                >>> s = AudioSignal.morlet(rate=1000, frequency=300, width=1)
                >>> # create spectrogram
                >>> spec = MagSpectrogram(s, winlen=0.2, winstep=0.05)
                >>> # show image
                >>> spec.plot()
                <Figure size 600x400 with 2 Axes>
                
                >>> plt.show()
                >>> plt.close()
                >>> # apply very small amount (0.01 sec) of horizontal blur
                >>> # and significant amount of vertical blur (30 Hz)  
                >>> spec.blur_gaussian(tsigma=0.01, fsigma=30)
                >>> # show blurred image
                >>> spec.plot()
                <Figure size 600x400 with 2 Axes>

                >>> plt.show()
                >>> plt.close()
                
                .. image:: ../../_static/morlet_spectrogram.png

                .. image:: ../../_static/morlet_spectrogram_blurred.png
        """
        assert sigma_time > 0, "sigma_time must be strictly positive"
        sig_t = sigma_time / self.time_res()

        if sigma_freq > 0:
            assert isinstance(self.freq_ax, LinearAxis), "Frequency axis must be linear when sigma_freq > 0"
            sig_f = sigma_freq / self.freq_ax.bin_width()
        else:
            sig_f = 0

        self.image = ndimage.gaussian_filter(input=self.image, sigma=(sig_t, sig_f))

    def enhance_signal(self, enhancement=1.):
        """ Enhance the contrast between regions of high and low intensity.

            See :func:`audio_processing.image.enhance_image` for implementation details.

            Args:
                enhancement: float
                    Parameter determining the amount of enhancement.
        """
        self.image = enhance_signal(self.image, enhancement=enhancement)

    def reduce_tonal_noise(self, method='MEDIAN', **kwargs):
        """ Reduce continuous tonal noise produced by e.g. ships and slowly varying 
            background noise

            See :func:`audio_processing.image.reduce_tonal_noise` for implementation details.

            Currently, offers the following two methods:

                1. MEDIAN: Subtracts from each row the median value of that row.
                
                2. RUNNING_MEAN: Subtracts from each row the running mean of that row.
                
            The running mean is computed according to the formula given in 
            Baumgartner & Mussoline, JASA 129, 2889 (2011); doi: 10.1121/1.3562166

            Args:
                method: str
                    Options are 'MEDIAN' and 'RUNNING_MEAN'
            
            Optional args:
                time_constant: float
                    Time constant in seconds, used for the computation of the running mean.
                    Must be provided if the method 'RUNNING_MEAN' is chosen.

            Example:
                >>> # read audio file
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> aud = AudioSignal.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # compute the spectrogram
                >>> from ketos.audio_processing.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram(aud, winlen=0.2, winstep=0.02, decibel=True)
                >>> # keep only frequencies below 800 Hz
                >>> spec.crop(fhigh=800)
                >>> # show spectrogram as is
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_before_tonal.png")
                >>> plt.close(fig)
                >>> # tonal noise reduction
                >>> spec.tonal_noise_reduction()
                >>> # show modified spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_after_tonal.png")
                >>> plt.close(fig)

                .. image:: ../../../../ketos/tests/assets/tmp/spec_before_tonal.png

                .. image:: ../../../../ketos/tests/assets/tmp/spec_after_tonal.png

        """
        time_const_len = kwargs['time_constant'] / self.time_ax.bin_width()
        self.image = reduce_tonal_noise(self.image, method=method, time_const_len=time_const_len)

    def plot(self, spec_id=0, show_annot=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the spectrogram.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                spec_id: int
                    Spectrogram to be plotted. Only relevant if the spectrogram object 
                    contains multiple, stacked spectrograms.
                show_annot: bool
                    Display annotations
            
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
        # create canvas and axes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7), sharex=True)

        # select image data
        x = self.data(spec_id)

        # axes ranges
        extent = (0., self.length(), self.freq_min(), self.freq_max())

        # draw image
        img = ax.imshow(x.T, aspect='auto', origin='lower', extent=extent)

        # axis labels
        ax.set_xlabel(self.time_ax.label)
        ax.set_ylabel(self.freq_ax.label)

        # colobar
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

        # title
        title = ""
        if self.filename: title += "{0}".format(self.filename)           
        if self.label:
            if len(title) > 0: title += ", "
            title += "{0}".format(self.label)

        fig.title(title)

        # if offset is non-zero, add a second time axis at the top 
        # showing the `absolute` time
        if self.offset != 0:
            axt = ax.twiny()
            axt.set_xlim(offset, offset + self.length())

        # superimpose annotation boxes
        if show_annot:
            ans = self.annotations(spec_id)
            if ans:
                print('Drawing of annotations not yet implemented')
            
        fig.tight_layout()
        return fig

class MagSpectrogram(Spectrogram):
    """ Create a Magnitude Spectrogram from an :class:`audio_signal.AudioSignal` by 
        computing the Short Time Fourier Transform (STFT).
    
        Args:
            audio: AudioSignal
                Audio signal 
            window: float
                Window length in seconds
            step: float
                Step size in seconds
            seg_args: dict
                Input arguments used for evaluating :func:`audio_processing.audio_processing.segment_args`. 
                Optional. If specified, the arguments `window` and `step` are ignored.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning

        Attrs:
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
    """
    def __init__(self, audio, window=None, step=None, seg_args=None, window_func='hamming'):

        # compute STFT
        img, freq_max, num_fft, seg_args = ap.stft(x=audio.data, rate=audio.rate, window=window,\
            step=step, seg_args=seg_args, window_func=window_func)

        # create frequency axis
        ax = LinearAxis(bins=img.shape[1], extent=(0., freq_max), label='Frequency (Hz)')

        # create spectrogram
        super().__init__(image=img, time_res=step, spec_type='Mag', freq_ax=ax,\
            filename=audio.filename, offset=audio.offset, label=audio.label,\
            annot=audio.annot)

        # store number of points used for FFT, sampling rate, and window function
        self.num_fft = num_fft
        self.rate = audio.rate
        self.window_func = window_func
        self.seg_args = seg_args

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None,\
            window_func='hamming', offset=0, duration=None,\
            resample_method='scipy'):
        """ Create magnitude spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning
                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogram in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.

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
        # load audio
        audio, seg_args = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,\
            offset=offset, duration=duration, resample_method=resample_method)

        # compute spectrogram
        cls(audio=audio, seg_args=seg_args, window_func=window_func)

        return spec

    def recover_audio(self, num_iters=25, phase_angle=0):
        """ Estimate audio signal from magnitude spectrogram.

            Uses :func:`audio_processing.audio_processing.spec2audio`.

            Args:
                num_iters: 
                    Number of iterations to perform.
                phase_angle: 
                    Initial condition for phase.

            Returns:
                audio: AudioSignal
                    Audio signal
        """
        mag = ap.from_decibel(self.image) #use linear scale

        # if the frequency axis has been cropped, pad with zeros to ensure that 
        # the spectrogram has the expected shape
        pad_low = max(0, -self.freq_ax.bin(0))
        pad_high = max(0, self.freq_ax.bin(self.rate / 2, closed_right=True) - self.freq_ax.bins)

        if pad_low or pad_high > 0:
            mag = np.pad(mag, pad_width=((0,0),(pad_low,pad_high)), mode='constant')

        # retrieve settings used for computing STFT
        num_fft = self.num_fft
        step_len = self.seg_args['step_len']
        if self.window_func:
            window_func = get_window(self.window_func, num_fft)
        else:
            window_func = np.ones(num_fft)

        # iteratively estimate audio signal
        audio = ap.spec2audio(image=mag, phase_angle=phase_angle, num_fft=num_fft,\
            step_len=step_len, num_iters=num_iters, window_func=window_func)

        # sampling rate of recovered audio signal should equal the original rate
        rate_orig = self.time_ax.bin_width() * 2 * mag.shape[1]
        rate = len(audio) / (self.length() + (num_fft - step_len) / rate_orig)
        
        assert abs(old_rate - rate) < 0.1, 'The sampling rate of the recovered audio signal ({0:.1f} Hz) does not match that of the original signal ({1:.1f} Hz).'.format(rate, old_rate)

        audio = AudioSignal(rate=rate, data=audio)

        return audio

class PowerSpectrogram(Spectrogram):
    """ Create a Power Spectrogram from an :class:`audio_signal.AudioSignal` by 
        computing the Short Time Fourier Transform (STFT).
    
        Args:
            audio: AudioSignal
                Audio signal 
            window: float
                Window length in seconds
            step: float
                Step size in seconds 
            seg_args: dict
                Input arguments used for evaluating :func:`audio_processing.audio_processing.segment_args`. 
                Optional. If specified, the arguments `window` and `step` are ignored.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning

        Attrs:
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
    """
    def __init__(self, audio, window=None, step=None, seg_args=None, window_func='hamming'):

        # compute STFT
        img, freq_max, num_fft = ap.stft(x=audio.data, rate=audio.rate, window=window,\
            step=step, seg_args=seg_args, window_func=window_func)
        img = mag2pow(img, num_fft) # Magnitude->Power conversion

        # create frequency axis
        ax = LinearAxis(bins=img.shape[1], extent=(0., freq_max), label='Frequency (Hz)')

        # create spectrogram
        super().__init__(image=img, time_res=step, spec_type='Pow', freq_ax=ax,\
            filename=audio.filename, offset=audio.offset, label=audio.label,\
            annot=audio.annot)

        # store number of points used for FFT and sampling rate
        self.num_fft = num_fft
        self.rate = audio.rate

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None,\
            window_func='hamming', offset=0, duration=None,\
            resample_method='scipy'):
        """ Create power spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning
                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.

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
        # load audio
        audio, seg_args = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,\
            offset=offset, duration=duration, resample_method=resample_method)

        # compute spectrogram
        cls(audio=audio, seg_args=seg_args, window_func=window_func)

        return spec

class MelSpectrogram(Spectrogram):
    """ Creates a Mel Spectrogram from an :class:`audio_signal.AudioSignal`.

        Args:
            audio: AudioSignal
                Audio signal 
            window: float
                Window length in seconds
            step: float
                Step size in seconds 
            seg_args: dict
                Input arguments used for evaluating :func:`audio_processing.audio_processing.segment_args`. 
                Optional. If specified, the arguments `window` and `step` are ignored.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning
            num_filters: int
                The number of filters in the filter bank.
            num_ceps: int
                The number of Mel-frequency cepstrums.
            cep_lifters: int
                The number of cepstum filters.

        Attrs:
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
            filter_banks: numpy.array
                Filter banks
    """
    def __init__(self, audio, window=None, step=None, seg_args=None, window_func='hamming',\
            num_filters=40, num_ceps=20, cep_lifter=20):

        # compute STFT
        img, freq_max, num_fft = ap.stft(x=audio.data, rate=audio.rate, window=window,\
            step=step, seg_args=seg_args, window_func=window_func)
        img, filter_banks = mag2mel(img, audio.rate, num_filters, num_ceps, cep_lifter) # Magnitude->Mel conversion

        # create frequency axis
        # TODO: This probably needs to be modified ...
        ax = LinearAxis(bins=img.shape[1], extent=(0., freq_max), label='Frequency (Hz)')

        # create spectrogram
        super().__init__(image=img, time_res=step, spec_type='Mel', freq_ax=ax,\
            filename=audio.filename, offset=audio.offset, label=audio.label,\
            annot=audio.annot)

        # store number of points used for FFT, sampling rate, and filter banks
        self.num_fft = num_fft
        self.rate = audio.rate
        self.filter_banks

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None,\
            window_func='hamming', num_filters=40, num_ceps=20, cep_lifter=20,\
            offset=0, duration=None, resample_method='scipy'):
        """ Create Mel spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning
                num_filters: int
                    The number of filters in the filter bank.
                num_ceps: int
                    The number of Mel-frequency cepstrums.
                cep_lifters: int
                    The number of cepstum filters.
                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.

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
        # load audio
        audio, seg_args = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,\
            offset=offset, duration=duration, resample_method=resample_method)

        # compute spectrogram
        cls(audio=audio, seg_args=seg_args, window_func=window_func)

        return spec

    def plot(self, filter_bank=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            TODO: Check implementation for filter_bank=True

            Args:
                filter_bank: bool
                    If True, plot the filter banks if True. If False (default), 
                    print the mel spectrogram.
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        if filter_bank:
            img = self.filter_banks
            fig, ax = plt.subplots()
            extent = (0,self.length,self.freq_min(),self.freq_max())
            img_plot = ax.imshow(img.T,aspect='auto',origin='lower',extent=extent)
            ax.set_xlabel(self.time_ax.label)
            ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(img_plot,format='%+2.0f dB')

        else:
            fig = super().plot()

        return fig

class CQTSpectrogram(Spectrogram):
    """ Magnitude Spectrogram computed from Constant Q Transform (CQT) using the librosa implementation:

            https://librosa.github.io/librosa/generated/librosa.core.cqt.html

        The frequency axis of a CQT spectrogram is essentially a logarithmic axis with base 2. It is 
        characterized by an integer number of bins per octave (an octave being a doubling of the frequency.) 

        For further details, see :func:`audio_processing.audio_processing.cqt`.

        Args:
            audio: AudioSignal
                Audio signal 
            step: float
                Step size in seconds 
            bins_per_oct: int
                Number of bins per octave
            freq_min: float
                Minimum frequency in Hz
                If None, it is set to 1 Hz.
            freq_max: float
                Maximum frequency in Hz. 
                If None, it is set equal to half the sampling rate.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning
    """
    def __init__(self, audio, step, bins_per_oct, freq_min=1, freq_max=None,\
        window_func='hamming'):

        # compute CQT
        img, step = ap.cqt(x=audio.data, rate=audio.rate, step=step,\
            bins_per_oct=bins_per_oct, freq_min=freq_min, freq_max=freq_max)

        # create logarithmic frequency axis
        ax = Log2Axis(bins=img.shape[1], bins_per_oct=bins_per_oct,\
            min_value=freq_min, label='Frequency (Hz)')

        # create spectrogram
        super().__init__(image=img, time_res=step, spec_type='CQT', freq_ax=ax,\
            filename=audio.filename, offset=audio.offset, label=audio.label,\
            annot=audio.annot)

        # store sampling rate
        self.rate = audio.rate

    @classmethod
    def from_wav(cls, path, step, bins_per_oct, freq_min=1, freq_max=None,\
        channel=0, rate=None, window_func='hamming', offset=0, duration=None, \
        resample_method='scipy'):
        """ Create CQT spectrogram directly from wav file.

            The arguments offset and duration can be used to select a segment of the audio file.

            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.
        
            Args:
                path: str
                    Complete path to wav file 
                step: float
                    Step size in seconds 
                bins_per_oct: int
                    Number of bins per octave
                freq_min: float
                    Minimum frequency in Hz
                    If None, it is set to 1 Hz.
                freq_max: float
                    Maximum frequency in Hz. 
                    If None, it is set equal to half the sampling rate.
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning
                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.

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
        if rate is None:
            rate = librosa.get_samplerate(path) #if not specified, use original sampling rate

        file_duration = librosa.get_duration(filename=path) #get file duration
        file_len = int(file_duration * rate) #file length (number of samples)
        
        assert offset < file_duration, 'Offset exceeds file duration'
        
        if duration is None:
            duration = file_duration - offset #if not specified, use file duration minus offset

        duration = min(duration, file_duration - offset) # cap duration at end of file minus offset        

        # load audio
        audio = AudioSignal.from_wav(path=path, rate=rate, channel=channel,\
            offset=offset, duration=duration, res_type=resample_method)

        # create CQT spectrogram
        spec = cls(audio=audio, step=step, bins_per_oct=bins_per_oct, freq_min=freq_min,\
            freq_max=freq_max, window_func=window_func)

        return spec

    def plot(self, spec_id=0, show_annot=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the spectrogram.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                spec_id: int
                    Spectrogram to be plotted. Only relevant if the spectrogram object 
                    contains multiple, stacked spectrograms.
                show_annot: bool
                    Display annotations
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        fig = super().plot(sped_id, show_annot)
        ticks, labels = self.freq_ax.ticks_and_labels()
        plt.yticks(ticks, labels)
        return fig
