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

""" audio module within the ketos library

    This module provides utilities to work with audio data.

    Contents:
        AudioSignal class: 
        TimeStampedAudioSignal class
"""
import os
import numpy as np
import datetime
import math
import scipy.io.wavfile as wave
from scipy import interpolate
from ketos.utils import morlet_func
import ketos.audio_processing.audio_processing as ap
import matplotlib.pyplot as plt
from scipy.integrate import quadrature
from scipy.stats import norm
from tqdm import tqdm
from ketos.utils import ensure_dir


import librosa
from ketos.data_handling.data_handling import read_wave
from ketos.audio_processing.annotation import AnnotationHandler
import ketos.audio_processing.audio_processing as ap
from ketos.audio_processing.axis import LinearAxis


class AudioSignal():
    """ Audio signal

        Args:
            rate: float
                Sampling rate in Hz
            data: numpy array
                Audio data 
            filename: str
                Filename of the original audio file, if available (optional)
            offset: float
                Position within the original audio file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional

        Attributes:
            rate: float
                Sampling rate in Hz
            data: 1numpy array
                Audio data 
            time_ax: LinearAxis
                Axis object for the time dimension
            filename: str
                Filename of the original audio file, if available (optional)
            offset: float
                Position within the original audio file, in seconds 
                measured from the start of the file. Defaults to 0 if not specified.
            label: int
                Spectrogram label.
            annot: AnnotationHandler
                AnnotationHandler object.
    """
    def __init__(self, rate, data, filename='', offset=0, label=None, annot=None):
        self.rate = rate
        self.data = data.astype(dtype=np.float32)
        length = rate * data.shape[0]
        self.time_ax = LinearAxis(bins=data.shape[0], extent=(0., length), label='Time (s)') #initialize time axis

        if np.ndim(image) == 2:
            mul = image.shape[1]
            filename, offset, label = ap.stack_audio_attrs(filename, offset, label, mul)
            if annot:
                assert annot.num_sets() == mul, 'Number of annotation sets ({0}) does not match spectrogram multiplicity ({1})'.format(annot.num_sets(), mul)

        self.filename = filename
        self.offset = offset
        self.label = label
        self.annot = annot

    @classmethod
    def from_wav(cls, path, channel=0, rate=None, offset=0, duration=None, resample_method='scipy'):
        """ Load audio data from wave file.

            Args:
                path: str
                    Path to input wave file
                channel: int
                    In the case of stereo recordings, this argument is used 
                    to specify which channel to read from. Default is 0.
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                offset: float
                    Position within the original audio file, in seconds 
                    measured from the start of the file. Defaults to 0 if not specified.
                duration: float
                    Length in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.

            Returns:
                Instance of AudioSignal
                    Audio signal

            Example:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # read audio signal from wav file
                >>> a = AudioSignal.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # show signal
                >>> fig = a.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/audio_grunt1.png")

                .. image:: ../../../../ketos/tests/assets/tmp/audio_grunt1.png
        """
        rate_orig = librosa.get_samplerate(path)
        start = num_samples(offset, rate_orig)
        if duration:
            stop = num_samples(offset + duration, rate_orig)

        rate_orig, data = read_wave(file=path, channel=channel, start=start, stop=stop)

        if rate and rate != rate_orig:
            data = librosa.core.resample(data, orig_sr=rate_orig, target_sr=rate, res_type=resample_method)
            
        return cls(rate=rate, data=data, filename=os.path.basename(path), offset=offset)

    @classmethod
    def gaussian_noise(cls, rate, sigma, samples, filename="gaussian_noise"):
        """ Generate Gaussian noise signal

            Args:
                rate: float
                    Sampling rate in Hz
                sigma: float
                    Standard deviation of the signal amplitude
                samples: int
                    Length of the audio signal given as the number of samples
                filename: str
                    Meta-data string (optional)

            Returns:
                Instance of AudioSignal
                    Audio signal sampling of Gaussian noise

            Example:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create gaussian noise with sampling rate of 10 Hz, standard deviation of 2.0 and 1000 samples
                >>> a = AudioSignal.gaussian_noise(rate=10, sigma=2.0, samples=1000)
                >>> # show signal
                >>> fig = a.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/audio_noise.png")

                .. image:: ../../../../ketos/tests/assets/tmp/audio_noise.png
        """        
        assert sigma > 0, "sigma must be strictly positive"

        y = np.random.normal(loc=0, scale=sigma, size=samples)
        return cls(rate=rate, data=y, filename=filename)

    @classmethod
    def morlet(cls, rate, frequency, width, samples=None, height=1, displacement=0, dfdt=0, filename="morlet"):
        """ Audio signal with the shape of the Morlet wavelet

            Uses :func:`util.morlet_func` to compute the Morlet wavelet.

            Args:
                rate: float
                    Sampling rate in Hz
                frequency: float
                    Frequency of the Morlet wavelet in Hz
                width: float
                    Width of the Morlet wavelet in seconds (sigma of the Gaussian envelope)
                samples: int
                    Length of the audio signal given as the number of samples (if no value is given, samples = 6 * width * rate)
                height: float
                    Peak value of the audio signal
                displacement: float
                    Peak position in seconds
                dfdt: float
                    Rate of change in frequency as a function of time in Hz per second.
                    If dfdt is non-zero, the frequency is computed as 

                        f = frequency + (time - displacement) * dfdt 

                filename: str
                    Meta-data string (optional)

            Returns:
                Instance of AudioSignal
                    Audio signal sampling of the Morlet wavelet 

            Examples:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a Morlet wavelet with frequency of 3 Hz and 1-sigma width of envelope set to 2.0 seconds
                >>> wavelet1 = AudioSignal.morlet(rate=100., frequency=3., width=2.0)
                >>> # show signal
                >>> fig = wavelet1.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_standard.png")

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_standard.png

                >>> # create another wavelet, but with frequency increasing linearly with time
                >>> wavelet2 = AudioSignal.morlet(rate=100., frequency=3., width=2.0, dfdt=0.3)
                >>> # show signal
                >>> fig = wavelet2.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_dfdt.png")

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_dfdt.png
        """        
        if samples is None:
            samples = int(6 * width * rate)

        N = int(samples)

        # compute Morlet function at N equally spaced points
        dt = 1. / rate
        stop = (N-1.)/2. * dt
        start = -stop
        time = np.linspace(start, stop, N)
        y = morlet_func(time=time, frequency=frequency, width=width, displacement=displacement, norm=False, dfdt=dfdt)        
        y *= height
        
        return cls(rate=rate, data=np.array(y), filename=filename)

    @classmethod
    def cosine(cls, rate, frequency, duration=1, height=1, displacement=0, filename="cosine"):
        """ Audio signal with the shape of a cosine function

            Args:
                rate: float
                    Sampling rate in Hz
                frequency: float
                    Frequency of the Morlet wavelet in Hz
                duration: float
                    Duration of the signal in seconds
                height: float
                    Peak value of the audio signal
                displacement: float
                    Phase offset in fractions of 2*pi
                filename: str
                    Meta-data string (optional)

            Returns:
                Instance of AudioSignal
                    Audio signal sampling of the cosine function 

            Examples:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a Cosine wave with frequency of 7 Hz
                >>> cos = AudioSignal.cosine(rate=1000., frequency=7.)
                >>> # show signal
                >>> fig = cos.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cosine_audio.png")

                .. image:: ../../../../ketos/tests/assets/tmp/cosine_audio.png
        """        
        N = int(duration * rate)

        # compute cosine function at N equally spaced points
        dt = 1. / rate
        stop = (N-1.)/2. * dt
        start = -stop
        time = np.linspace(start, stop, N)
        x = (time * frequency + displacement) * 2 * np.pi
        y = height * np.cos(x)
        
        return cls(rate=rate, data=np.array(y), filename=filename)

    def data(self):
        """ Get the underlying data numpy array.
            
            Returns:
                self.data: numpy array
                    Data
        """
        return self.data

    def segment(self, window, step=None):
        """ Divide the time axis into segments of uniform length, which may or may 
            not be overlapping.

            Window length and step size are converted to the nearest integer number 
            of time steps.

            If necessary, the audio signal will be padded with zeros at the end to 
            ensure that all segments have an equal number of samples. 

            Args:
                window: float
                    Length of each segment in seconds.
                step: float
                    Step size in seconds.

            Returns:
                segs: AudioSignal
                    Stacked audio signals
        """              
        if step_size is None:
            step_size = window_size

        time_res = self.time_res()
        win_len = ap.num_samples(window, 1. / time_res)
        step_len = ap.num_samples(step, 1. / time_res)

        # segment audio signal
        segs = ap.segment(x=self.image, win_len=win_len, step_len=step_len, pad_mode='zero')

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

        audio = self.__class__(rate=self.rate, data=segs, filename=self.filename, offset=offset, label=self.label, annot=annots)

        return audio

    def to_wav(self, path, auto_loudness=True):
        """ Save audio signal to wave file

            Args:
                path: str
                    Path to output wave file
                auto_loudness: bool
                    Automatically amplify the signal so that the 
                    maximum amplitude matches the full range of 
                    a 16-bit wav file (32760)
        """        
        ensure_dir(path)
        
        if auto_loudness:
            m = max(1, np.max(np.abs(self.data)))
            s = 32760 / m
        else:
            s = 1

        wave.write(filename=path, rate=int(self.rate), data=(s*self.data).astype(dtype=np.int16))

    def empty(self):
        """ Check if the signal contains any data

            Returns:
                bool
                    True if the length of the data array is zero or array is None
        """  
        if self.data is None:
            return True
        elif len(self.data) == 0:
            return True
        
        return False

    def length(self):
        """ Signal duration in seconds

            Returns:
                : float
                   Signal duration in seconds
        """    
        return self.time_ax.max()

    def max(self):
        """ Maximum value of the signal

            Returns:
                v: float
                   Maximum value of the data array
        """    
        v = max(self.data)
        return v

    def min(self):
        """ Minimum value of the signal

            Returns:
                v: float
                   Minimum value of the data array
        """    
        v = min(self.data)
        return v

    def std(self):
        """ Standard deviation of the signal

            Returns:
                v: float
                   Standard deviation of the data array
        """   
        v = np.std(self.data) 
        return v

    def average(self):
        """ Average value of the signal

            Returns:
                v: float
                   Average value of the data array
        """   
        v = np.average(self.data)
        return v

    def median(self):
        """ Median value of the signal

            Returns:
                v: float
                   Median value of the data array
        """   
        v = np.median(self.data)
        return v

    def plot(self):
        """ Plot the signal with proper axes ranges and labels
            
            Example:            
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a morlet wavelet
                >>> a = AudioSignal.morlet(rate=100, frequency=5, width=1)
                >>> # plot the wave form
                >>> fig = a.plot()

                .. image:: ../../_static/morlet.png
        """
        fig, ax = plt.subplots(nrows=1)
        start = 0.5 / self.rate
        stop = self.length() - 0.5 / self.rate
        num = len(self.data)
        ax.plot(np.linspace(start=start, stop=stop, num=num), self.data)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        return fig

    def annotate(self, label=None, start=None, end=None, df=None, audio_id=0):
        """ Add an annotation or a collection of annotations.
        
            Individual annotations may be added using the arguments start and end.
            
            Groups of annotations may be added by first collecting them in a pandas 
            DataFrame or dictionary and then adding them using the 'df' argument.
        
            Args:
                label: int
                    Integer label.
                start: str or float
                    Start time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                end: str or float
                    Stop time. Can be specified either as a float, in which case the 
                    unit will be assumed to be seconds, or as a string with an SI unit, 
                    for example, '22min'.
                df: pandas DataFrame or dict
                    Annotations stored in a pandas DataFrame or dict. Must have columns/keys 
                    'label', 'start', 'end', and optionally also 'freq_min' 
                    and 'freq_max'.
                spec_id: int or tuple
                    Unique identifier of the spectrogram. Only relevant for stacked spectrograms.
        """
        assert self.annot is not None, "Attempting to add annotations to an AudioSignal without an AnnotationHandler object" 

        self.annot.add(label=label, start=start, end=end, freq_min=None, freq_max=None, df=df, spec_id=audio_id)

    def crop(self, start=None, end=None, length=None, make_copy=False, **kwargs):
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
                a: AudioSignal
                    Cropped audio signal
        """
        if make_copy:
            a = self.deepcopy()
        else:
            a = self

        # crop axis
        b1, b2 = self.time_ax.cut(x_min=start, x_max=end, bins=length)

        # crop audio signal
        a.data = self.data[b1:b2+1]

        # crop annotations, if any
        if self.annot:
            self.annot.crop(start=start, end=end)

        self.offset += self.time_ax.low_edge(0) #update time offset
        self.time_ax.zero_offset() #shift time axis to start at t=0 

        return a

    def append(self, signal, delay=None, n_smooth=0, max_length=None):
        """ Append another audio signal to this signal.

            The two audio signals must have the same samling rate.
            
            If delay is None or 0, a smooth transition is made between the 
            two signals. The width of the smoothing region where the two signals 
            overlap is specified via the argument n_smooth.

            Note that the current implementation of the smoothing procedure is 
            quite slow, so it is advisable to use small overlap regions.

            If n_smooth is 0, the two signals are joint without any smoothing.

            If delay > 0, a signal with zero sound intensity and duration 
            delay is added between the two audio signals. 

            If the length of the combined signal exceeds max_length, only the 
            first max_length samples will be kept. 

            Args:
                signal: AudioSignal
                    Audio signal to be merged.
                delay: float
                    Delay between the two audio signals in seconds. 
                n_smooth: int
                    Width of the smoothing/overlap region (number of samples).
                max_length: int
                    Maximum length of the combined signal (number of samples).

            Returns:
                append_time: float
                    Start time of appended part in seconds from the beginning of the original signal.

            Example:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a morlet wavelet
                >>> mor = AudioSignal.morlet(rate=100, frequency=5, width=1)
                >>> # create a cosine wave
                >>> cos = AudioSignal.cosine(rate=100, frequency=3, duration=4)
                >>> # append the cosine wave to the morlet wavelet, using a overlap of 100 bins
                >>> mor.append(signal=cos, n_smooth=100)
                5.0
                >>> # show the wave form
                >>> fig = mor.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_cosine.png")

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_cosine.png
        """   
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        # if appending signal to itself, make a copy
        if signal is self:
            signal = self.copy()

        if delay is None:
            delay = 0

        delay = max(0, delay)
        if max_length is not None:
            delay = min(max_length / self.rate, delay)

        # ensure that overlap region is shorter than either signals
        n_smooth = min(n_smooth, len(self.data) - 1)
        n_smooth = min(n_smooth, len(signal.data) - 1)

        # compute total length
        len_tot = self.merged_length(signal, delay, n_smooth)

        append_time = len(self.data) / self.rate

        # extract data from overlap region
        if delay == 0 and n_smooth > 0:

            # signal 1
            a = self.split(-n_smooth)

            # signal 2
            b = signal.split(n_smooth)

            # superimpose a and b
            # TODO: If possible, vectorize this loop for faster execution
            # TODO: Cache values returned by _smoothclamp to avoid repeated calculation
            # TODO: Use coarser binning for smoothing function to speed things up even more
            c = np.empty(n_smooth)
            for i in range(n_smooth):
                w = _smoothclamp(i, 0, n_smooth-1)
                c[i] = (1.-w) * a.data[i] + w * b.data[i]
            
            append_time = len(self.data) / self.rate

            # append
            self.data = np.append(self.data, c)

        elif delay > 0:
            z = np.zeros(int(delay * self.rate))
            self.data = np.append(self.data, z)
            append_time = len(self.data) / self.rate
            
        self.data = np.append(self.data, signal.data) 
        
        assert len(self.data) == len_tot # check that length of merged signal is as expected

        # mask inserted zeros
        if delay > 0:
            self.data = np.ma.masked_values(self.data, 0)

        # remove all appended data from signal        
        if max_length is not None:
            if len_tot > max_length:
                self._crop(i1=0, i2=max_length)
                i2 = len(signal.data)
                i1 = max(0, i2 - (len_tot - max_length))
                signal._crop(i1=i1, i2=i2)
            else:
                signal.data = None
        else:
            signal.data = None
        
        # re-init time axis
        length = self.rate * data.shape[0]
        self.time_ax = LinearAxis(bins=data.shape[0], extent=(0., length), label='Time (s)') 

        return append_time

    def merged_length(self, signal=None, delay=None, n_smooth=None):
        """ Compute sample size of merged signal (without actually merging the signals)

            Args:
                signal: AudioSignal
                    Audio signal to be merged
                delay: float
                    Delay between the two audio signals in seconds.

            Returns:
                l: int
                    Merged length
        """   
        if signal is None:
            return len(self.data)

        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        if delay is None:
            delay = 0
            
        if n_smooth is None:
            n_smooth = 0

        m = len(self.data)
        n = len(signal.data)
        l = m + n
        
        if delay > 0:
            l += int(delay * self.rate)
        elif delay == 0:
            l -= n_smooth
        
        return l

    def add_gaussian_noise(self, sigma):
        """ Add Gaussian noise to the signal

            Args:
                sigma: float
                    Standard deviation of the gaussian noise

            Example:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a morlet wavelet
                >>> morlet = AudioSignal.morlet(rate=100, frequency=2.5, width=1)
                >>> morlet_pure = morlet.copy() # make a copy
                >>> # add some noise
                >>> morlet.add_gaussian_noise(sigma=0.3)
                >>> # show the wave form
                >>> fig = morlet_pure.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_wo_noise.png")
                >>> fig = morlet.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_w_noise.png")

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_wo_noise.png

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_w_noise.png
        """
        noise = AudioSignal.gaussian_noise(rate=self.rate, sigma=sigma, samples=len(self.data))
        self.add(noise)

    def add(self, signal, offset=0, scale=1):
        """ Add the amplitudes of the two audio signals.
        
            The audio signals must have the same sampling rates.

            The summed signal always has the same length as the present instance.

            If the audio signals have different lengths and/or a non-zero delay is selected, 
            only the overlap region will be affected by the operation.
            
            If the overlap region is empty, the original signal is unchanged.

            Args:
                signal: AudioSignal
                    Audio signal to be added
                offset: float
                    Shift the audio signal by this many seconds
                scale: float
                    Scaling factor for signal to be added

            Example:
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> # create a morlet wavelet
                >>> mor = AudioSignal.morlet(rate=100, frequency=5, width=1)
                >>> # create a cosine wave
                >>> cos = AudioSignal.cosine(rate=100, frequency=3, duration=4)
                >>> # add the cosine on top of the morlet wavelet, with a delay of 2 sec and a scaling factor of 0.3
                >>> mor.add(signal=cos, delay=2.0, scale=0.3)
                >>> # show the wave form
                >>> fig = mor.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_cosine_added.png")

                .. image:: ../../../../ketos/tests/assets/tmp/morlet_cosine_added.png
        """
        assert self.rate == signal.rate, "Cannot add audio signals with different sampling rates."

        # compute cropping boundaries for time axis
        start = -offset
        end = self.length() - offset

        # convert to bin number
        b = self.time_ax.bin(start, truncate=True)

        # crop signal that is being added
        signal = signal.crop(start=start, end=end)

        # add the two signals
        bins = signal.data.shape[0]
        self.data[b:b+bins] += signal

    def resample(self, new_rate):
        """ Resample the acoustic signal with an arbitrary sampling rate.

        Note: Code adapted from Kahl et al. (2017)
              Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
              Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py  

        Args:
            new_rate: int
                New sampling rate in Hz
        """
        if len(self.data) < 2:
            self.rate = new_rate

        else:                
            orig_rate = self.rate
            sig = self.data

            duration = sig.shape[0] / orig_rate

            time_old  = np.linspace(0, duration, sig.shape[0])
            time_new  = np.linspace(0, duration, int(sig.shape[0] * new_rate / orig_rate))

            interpolator = interpolate.interp1d(time_old, sig.T)
            new_audio = interpolator(time_new).T

            new_sig = np.round(new_audio).astype(sig.dtype)

            self.rate = new_rate
            self.data = new_sig

    def deepcopy(self):
        """ Make a deep copy of the present instance

            See https://docs.python.org/2/library/copy.html

            Returns:
                : AudioSignal
                Deep copy.
        """
        return copy.deepcopy(self)

def _smoothclamp(x, mi, mx): 
        """ Smoothing function
        """    
        return (lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )