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

""" Waveform module within the ketos library

    This module provides utilities to work with audio data.

    Contents:
        Waveform class
"""
import os
import numpy as np
import librosa
import warnings
import scipy.io.wavfile as wave
from scipy import interpolate
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ketos.utils import ensure_dir, morlet_func
from ketos.data_handling.data_handling import read_wave
from ketos.audio.annotation import AnnotationHandler
from ketos.audio.utils.axis import LinearAxis
from ketos.audio.base_audio import BaseAudioTime, segment_data
import ketos.audio.utils.misc as aum


def _validate_wf_args(path, offset, duration):
    ''' Validate and standardize values

        Args:
            path: str or list(str)
                Path to input audio file(s)
            offset: float or list(float)
                Start of segment measured in seconds from the start of the file.
            duration: float or list(float)
                Segment length in seconds.

        Returns:
            path, offset, duration: list
                Validated and standardized values
    '''
    if np.ndim(path) == 0:
        path = [path]

    if np.ndim(offset) == 0:
        offset = [offset for _ in path]

    if np.ndim(duration) == 0:
        duration = [duration for _ in path]

    assert len(offset) == len(path), "offset and path must have the same length"
    assert len(duration) == len(path), "duration and path must have the same length"

    return path, offset, duration


def get_sampling_rate(path):
    ''' Get the (common or lowest) sampling rate of the specified audio segments.

        Args:
            path: str or list(str)
                Path to input audio file(s)

        Returns:
            : float
                Inferred sampling rate in Hz
    '''
    if np.ndim(path) == 0:
        path = [path]
        
    rates = [librosa.get_samplerate(p) for p in path if p is not None]

    if len(rates) == 0:
        warnings.warn("Sampling rate could not be inferred. This may cause problems.", UserWarning)
        return None

    elif len(rates) == 1:
        return rates[0]

    else:
        if np.sum(np.diff(rates)) > 0:
            warnings.warn("Audio files have different sampling rates. Files with higher sampling rate "\
                "will be downsampled to obtain consisten sampling rates as required to stitch the files "\
                "together.", UserWarning)

        rate = np.min(rates)
        return rate

def get_duration(path, offset=0, duration=None):
    ''' Get the durations of the specified audio file segments.

        Args:
            path: str or list(str)
                Path to input audio file(s)
            offset: float or list(float)
                Start of segment measured in seconds from the start of the file.
            duration: float or list(float)
                Segment length in seconds.

        Returns:
            res: list
                Durations in seconds
    '''
    path, offset, duration = _validate_wf_args(path, offset, duration)

    res = []    
    for i in range(len(path)):
        if duration[i] is None:
            assert path[i] is not None, "duration must be specified if path is None"
            d = librosa.get_duration(filename=path[i]) - offset[i] 
        else:
            d = duration[i]

        res.append(d)

    return res
    

def merge(waveforms, smooth=0.01):
    ''' Merge waveforms by stitching them together with the `append` method.

        All waveforms must have the same sampling rate. If this is not the case, 
        an AssertionError is thrown.

        Args:
            waveforms: list
                Waveform instances to be merged
            smooth: float
                Width in seconds of the smoothing region used for stitching together audio files.

        Returns:
            wf0: Instance of Waveform
                Merged waveforms
    '''
    if np.ndim(waveforms) == 0:
        waveforms = [waveforms]

    if len(waveforms) == 1:
        return waveforms[0]

    wf0 = waveforms[0].deepcopy()
    for wf in waveforms[1:]:
        n_smooth = int(smooth * wf.rate)
        wf0.append(wf, n_smooth=n_smooth)

    return wf0


def plot(waveforms, labels="", figsize=(5,4), title="", offset=0, duration=None):
    """ Plot one or several waveforms superimposed on one another.

        Note: The resulting figure can be shown (fig.show())
        or saved (fig.savefig(file_name))

        Args:
            waveforms: Waveform or list(Waveform)
                Waveforms to be plotted
            labels: str or list(str)
                Labels used to identify the waveforms. 
                Must have the same length as waveforms.
            figsize: tuple
                Figure size
            title: str
                Figure title.
            offset, duration: float
                Start time and length of the plotted segment in seconds. 
                If not specified, the full waveform will be plotted.
        
        Returns:
            fig: matplotlib.figure.Figure
                Figure object.
    """
    if isinstance(waveforms, Waveform): waveforms = [waveforms]
    if isinstance(labels, str): labels = [labels]

    assert len(waveforms) == len(labels), "waveforms and labels must have the same length"

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    colors = [f"C{i}" for i in range(6)]
    lstyles = ['-','--',':','-.']

    for i,wf in enumerate(waveforms):
        start = min(offset, wf.duration())
        end = wf.duration()
        if duration != None: end = min(end, start + duration)
        wfc = wf.crop(start=start, end=end, make_copy=True)
        col = colors[i%len(colors)]
        lsty = lstyles[i%len(lstyles)]
        x = np.linspace(start=start, stop=end, num=wfc.data.shape[0])
        y = wfc.get_data()
        ax.plot(x, y, label=labels[i], color=col, linestyle=lsty)
        ax.set_xlabel(wfc.time_ax.label)
        ax.set_ylabel('Amplitude')
        ax.set_title(title)

    if len(waveforms) > 1: ax.legend()

    return fig


class Waveform(BaseAudioTime):
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
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to this instance. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this instance

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
            transform_log: list(dict)
                List of transforms that have been applied to this instance
    """
    def __init__(self, data, time_res=None, filename='', offset=0, label=None, annot=None, transforms=None,
                    transform_log=None, **kwargs):

        assert time_res is not None or 'rate' in kwargs, "either time_res or rate must be specified"

        if time_res is None:
            self.rate = kwargs['rate']
        else:
            self.rate = 1. / time_res

        super().__init__(data=data, time_res=1./self.rate, filename=filename, offset=offset, label=label, annot=annot, 
                            transform_log=transform_log, **kwargs)

        self.allowed_transforms.update({'add_gaussian_noise': self.add_gaussian_noise, 
                                        'bandpass_filter': self.bandpass_filter})
        
        self.apply_transforms(transforms)

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'rate':self.rate, 'type':self.__class__.__name__})
        return attrs

    @classmethod
    def from_wav(cls, path, channel=0, rate=None, offset=0, duration=None, resample_method='scipy',
        id=None, normalize_wav=False, transforms=None, pad_mode="zero", smooth=0.01, **kwargs):
        """ Load audio data from one or several audio files.

            When loading from several audio files, the waveforms are stitched together in 
            the order in which they are provided using the `append` method. Note that only 
            the name and offset of the first file are stored in the `filename` and `offset` 
            attributes.  

            Args:
                path: str or list(str)
                    Path to input wave file(s).
                channel: int
                    In the case of stereo recordings, this argument is used 
                    to specify which channel to read from. Default is 0.
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                offset: float or list(float)
                    Position within the original audio file, in seconds 
                    measured from the start of the file. Defaults to 0 if not specified.
                duration: float or list(float)
                    Length in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                        
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                id: str
                    Unique identifier (optional). If provided, it is stored in the `filename` class attribute 
                    instead of the filename. A common use of the `id` argument is to specify a full or relative 
                    path to the file, including one or several directory levels.  
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1). Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to this instance. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                smooth: float
                    Width in seconds of the smoothing region used for stitching together audio files.
                pad_mode: str
                    Padding mode. Select between 'reflect' (default) and 'zero'.

            Returns:
                Instance of Waveform
                    Audio signal

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # read audio signal from wav file
                >>> a = Waveform.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # show signal
                >>> fig = a.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/audio_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/audio_grunt1.png
        """
        path, offset, duration = _validate_wf_args(path, offset, duration)

        if rate is None:
            rate = get_sampling_rate(path)

        waveforms = []
        for i in range(len(path)):
            wf = cls._from_single_file(path=path[i], channel=channel, rate=rate, offset=offset[i], 
                duration=duration[i], resample_method=resample_method, id=id, normalize_wav=normalize_wav, 
                transforms=transforms, pad_mode=pad_mode, **kwargs)

            waveforms.append(wf)

        wf = merge(waveforms, smooth=smooth)
        return wf

    @classmethod
    def _from_single_file(cls, path, channel=0, rate=None, offset=0, duration=None, resample_method='scipy',
        id=None, normalize_wav=False, transforms=None, pad_mode="reflect", **kwargs):
        """ Load audio data from a single audio file.

            If `duration` (and `offset`) are specified and `offset + duration` exceeds the 
            length of the audio file, the signal will be padded with its own reflection on 
            the right to achieve the desired duration. Similarly, if `offset < 0`, the signal 
            will be padded on the left. In both cases, a RuntimeWarning is issued.

            If `offset` exceeds the file duration, an empty waveform is returned and a 
            RuntimeWarning is issued.

            If `path` is None a waveform with length `int(rate * duration)` with purely zero 
            values will be returned. (Requires that both `rate` and `duration` are specified.)

            Args:
                path: str
                    Path to input audio file
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
                id: str
                    Unique identifier (optional). If provided, it is stored in the `filename` class attribute 
                    instead of the filename. A common use of the `id` argument is to specify a full or relative 
                    path to the file, including one or several directory levels.  
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1). Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to this instance. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                pad_mode: str
                    Padding mode. Select between 'reflect' (default) and 'zero'.

            Returns:
                Instance of Waveform
                    Audio signal
        """
        if path is None:
            assert duration is not None, "duration must be specified if path is None"
            assert rate is not None, "rate must be specified if path is None"
            return cls(rate=rate, data=np.zeros(int(rate*duration)), filename=id, offset=0)

        if transforms is None: transforms = []

        assert duration is None or duration >= 0, 'duration must be non-negative'

        # if 'id' is not specified, use the filename
        if id is None: id = os.path.basename(path)

        # original sampling rate in Hz
        rate_orig = librosa.get_samplerate(path)

        # file duration in seconds
        file_duration = librosa.get_duration(filename=path)

        # if the offset exceeds the file duration, return an empty array
        # and issue a warning
        if offset >= file_duration:
            data = np.array([], dtype=np.float64)
            if rate is None: rate = rate_orig
            warnings.warn("Offset exceeds file duration. Empty waveform returned", RuntimeWarning)
            return cls(rate=rate, data=data, filename=id, offset=offset)

        # if the duration is specified to 0, return an empty array
        # and issue a warning
        if duration is not None and duration == 0:
            data = np.array([], dtype=np.float64)
            if rate is None: rate = rate_orig
            warnings.warn("Duration is zero. Empty waveform returned", RuntimeWarning)
            return cls(rate=rate, data=data, filename=id, offset=offset)

        # if the offset is negative, pad with zeros on the left
        num_pad_left = 0
        if offset is not None and offset < 0:
            sr = rate_orig if rate is None else rate
            if duration is None:
                num_pad_left = int(-offset*sr)
            else:
                num_pad_left = int(min(-offset, duration)*sr)
                duration += offset
                duration = max(0, duration)

        num_pad_left = max(0, num_pad_left)

        if duration is not None and duration == 0:
            data = np.array([], dtype=np.float64)
            if rate is None: rate = rate_orig
            warnings.warn("Stop is before file start. Empty waveform returned", RuntimeWarning)
            return cls(rate=rate, data=data, filename=id, offset=offset)

        # determine start and stop times for reading the wav files
        start = aum.num_samples(max(0,offset), rate_orig)
        if duration is not None:
            stop = aum.num_samples(max(0,offset) + duration, rate_orig)
        else:
            stop = None

        # read data and sampling rate
        rate_orig, data = read_wave(file=path, channel=channel, start=start, stop=stop)

        # if necessary, re-sample
        if rate is not None and rate != rate_orig:
            data = librosa.core.resample(data, orig_sr=rate_orig, target_sr=rate, res_type=resample_method)
        else:
            rate = rate_orig

        # pad on left and/or right to achieve desired duration, if necessary
        if duration is not None:
            num_pad_right = max(0, int(duration * rate - data.shape[0]))
            if num_pad_right > 0 or num_pad_left > 0:
                if pad_mode.lower() == 'reflect':
                    data = aum.pad_reflect(data, pad_left=num_pad_left, pad_right=num_pad_right)
                    warnings.warn("Waveform padded with its own reflection to achieve desired length", RuntimeWarning)
                else:
                    data = aum.pad_zero(data, pad_left=num_pad_left, pad_right=num_pad_right)
                    warnings.warn("Waveform padded with zeros to achieve desired length", RuntimeWarning)

        if normalize_wav: 
            transforms.append({'name':'normalize','mean':0.0,'std':1.0})

        return cls(rate=rate, data=data, filename=id, offset=offset, transforms=transforms, **kwargs)

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
                Instance of Waveform
                    Audio signal sampling of Gaussian noise

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # create gaussian noise with sampling rate of 10 Hz, standard deviation of 2.0 and 1000 samples
                >>> a = Waveform.gaussian_noise(rate=10, sigma=2.0, samples=1000)
                >>> # show signal
                >>> fig = a.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/audio_noise.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/audio_noise.png
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
                Instance of Waveform
                    Audio signal sampling of the Morlet wavelet 

            Examples:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a Morlet wavelet with frequency of 3 Hz and 1-sigma width of envelope set to 2.0 seconds
                >>> wavelet1 = Waveform.morlet(rate=100., frequency=3., width=2.0)
                >>> # show signal
                >>> fig = wavelet1.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_standard.png")

                .. image:: ../../../ketos/tests/assets/tmp/morlet_standard.png

                >>> # create another wavelet, but with frequency increasing linearly with time
                >>> wavelet2 = Waveform.morlet(rate=100., frequency=3., width=2.0, dfdt=0.3)
                >>> # show signal
                >>> fig = wavelet2.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_dfdt.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/morlet_dfdt.png
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
                Instance of Waveform
                    Audio signal sampling of the cosine function 

            Examples:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a Cosine wave with frequency of 7 Hz
                >>> cos = Waveform.cosine(rate=1000., frequency=7.)
                >>> # show signal
                >>> fig = cos.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cosine_audio.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/cosine_audio.png
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

    def plot(self, show_annot=False, figsize=(5,4), label_in_title=True, append_title='', show_envelope=False):
        """ Plot the data with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the data.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                show_annot: bool
                    Display annotations
                figsize: tuple
                    Figure size
                label_in_title: bool
                    Include label (if available) in figure title
                append_title: str
                    Append this string to the title
                show_envelope: bool
                    Display envelope on top of signal
            
            Returns:
                fig: matplotlib.figure.Figure
                    Figure object.

            Example:            
                >>> from ketos.audio.waveform import Waveform
                >>> # create a morlet wavelet
                >>> a = Waveform.morlet(rate=100, frequency=5, width=1)
                >>> # plot the wave form
                >>> fig = a.plot()
                >>> plt.close(fig)

                .. image:: ../_static/morlet.png
        """
        fig, ax = super().plot(figsize, label_in_title, append_title)

        y = self.get_data()

        x = np.linspace(start=0, stop=self.duration(), num=self.data.shape[0])
        ax.plot(x, y)
        ax.set_ylabel('Amplitude')

        # superimpose envelope
        if show_envelope:
            z = np.abs(scipy.signal.hilbert(y))
            ax.plot(x, z, color='C1')

        # superimpose annotation boxes
        if show_annot: self._draw_annot_boxes(ax)

        #fig.tight_layout()
        return fig

    def _draw_annot_boxes(self, ax):
        """Draws annotations boxes on top of the spectrogram

            Args:
                ax: matplotlib.axes.Axes
                    Axes object
        """
        annots = self.get_annotations()
        if annots is None: return
        y1, y2 = ax.get_ylim()
        y1 *= 0.95
        y2 *= 0.95
        for idx,annot in annots.iterrows():
            x1 = annot['start']
            x2 = annot['end']
            box = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='C3',facecolor='none')
            ax.add_patch(box)
            ax.text(x1, y2, int(annot['label']), ha='left', va='bottom', color='C3')

    def append(self, signal, n_smooth=0):
        """ Append another audio signal to the present instance.

            The two audio signals must have the same samling rate.
            
            If n_smooth > 0, a smooth transition is made between the 
            two signals by padding the signals with their reflections 
            to form an overlap region of length n_smooth in which a 
            linear transition is made using the `_smoothclamp` function.
            This is done in manner that ensure that the duration of the 
            output signal is exactly the sum of the durations of the two 
            input signals.

            Note that the current implementation of the smoothing procedure is 
            quite slow, so it is advisable to use small value for n_smooth.

            Args:
                signal: Waveform
                    Audio signal to be appended.
                n_smooth: int
                    Width of the smoothing/overlap region (number of samples).

            Returns:
                None

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a morlet wavelet
                >>> mor = Waveform.morlet(rate=100, frequency=5, width=1)
                >>> # create a cosine wave
                >>> cos = Waveform.cosine(rate=100, frequency=3, duration=4)
                >>> # append the cosine wave to the morlet wavelet, using a overlap of 100 bins
                >>> mor.append(signal=cos, n_smooth=100)
                >>> # show the wave form
                >>> fig = mor.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_cosine.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/morlet_cosine.png
        """   
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        # if appending signal to itself, make a copy
        if signal is self:
            signal = self.deepcopy()

        # ensure that overlap region is shorter than either signal
        n_smooth = min(n_smooth, len(self.data) - 1)
        n_smooth = min(n_smooth, len(signal.data) - 1)

        # make sure n_smooth is even
        n_smooth += n_smooth % 2

        if n_smooth == 0:
            self.data = np.concatenate([self.data, signal.data], axis=0)

        else:# smoothly join
            # extend by own reflections
            a = np.concatenate([self.data, self.data[-2:int(-2-n_smooth/2):-1]])
            b = np.concatenate([signal.data[n_smooth//2:0:-1], signal.data])

            # split into separate and overlap 
            ao = a[-n_smooth:]
            bo = b[:n_smooth]
            a = a[:-n_smooth]
            b = b[n_smooth:]

            # compute values in overlap region
            c = np.empty(n_smooth)
            for i in range(n_smooth):
                w = _smoothclamp(i, 0, n_smooth-1)
                c[i] = (1.-w) * ao[i] + w * bo[i]

            self.data = np.concatenate([a,c,b], axis=0)
        
        # re-init time axis
        length = self.data.shape[0] / self.rate
        self.time_ax = LinearAxis(bins=self.data.shape[0], extent=(0., length), label='Time (s)') 

    def add_gaussian_noise(self, sigma):
        """ Add Gaussian noise to the signal

            Args:
                sigma: float
                    Standard deviation of the gaussian noise

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a morlet wavelet
                >>> morlet = Waveform.morlet(rate=100, frequency=2.5, width=1)
                >>> morlet_pure = morlet.deepcopy() # make a copy
                >>> # add some noise
                >>> morlet.add_gaussian_noise(sigma=0.3)
                >>> # show the wave form
                >>> fig = morlet_pure.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_wo_noise.png")
                >>> fig = morlet.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_w_noise.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/morlet_wo_noise.png

                .. image:: ../../../ketos/tests/assets/tmp/morlet_w_noise.png
        """
        noise = Waveform.gaussian_noise(rate=self.rate, sigma=sigma, samples=len(self.data))
        self.add(noise)
        self.transform_log.append({'name':'add_gaussian_noise', 'sigma':sigma})

    def bandpass_filter(self, freq_min=None, freq_max=None, N=3):
        """ Apply a lowpass, highpass, or bandpass filter to the signal.

            Uses SciPy's implementation of an Nth-order digital Butterworth filter.

            The critical frequencies, freq_min and freq_max, correspond to the points 
            at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).

            Args:
                freq_min: float
                    Lower limit of the frequency window in Hz.
                    (Also sometimes referred to as the highpass frequency).
                    If None, a lowpass filter is applied. 
                freq_max: float
                    Upper limit of the frequency window in Hz.
                    (Also sometimes referred to as the lowpass frequency)
                    If None, a highpass filter is applied. 
                N: int
                    The order of the filter. The default value is 3.

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a Cosine waves with frequencies of 7 and 14 Hz
                >>> cos = Waveform.cosine(rate=1000., frequency=7.)
                >>> cos14 = Waveform.cosine(rate=1000., frequency=14.)
                >>> cos.add(cos14)
                >>> # show combined signal
                >>> fig = cos.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cosine_double_audio.png")
                >>> plt.close(fig)
                >>> # apply 10 Hz highpass filter
                >>> cos.bandpass_filter(freq_max=10)
                >>> # show filtered signal
                >>> fig = cos.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cosine_double_hp_audio.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/cosine_double_audio.png

                .. image:: ../../../ketos/tests/assets/tmp/cosine_double_hp_audio.png
        """
        if freq_min is None and freq_max is None: return

        if freq_min is None: 
            Wn = freq_max
            btype = 'lowpass'
        elif freq_max is None: 
            Wn = freq_min            
            btype = 'highpass'
        else: 
            Wn = (freq_min, freq_max)            
            btype = 'bandpass'

        b,a = scipy.signal.butter(N=N, Wn=Wn, btype=btype, fs=self.rate)
        self.data = scipy.signal.filtfilt(b, a, self.data)
        self.transform_log.append({'name':'bandpass_filter', 'freq_min':freq_min, 'freq_max':freq_max, 'N':N})

    def add(self, signal, offset=0, scale=1):
        """ Add the amplitudes of the two audio signals.
        
            The audio signals must have the same sampling rates.
            The summed signal always has the same length as the present instance.
            If the audio signals have different lengths and/or a non-zero delay is selected, 
            only the overlap region will be affected by the operation.
            If the overlap region is empty, the original signal is unchanged.

            Args:
                signal: Waveform
                    Audio signal to be added
                offset: float
                    Shift the audio signal by this many seconds
                scale: float
                    Scaling factor applied to signal that is added

            Example:
                >>> from ketos.audio.waveform import Waveform
                >>> # create a cosine wave
                >>> cos = Waveform.cosine(rate=100, frequency=1., duration=4)
                >>> # create a morlet wavelet
                >>> mor = Waveform.morlet(rate=100, frequency=7., width=0.5)
                >>> mor.duration()
                3.0
                >>> # add the morlet wavelet on top of the cosine, with a shift of 1.5 sec and a scaling factor of 0.5
                >>> cos.add(signal=mor, offset=1.5, scale=0.5)
                >>> # show the wave form
                >>> fig = cos.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/morlet_cosine_added.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/morlet_cosine_added.png
        """
        assert self.rate == signal.rate, "Cannot add audio signals with different sampling rates."

        # if appending signal to itself, make a copy
        if signal is self:
            signal = self.deepcopy()

        # convert to bin numbers
        bin_offset = self.time_ax.bin(offset, truncate=True)
        bin_start = self.time_ax.bin(-offset, truncate=True)

        # crop signal that is being added
        length = self.data.shape[0] - bin_offset
        signal = signal.crop(start=-offset, length=length)

        # add the two signals
        b = bin_offset
        bins = signal.data.shape[0]
        self.data[b:b+bins] = self.data[b:b+bins] + scale * signal.data

    def resample(self, new_rate, resample_method='scipy'):
        """ Resample the acoustic signal with an arbitrary sampling rate.

        Args:
            new_rate: int
                New sampling rate in Hz
            resample_method: str
                Resampling method. Only relevant if `rate` is specified. Options are
                    * kaiser_best
                    * kaiser_fast
                    * scipy (default)
                    * polyphase
                    
                See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                for details on the individual methods.
        """
        if len(self.data) < 2:
            self.rate = new_rate

        else:                
            self.data = librosa.core.resample(self.get_data(), orig_sr=self.rate, target_sr=new_rate, res_type=resample_method)
            self.rate = new_rate

        self.time_ax = LinearAxis(bins=self.data.shape[0], extent=(0., self.data.shape[0] / self.rate), label='Time (s)') #new time axis


def _smoothclamp(x, mi, mx): 
        """ Smoothing function
        """    
        return (lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )
