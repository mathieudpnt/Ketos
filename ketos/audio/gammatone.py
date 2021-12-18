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

""" 'audio.gammatone' module within the ketos library.

    This module provides utilities to process audio data 
    using a auditory filterbank based on the gammatone function.

    Contents:
        GammatoneFilterBank class:
        AuralFeatures class
"""
import sys
import os
import copy
import librosa
import warnings
import numpy as np
from version_parser.version import Version
import scipy
import matplotlib.pyplot as plt
from ketos.audio.waveform import Waveform
from ketos.audio.annotation import AnnotationHandler
from ketos.audio.base_audio import BaseAudio, BaseAudioTime, segment_data


gammatone_filter_coeff = dict()


def compute_center_freqs(num_chan, sampl_rate, freq_min):
    """ Compute the center frequencies of the filter bank

        Args:
            num_chan: int
                Number of channels in the filter bank
            sampl_rate: float
                Sampling rate in Hz
            freq_min: float
                Minimum frequency in Hz

        Returns:
            freqs: float
                Center frequencies in Hz
    """
    f0 = 228.8
    i = np.linspace(1, num_chan, num_chan)
    freqs = np.exp(i / num_chan * (np.log(freq_min + f0) - np.log(sampl_rate / 2. + f0))) * (sampl_rate / 2. + f0) - f0
    freqs = np.flip(freqs)
    return freqs

def filter_signal(signal, sampl_rate, freqs):
    """ Pass the signal through the gammatone filters

        Args:
            signal: numpy.array
                Audio signal
            sampl_rate: float
                Sampling rate in Hz
            freqs: numpy.array
                Center frequencies of the filter bank

        Returns:
            x: numpy.array
                The filtered signals stacked vertically into 2D array
    """
    x = []
    for freq in freqs:
        (b, a) = get_filter_coeffs(sampl_rate, freq)
        x.append(scipy.signal.filtfilt(b, a, signal))

    x = np.array(x)
    x = np.swapaxes(x, 0, 1)
    return x

def apply_weight_func(x, freqs):
    """ Apply C weighting function.
    
        This weighting function represents the approximate frequency sensitivity 
        of the human auditory system.

        Args:
            x: numpy.array
                The filtered signals
            freqs: numpy.array
                Center frequencies of the filter bank

        Returns:
            x: numpy.array
                The C-weighted filtered signals
    """
    C = 1.007 * 12200**2 * freqs**2 / (freqs**2 + 20.6**2) / (freqs**2 + 12200**2)
    return x * C[np.newaxis,:]

def get_filter_coeffs(sampl_rate, freq):
    """ Get the gammatone filter coefficients.
    
        Args:
            sampl_rate: float
                Sampling rate in Hz
            freq: float
                Center frequency in Hz

        Returns:
            : tuple
                Coefficients of the gammatone filter
    """        
    key = (sampl_rate, freq) 
    if key not in gammatone_filter_coeff.keys():
        gammatone_filter_coeff[key] = compute_filter_coeffs(sampl_rate, freq)

    return gammatone_filter_coeff[key]        
        
def compute_filter_coeffs(sampl_rate, freq):
    """ Compute the gammatone filter coefficients.
    
        Args:
            sampl_rate: float
                Sampling rate in Hz
            freq: float
                Center frequency in Hz

        Returns:
            (b, a): tuple
                Coefficients of the gammatone filter
    """        
    if Version(scipy.__version__) < Version("1.6.0"):
        print('The `compute_filter_coeffs` method in the `gammatone` module requires Scipy>=1.6.0')
        print(f'The present environment only has Scipy=={Version(scipy.__version__)}')
        print('Note that Scipy>=1.6.0 requires Python>=3.7.0')
        exit(1)

    b, a = scipy.signal.gammatone(freq=freq, ftype='iir', fs=sampl_rate)
    return (b, a)


class GammatoneFilterBank(BaseAudioTime):
    """ Gammatone filter bank.

        The filtered signals are stored in a 2D numpy array, where the first axis 
        (0) is the time dimension and the second axis (1) is the frequency dimension.

        Args:
            data: 2d numpy array
                Filtered data 
            rate: float
                Sampling rate in Hz
            freqs: array-like
                Center frequencies of the filter bank in Hz
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            weight_func: bool
                Apply C weighting function. Default is True.
            
        Attributes:
            data: 2d numpy array
                Filtered data 
            rate: float
                Sampling rate in Hz
            freqs: array-like
                Center frequencies of the filter bank in Hz
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            weight_func: bool
                Apply C weighting function.
"""
    def __init__(self, data, rate, freqs, filename=None, offset=0, label=None, annot=None, weight_func=True, **kwargs):
        self.rate = rate
        self.freqs = freqs
        self.weight_func = weight_func
        super().__init__(data=data, time_res=1./rate, filename=filename, offset=offset, label=label, annot=annot)

    @classmethod
    def empty(cls):
        """ Creates an empty GammatoneFilterBank object
        """
        return cls(data=np.empty(shape=(0,0), dtype=np.float64), rate=1, freqs=[])

    @classmethod
    def from_waveform(cls, audio, num_chan=20, freq_min=1, weight_func=True):
        """ Create a Gammatone Filter Bank from an instance of :class:`audio_signal.Waveform`.
        
            Args:
                audio: Waveform
                    Audio signal 
                num_chan: int
                    Number of channels in the filter bank
                freq_min: float
                    Minimum frequency of the filter bank in Hz
                weight_func: bool
                    Apply C weighting function. Default is True.

            Returns:
                gfb: GammatoneFilterBank
                    Gammatone filter bank
        """
        center_freqs = compute_center_freqs(num_chan=num_chan, sampl_rate=audio.rate, freq_min=freq_min)

        filtered_signals = filter_signal(signal=audio.data, sampl_rate=audio.rate, freqs=center_freqs)

        if weight_func:
            filtered_signals = apply_weight_func(x=filtered_signals, freqs=center_freqs)

        gfb = cls(data=filtered_signals, rate=audio.rate, freqs=center_freqs,
            filename=audio.filename, offset=audio.offset, label=audio.label, 
            annot=audio.annot, weight_func=weight_func)

        return gfb

    @classmethod
    def from_wav(cls, path, num_chan=20, freq_min=1, channel=0, rate=None, offset=0, duration=None,
            resample_method='scipy', id=None, normalize_wav=False, weight_func=True, **kwargs):
        """ Create a Gammatone Filter Bank directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments offset and duration may be subject 
            to slight adjustments to ensure that the selected portion corresponds to an integer 
            number of samples.

            Args:
                path: str
                    Path to wav file
                num_chan: int
                    Number of channels in the filter bank
                freq_min: float
                    Minimum frequency of the filter bank in Hz
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                offset: float
                    Start time of selection in seconds, relative the start of the wav file.
                duration: float
                    Length of selection in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase

                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                weight_func: bool
                    Apply C weighting function. Default is True.

            Returns:
                : GammatoneFilterBank
                    Gammatone filter bank

            Example:
                >>> # load gammatone filter bank from wav file
                >>> from ketos.audio.gammatone import GammatoneFilterBank
                >>> gfb = GammatoneFilterBank.from_wav('ketos/tests/assets/grunt1.wav', num_chan=20, freq_min=10, rate=1000)
                >>> # print the center frequencies rounded to 1 decimal
                >>> print(np.round(gfb.freqs,1))
                [ 10.   23.7  38.2  53.5  69.7  86.8 104.9 124.1 144.3 165.7 188.4 212.3
                 237.6 264.4 292.7 322.6 354.2 387.7 423.1 460.5]
                >>> # display the 4th filter bank signal
                >>> fig = gfb.plot(filter_id=3)
                >>> fig.savefig("ketos/tests/assets/tmp/gfb3_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/gfb3_grunt1.png
        """
        # load audio
        audio = Waveform.from_wav(path=path, channel=channel, rate=rate, offset=offset, duration=duration, 
            resample_method=resample_method, id=id, normalize_wav=normalize_wav, **kwargs)

        if len(audio.get_data()) == 0:
            warnings.warn("Empty GammatoneFilterBank returned", RuntimeWarning)
            return cls.empty()

        # compute gammatone filter bank
        return cls.from_waveform(audio=audio, num_chan=num_chan, freq_min=freq_min, weight_func=weight_func)

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'rate':self.rate, 'freqs':self.freqs, 
            'weight_func':self.weight_func, 'type':self.__class__.__name__})
        return attrs

    def plot(self, filter_id, show_annot=False, figsize=(5,4), label_in_title=True, show_envelope=False):
        """ Plot the filtered signal with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the signal.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                filter_id: int
                    Filter to be plotted.
                show_annot: bool
                    Display annotations
                figsize: tuple
                    Figure size
                label_in_title: bool
                    Include label (if available) in figure title
                show_envelope: bool
                    Display envelope on top of signal
            
            Returns:
                : matplotlib.figure.Figure
                    A figure object.

            Example:
                >>> from ketos.audio.gammatone import GammatoneFilterBank
                >>> # load gammatone filter bank
                >>> gfb = GammatoneFilterBank.from_wav('ketos/tests/assets/grunt1.wav', num_chan=20, freq_min=10, rate=1000)
                >>> # add an annotation
                >>> gfb.annotate(start=1.2, end=1.6, freq_min=70, freq_max=600, label=1)
                >>> # show the 4th filter bank with annotation box
                >>> fig = gfb.plot(filter_id=3, show_annot=True)
                >>> fig.savefig("ketos/tests/assets/tmp/gfb3_w_annot_box.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/gfb3_w_annot_box.png
        """
        x = self.get_data()[:,filter_id] # select the filtered signal 

        wf = Waveform(data=x, rate=self.rate, filename=self.get_filename(), 
            offset=self.get_offset(), label=self.get_label(), annot=self.get_annotations())

        return wf.plot(show_annot=show_annot, figsize=figsize, label_in_title=label_in_title,
            append_title=f', {self.freqs[filter_id]:.1f} Hz', show_envelope=show_envelope)



class AuralFeatures(BaseAudio):
    """ Aural features computed with the aural-features package (https://pypi.org/project/aural-features/).

        Args:
            data: 1d numpy array
                Feature values
            filename: str
                Name of the source audio file, if available.   
            offset: float
                Position in seconds of the left edge of the audio segment within the source 
                audio file, if available.
            label: int
                Label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
"""
    def __init__(self, data, filename=None, offset=0, label=None, annot=None, waveform_transform_log=None, **kwargs):
        super().__init__(data=data, filename=filename, offset=offset, label=label, annot=annot)

        if waveform_transform_log is None: waveform_transform_log = []
        self.waveform_transform_log = waveform_transform_log

    @classmethod
    def from_waveform(cls, audio, filter_pad_samples=64, global_km_window_seconds=0.25, 
        local_km_window_seconds=0.008, filter_n=100, filter_min_hz=50):
        """ Compute aural features from an instance of :class:`audio_signal.Waveform`.
        
            Args:
                audio: Waveform
                    Audio signal 
                filter_pad_samples: int
                    Number of samples used for padding
                global_km_window_seconds: float
                    Length of global KM window in seconds
                local_km_window_seconds: float
                    Length of local KM window in seconds
                filter_n: int
                    Number of filters
                filter_min_hz: float
                    Min filter frequency in Hz 

            Returns:
                : AuralFeatures
                    Aural features
        """
        if 'aural' not in sys.modules:
            try:
                import aural.meridian as au
            except ImportError:
                print('aural-features package not found.')
                print('aural-features is required by the AuralFeatures class.')
                print('install with `pip install aural-features`.')
                print('note that aural-features requires Scipy>=1.6 and Python>=3.7')
                raise ImportError

        conf = au.Config() # For defaults, should be safe to start

        conf.filter_pad_samples = filter_pad_samples
        conf.global_km_window_seconds = global_km_window_seconds
        conf.local_km_window_seconds = local_km_window_seconds
        conf.filter_n = filter_n
        conf.filter_min_hz = filter_min_hz
        
        try:
            duration, features = au.extract(audio.get_data(), audio.rate, conf)

            values = [duration]
            for x in features:
                for name, value in x._asdict().items():
                    values.append(value)

            values = np.array(values)
            values = np.nan_to_num(values, nan=0.0) #replace NaN's with zeros

        except au.IsolationFailed:
            values = np.zeros(46)

        return cls(data=values, filename=audio.filename, offset=audio.offset, label=audio.label, annot=audio.annot)

    @classmethod
    def from_wav(cls, path, filter_pad_samples=64, global_km_window_seconds=0.25, 
            local_km_window_seconds=0.008, filter_n=100, filter_min_hz=50, 
            channel=0, rate=None, offset=0, duration=None,
            resample_method='scipy', id=None, normalize_wav=False, 
            waveform_transforms=None, **kwargs):
        """ Compute aural features directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments offset and duration may be subject 
            to slight adjustments to ensure that the selected portion corresponds to an integer 
            number of samples.

            Args:
                path: str
                    Path to wav file
                filter_min: float
                    Min filter frequency in Hz 
                local_km_window: float
                    Length of local KM window in seconds
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                offset: float
                    Start time of selection in seconds, relative the start of the wav file.
                duration: float
                    Length of selection in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase

                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                waveform_transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the waveform before generating 
                    the spectrogram. For example,
                    {"name":"add_gaussian_noise", "sigma":0.5}

            Returns:
                : AuralFeatures
                    Aural features
        """
        # load audio
        audio = Waveform.from_wav(path=path, channel=channel, rate=rate, offset=offset, duration=duration, 
            resample_method=resample_method, id=id, normalize_wav=normalize_wav, 
            transforms=waveform_transforms, **kwargs)

        if len(audio.get_data()) == 0:
            warnings.warn("Empty AuralFeatures returned", RuntimeWarning)
            return cls.empty()

        # compute gammatone filter bank
        return cls.from_waveform(audio=audio, filter_pad_samples=filter_pad_samples, global_km_window_seconds=global_km_window_seconds, 
            local_km_window_seconds=local_km_window_seconds, filter_n=filter_n, filter_min_hz=filter_min_hz)

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'type':self.__class__.__name__})
        return attrs
