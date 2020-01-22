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

""" Audio processing module within the ketos library

    This module provides utilities to perform various types of 
    operations on audio data, acting either in the time domain 
    (waveform) or in the frequency domain (spectrogram), or 
    both.
"""
import numpy as np
import librosa
from sys import getsizeof
from psutil import virtual_memory


def pad_reflect(x, pad_left=0, pad_right=0):
    """ Pad array with its own reflection along its first axis (0).

        Args: 
            x: numpy.array
                The data to be padded.
            pad_left: int
                Amount of padding on the left
            pad_right: int
                Amount of padding on the right

        Returns:
            x_padded: numpy.array
                Padded array
    """
    if pad_left == 0 and pad_right == 0:
        x_padded = x

    else:
        x_padded = x.copy()
        if pad_left > 0:
            x_pad = np.flip(x[:pad_left], axis=0)
            x_padded = np.concatenate((x_pad, x_padded))

        if pad_right > 0:
            x_pad = np.flip(x[-pad_right:], axis=0)
            x_padded = np.concatenate((x_padded, x_pad))

    return x_padded

def segment_args(rate, duration, offset, window, step):
    """ Computes input arguments for :func:`audio_processing.audio_processing.make_segment` 
        to produce a centered spectrogram with properties as close as possible to the 
        those specified.

        Args:
            rate: float
                Sampling rate in Hz
            duration: float
                Duration in seconds
            offset: float
                Offset in seconds
            window: float
                Window size in seconds
            step: float
                Window size in seconds

        Returns:
            : dict
            Dictionary with following keys and values:
                * num_segs: Number of steps (int)
                * offset_len: Offset in number of samples (int)
                * win_len: Window size in number of samples (int)
                * step_len: Step size in number of samples (int)
    """
    win_len = num_samples(window, rate=rate, even=True) 
    step_len = num_samples(step, rate=rate, even=True)
    num_segs = num_samples(duration, rate=rate/step_len)
    offset_len = num_samples(offset, rate=rate) - int(win_len/2) + int(step_len/2)
    return {'num_segs':num_segs, 'offset_len':offset_len, 'win_len':win_len, 'step_len':step_len}

def segment(x, num_segs, offset_len, win_len, step_len, warnings=True):
    """ Divide an array into segments of equal length along its first 
        axis (0), each segment being shifted by a fixed amount with respetive to the 
        previous segment.

        If offset_len is negative the input array will be padded with its own 
        reflection on the left. 
        If the combined length of the segments exceeds the length of the input 
        array (minus any positive offset), the array will also be padded with its 
        own reflection on the right.

        Args: 
            x: numpy.array
                The data to be segmented
            num_segs: int
                Number of segments
            offset_len: int
                Position of the first frame
            win_len: int
                Window length.
            step_len: float
                Step size.
            warnings: bool
                Print warning if the size of the array exceeds 10% of the 
                available memory.

        Returns:
            segs: numpy.array
                Segmented data.

        Example:
            >>> from ketos.audio_processing.audio_processing import make_frames
            >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> f = make_frames(x=x, winlen=4, winstep=2)    
            >>> print(f.shape)
            (4, 4)
            >>> print(f)
            [[ 1  2  3  4]
             [ 3  4  5  6]
             [ 5  6  7  8]
             [ 7  8  9 10]]
    """
    mem = virtual_memory() #memory available
    siz = getsizeof(x) * win_len / step_len #estimated size of output array
    if siz > 0.1 * mem.total: #print warning, if output array is very large
        print("Warning: size of output frames exceeds 10% of memory")
        print("Consider reducing the array size and/or increasing the step size and/or reducing the window size")

    # pad, if necessary
    pad_left = max(0, -offset_len)
    pad_right = max(0, max(0, offset_len) + num_segs * step_len + win_len - x.shape[0])    
    x_pad = pad_reflect(x, pad_left, pad_right)

    # tile    
    indices = np.tile(np.arange(0, win_len), (num_segs, 1)) + np.tile(np.arange(0, num_segs * step_len, step_len), (win_len, 1)).T
    segs = x_pad[indices.astype(np.int32, copy=False)]

    return segs

def num_samples(time, rate, even=False):
    """ Convert time interval to number of samples. 
        
        If the time corresponds to a non-integer number of samples, 
        round to the nearest larger integer value.

        Args:
            time: float
                Timer interval in seconds
            rate: float
                Sampling rate in Hz
            even: bool
                Convert to nearest larger even integer.

        Returns:
            n: int
                Number of samples
    """
    if even: 
        n = int(2 * np.ceil(0.5 * time * rate))
    else:
        n = int(np.ceil(time * rate))
    
    return n

def stft(x, rate, window=None, step=None, seg_args=None, window_func='hamming'):
    """ Compute Short Time Fourier Transform (STFT).

        Uses :func:`audio_processing.audio_processing.segment_args` to convert 
        the window size and step size into an even integer number of samples.
        
        The number of points used for the Fourier Transform is equal to the 
        number of samples in the window.
    
        Args:
            x: numpy.array
                Audio signal 
            rate: float
                Sampling rate in Hz
            window: float
                Window length in seconds
            step: float
                Step size in seconds 
            seg_args: dict
                Input arguments for :func:`audio_processing.audio_processing.segment_args`. 
                Optional. If specified, the arguments `window` and `step` are ignored.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning

        Returns:
            img: numpy.array
                Short Time Fourier Transform of the input signal.
            freq_max: float
                Maximum frequency in Hz
            num_fft: int
                Number of points used for the Fourier Transform.
            seg_args: dict
                Input arguments used for evaluating :func:`audio_processing.audio_processing.segment_args`. 
    """
    if seg_args is None:
        assert window and step, "if seg_args is not specified, window and step must both be specified."

        seg_args = segment_args(rate=rate, duration=len(x)*rate,\
            offset=0, window=window, step=step) #compute input arguments for segment method

    segs = segment(seg_args) #divide audio signal into segments

    if window_func:
        segs *= eval("np.{0}".format(window_func))(segs.shape[1]) #apply Window function

    fft = np.fft.rfft(segs) #Compute fast fourier transform
    img = to_decibel(np.abs(fft)) #Compute magnitude on dB scale
    num_fft = frames.shape[1] #Number of points used for the Fourier Transform        
    freq_max = audio_signal.rate / 2. #Maximum frequency
    return img, freq_max, num_fft, seg_args

def cqt(self, x, rate, step, bins_per_oct, freq_min, freq_max=None, window_func='hamming'):
    """ Compute the CQT spectrogram of an audio signal.

        Uses the librosa implementation, 
        
            https://librosa.github.io/librosa/generated/librosa.core.cqt.html

        To compute the CQT spectrogram, the user must specify the step size, the minimum and maximum 
        frequencies, :math:`f_{min}` and :math:`f_{max}`, and the number of bins per octave, :math:`m`.
        While :math:`f_{min}` and :math:`m` are fixed to the input values, the step size and 
        :math:`f_{max}` are adjusted as detailed below, attempting to match the input values 
        as closely as possible.

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
        For the above example, we find :math:`f_{max} = 20480` Hz, i.e., somewhat larger than the 
        requested maximum value.

        Note that if :math:`f_{max}` exceeds the Nyquist frequency, :math:`f_{nyquist} = 0.5 \cdot s`, 
        where :math:`s` is the sampling rate, the number of octaves, :math:`k`, is reduced to ensure 
        that :math:`f_{max} \leq f_{nyquist}`. 

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
            x: numpy.array
                Audio signal 
            rate: float
                Sampling rate in Hz
            step: float
                Step size in seconds 
            bins_per_oct: int
                Number of bins per octave
            freq_min: float
                Minimum frequency in Hz
            freq_max: float
                Maximum frequency in Hz. If None, it is set equal to half the sampling rate.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning

        Returns:
            img: numpy.array
                Resulting CQT spectrogram image.
            step: float
                Adjusted step size in seconds.
    """
    f_nyquist = 0.5 * rate
    k_nyquist = int(np.floor(np.log2(f_nyquist / freq_min)))

    if freq_max is None:
        k = k_nyquist
    else:    
        k = int(np.ceil(np.log2(freq_max/freq_min)))
        k = min(k, k_nyquist)

    h0 = int(2**k)
    b = bins_per_oct
    bins = k * b

    h = rate * step
    r = int(np.ceil(h / h0))
    h = int(r * h0)

    img = librosa.core.cqt(y=x, sr=rate, hop_length=h, fmin=freq_min, n_bins=bins, bins_per_octave=b)
    img = to_decibel(np.abs(img))
    img = np.swapaxes(img, 0, 1)
    
    step = h / rate

    return img, step

def to_decibel(x):
    """ Convert to decibels

    Args:
        x : numpy array
            Input array
    
    Returns:
        y : numpy array
            Converted array

    Example:

        >>> from ketos.audio_processing.audio_processing import to_decibel 
        >>> import numpy as np
        >>> img = np.array([[10., 20.],[30., 40.]])
        >>> img_db = to_decibel(img)
        >>> img_db = np.around(img_db, decimals=2) # only keep up to two decimals
        >>> print(img_db)
        [[20.0 26.02]
         [29.54 32.04]]
    """
    y = 20 * np.ma.log10(x)

    return y

def from_decibel(y):
    """ Convert from decibels

    Args:
        y : numpy array
            Input array
    
    Returns:
        x : numpy array
            Converted array

    Example:

        >>> from ketos.audio_processing.audio_processing import from_decibel 
        >>> import numpy as np
        >>> img = np.array([[10., 20.],[30., 40.]])
        >>> img_db = from_decibel(img)
        >>> img_db = np.around(img_db, decimals=2) # only keep up to two decimals
        >>> print(img_db)
        [[  3.16  10.  ]
         [ 31.62 100.  ]]
    """

    x = np.power(10., y/20.)
    return x

def spec2audio(image, phase_angle, num_fft, step_len, num_iters, window_func):
    """ Estimate audio signal from magnitude spectrogram.

        Implements the algorithm described in 

            D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,” IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.            

        Follows closely the implentation of https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py

        Args:
            image: 2d numpy array
                Magnitude spectrogram
            phase_angle: 
                Initial condition for phase.
            num_fft: int
                Number of points used for the Fast-Fourier Transform. Same as window size.
            step_len: int
                Step size.
            num_iters: 
                Number of iterations to perform.
            window_func: string, tuple, number, function, np.ndarray [shape=(num_fft,)]
                - a window specification (string, tuple, or number); see `scipy.signal.get_window`
                - a window function, such as `scipy.signal.hamming`
                - a user-specified window vector of length `num_fft`

        Returns:
            audio: 1d numpy array
                Audio signal

        Example:
            >>> # create a simple sinusoidal audio signal with frequency of 10 Hz
            >>> import numpy as np
            >>> x = np.arange(1000)
            >>> sig = 32600 * np.sin(2 * np.pi * 10 * x / 1000) 
            >>> # compute the magnitude spectrogram with window size of 200, step size of 40,
            >>> # and using a Hamming window
            >>> from ketos.audio_processing.audio_processing import make_frames
            >>> frames = make_frames(sig, 200, 40) 
            >>> frames *= np.hamming(frames.shape[1])
            >>> mag = np.abs(np.fft.rfft(frames))
            >>> # estimate the original signal            
            >>> from ketos.audio_processing.audio_processing import estimate_audio_signal
            >>> sig_est = estimate_audio_signal(image=mag, phase_angle=0, num_fft=200, hop=40, num_iters=25, window=np.hamming(frames.shape[1]))
            >>> # plot the original and the estimated signal
            >>> import matplotlib.pyplot as plt
            >>> plt.clf()
            >>> _ = plt.plot(sig)
            >>> plt.savefig("ketos/tests/assets/tmp/sig_orig.png")
            >>> _ = plt.plot(sig_est)
            >>> plt.savefig("ketos/tests/assets/tmp/sig_est.png")

            .. image:: ../../../../ketos/tests/assets/tmp/sig_est.png
    """
    # swap axis to conform with librosa 
    image = np.swapaxes(image, 0, 1)

    # settings for FFT and inverse FFT    
    fft_config = dict(n_fft=num_fft, win_length=num_fft, hop_length=step_len, center=False, window=window_func)
    ifft_config = dict(win_length=num_fft, hop_length=step_len, center=False, window=window_func)

    # initial spectrogram for iterative algorithm
    complex_specgram = inv_magphase(image, phase_angle)

    # Griffin-Lim iterative algorithm
    for i in range(num_iters):
        audio = librosa.istft(complex_specgram, **ifft_config)
        if i != num_iters - 1:
            complex_specgram = librosa.stft(audio, **fft_config)
            _, phase = librosa.magphase(complex_specgram)
            angle = np.angle(phase)
            complex_specgram = inv_magphase(image, angle)

    return audio
