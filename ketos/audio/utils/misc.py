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

""" 'audio.utils.misc' module within the ketos library

    This module provides utilities to perform various types of 
    operations on audio data, acting either in the time domain 
    (waveform) or in the frequency domain (spectrogram), or 
    both.
"""
import numpy as np
import librosa
from sys import getsizeof
from psutil import virtual_memory
from ketos.utils import complex_value

def pad_reflect(x, pad_left=0, pad_right=0):
    """ Pad array with its own (inverted) reflection along 
        the first axis (0).

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

        Example:
            >>> from ketos.audio.utils.misc import pad_reflect
            >>> arr = np.arange(9) #create a simple array
            >>> print(arr)
            [0 1 2 3 4 5 6 7 8]
            >>> arr = pad_reflect(arr, pad_right=3) #pad on the right
            >>> print(arr)
            [ 0  1  2  3  4  5  6  7  8  9 10 11]
    """
    if pad_left == 0 and pad_right == 0:
        x_padded = x

    else:
        pad_left_residual  = 0
        pad_right_residual = 0
        x_padded = x.copy()
        if pad_left > 0:
            x_pad = 2*x[0] - np.flip(x[1:pad_left+1], axis=0)
            pad_left_residual = max(0, pad_left - x_pad.shape[0])
            x_padded = np.concatenate((x_pad, x_padded))

        if pad_right > 0:
            x_pad = 2*x[-1] - np.flip(x[-pad_right-1:-1], axis=0)
            pad_right_residual = max(0, pad_right - x_pad.shape[0])
            x_padded = np.concatenate((x_padded, x_pad))

        if pad_left_residual + pad_right_residual > 0:
            x_padded = pad_zero(x_padded, pad_left_residual, pad_right_residual)

    return x_padded

def pad_zero(x, pad_left=0, pad_right=0):
    """ Pad array with zeros along the first axis (0).

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

        Example:
            >>> from ketos.audio.utils.misc import pad_zero
            >>> arr = np.arange(9) #create a simple array
            >>> print(arr)
            [0 1 2 3 4 5 6 7 8]
            >>> arr = pad_zero(arr, pad_right=3) #pad on the right
            >>> print(arr)
            [0 1 2 3 4 5 6 7 8 0 0 0]
    """
    if pad_left == 0 and pad_right == 0:
        x_padded = x

    else:
        x_padded = x.copy()
        pad_shape = x.shape
        if pad_left > 0:
            pad_shape = tuple([pad_left] + list(x.shape)[1:])
            x_pad = np.zeros(pad_shape, dtype=x.dtype)
            x_padded = np.concatenate((x_pad, x_padded))

        if pad_right > 0:
            pad_shape = tuple([pad_right] + list(x.shape)[1:])
            x_pad = np.zeros(pad_shape, dtype=x.dtype)
            x_padded = np.concatenate((x_padded, x_pad))

    return x_padded

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

        Example:
            >>> from ketos.audio.utils.misc import num_samples
            >>> print(num_samples(rate=1000., time=0.0))
            0
            >>> print(num_samples(rate=1000., time=2.0))
            2000
            >>> print(num_samples(rate=1000., time=2.001))
            2001
            >>> print(num_samples(rate=1000., time=2.001, even=True))
            2002
    """
    if even: 
        n = int(2 * np.ceil(0.5 * time * rate))
    else:
        n = int(np.ceil(time * rate))
    
    return n

def segment_args(rate, duration, offset, window, step):
    """ Computes input arguments for :func:`audio.utils.misc.make_segment` 
        to produce a centered spectrogram with properties as close as possible to 
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
                    * win_len: Window size in number of samples (int)
                    * step_len: Step size in number of samples (int)
                    * num_segs: Number of steps (int)
                    * offset_len: Offset in number of samples (int)

        Example: 
            >>> from ketos.audio.utils.misc import segment_args
            >>> args = segment_args(rate=1000., duration=3., offset=0., window=0.1, step=0.02)
            >>> for key,value in sorted(args.items()):
            ...     print(key,':',value)
            num_segs : 150
            offset_len : -40
            step_len : 20
            win_len : 100
    """
    win_len = num_samples(window, rate=rate, even=True) 
    step_len = num_samples(step, rate=rate, even=True)
    num_segs = num_samples(duration, rate=rate/step_len)
    offset_len = num_samples(offset, rate=rate) - int(win_len/2) + int(step_len/2)
    return {'win_len':win_len, 'step_len':step_len, 'num_segs':num_segs, 'offset_len':offset_len}

def segment(x, win_len, step_len, num_segs=None, offset_len=0, pad_mode='reflect', mem_warning=True):
    """ Divide an array into segments of equal length along its first 
        axis (0), each segment being shifted by a fixed amount with respetive to the 
        previous segment.

        If offset_len is negative the input array will be padded with its own 
        inverted reflection on the left. 

        If the combined length of the segments exceeds the length of the input 
        array (minus any positive offset), the array will be padded with its 
        own inverted reflection on the right.

        Args: 
            x: numpy.array
                The data to be segmented
            win_len: int
                Window length in no. of samples
            step_len: float
                Step size in no. of samples
            num_segs: int
                Number of segments. Optional.
            offset_len: int
                Position of the first frame in no. of samples. Defaults to 0, if not specified.
            pad_mode: str
                Padding mode. Select between 'reflect' (default) and 'zero'.
            mem_warning: bool
                Print warning if the size of the array exceeds 10% of the 
                available memory.

        Returns:
            segs: numpy.array
                Segmented data, has shape (num_segs, win_len, x.shape[1:])

        Example:
            >>> from ketos.audio.utils.misc import segment
            >>> x = np.arange(10)
            >>> print(x)
            [0 1 2 3 4 5 6 7 8 9]
            >>> y = segment(x, win_len=4, step_len=2, num_segs=3, offset_len=0)    
            >>> print(y)
            [[0 1 2 3]
             [2 3 4 5]
             [4 5 6 7]]
            >>> y = segment(x, win_len=4, step_len=2, num_segs=3, offset_len=-3)    
            >>> print(y)
            [[-3 -2 -1  0]
             [-1  0  1  2]
             [ 1  2  3  4]]
    """
    mem = virtual_memory() #memory available
    siz = getsizeof(x) * win_len / step_len #estimated size of output array
    if siz > 0.1 * mem.total: #print warning, if output array is very large
        print("Warning: size of output frames exceeds 10% of memory")
        print("Consider reducing the array size and/or increasing the step size and/or reducing the window size")

    # if not specified, compute number of segments so entire array is used
    if num_segs is None:
        num_segs = int(np.ceil((len(x) - offset_len - win_len) / step_len)) + 1

    # pad, if necessary
    pad_left = max(0, -offset_len)
    pad_right = max(0, max(0, offset_len) + num_segs * step_len + win_len - x.shape[0])    
    if pad_mode == 'reflect':
        x_pad = pad_reflect(x, pad_left, pad_right)
    else:
        x_pad = pad_zero(x, pad_left, pad_right)

    # tile    
    indices = np.tile(np.arange(0, win_len), (num_segs, 1)) + np.tile(np.arange(0, num_segs * step_len, step_len), (win_len, 1)).T
    segs = x_pad[indices.astype(np.int32, copy=False)]

    return segs

def stft(x, rate, window=None, step=None, seg_args=None, window_func='hamming', decibel=True, compute_phase=False):
    """ Compute Short Time Fourier Transform (STFT).

        Uses :func:`audio.utils.misc.segment_args` to convert 
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
                Input arguments for :func:`audio.utils.misc.segment_args`. 
                Optional. If specified, the arguments  `window` and `step` are ignored.
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning
            decibel: bool
                Convert to dB scale
            compute_phase: bool
                Compute complex phase angle. Default it False

        Returns:
            img: numpy.array
                Short Time Fourier Transform of the input signal.
            freq_max: float
                Maximum frequency in Hz
            num_fft: int
                Number of points used for the Fourier Transform.
            seg_args: dict
                Input arguments used for evaluating :func:`audio.utils.misc.segment_args`. 
            cpx_angle: numpy.array
                Complex phase angle in radians. None unless compute_phase=True
    """
    if seg_args is None:
        assert window and step, "if seg_args is not specified, window and step must both be specified."

        seg_args = segment_args(rate=rate, duration=len(x)/rate,\
            offset=0, window=window, step=step) #compute input arguments for segment method

    segs = segment(x, **seg_args) #divide audio signal into segments

    if window_func:
        segs *= eval("np.{0}".format(window_func))(segs.shape[1]) #apply Window function

    fft = np.fft.rfft(segs) #Compute fast fourier transform

    img = np.abs(fft)
    if decibel:
        img = to_decibel(img) #Compute magnitude on dB scale
    
    num_fft = segs.shape[1] #Number of points used for the Fourier Transform        
    freq_max = rate / 2. #Maximum frequency
    
    if compute_phase: cpx_angle = np.angle(fft)
    else: cpx_angle = None
    
    return img, freq_max, num_fft, seg_args, cpx_angle

def cqt(x, rate, step, bins_per_oct, freq_min, freq_max=None, window_func='hamming'):
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

    img = librosa.core.cqt(y=x, sr=rate, hop_length=h, fmin=freq_min, n_bins=bins, bins_per_octave=b, window=window_func)
    img = to_decibel(np.abs(img))
    img = np.swapaxes(img, 0, 1)
    
    step = h / rate

    return img, step

def to_decibel(x):
    """ Convert any data array, :math:`y`, typically a spectrogram, from linear scale 
        to decibel scale by applying the operation :math:`20\log_{10}(y)`.

    Args:
        x : numpy array
            Input array
    
    Returns:
        y : numpy array
            Converted array

    Example:
        >>> import numpy as np
        >>> from ketos.audio.utils.misc import to_decibel 
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
    """ Convert any data array, :math:`y`, typically a spectrogram, from decibel scale 
        to linear scale by applying the operation :math:`10^{y/20}`.

    Args:
        y : numpy array
            Input array
    
    Returns:
        x : numpy array
            Converted array

    Example:
        >>> import numpy as np
        >>> from ketos.audio.utils.misc import from_decibel 
        >>> img = np.array([[10., 20.],[30., 40.]])
        >>> img_db = from_decibel(img)
        >>> img_db = np.around(img_db, decimals=2) # only keep up to two decimals
        >>> print(img_db)
        [[  3.16  10.  ]
         [ 31.62 100.  ]]
    """
    x = np.power(10., y/20.)
    return x

def mag2pow(img, num_fft):
    """ Convert a Magnitude spectrogram to a Power spectrogram.

        Args:
            img: numpy.array
                Magnitude spectrogram image, linear axis
            num_fft: int
                Number of points used for the FFT.
        
        Returns:
            : numpy.array
                Power spectrogram image
    """
    return (1.0 / num_fft) * (img ** 2)

def hz_to_mel(freq):
    """ Convert frequency to position on Mel scale
    
        Args:
            freq: array-like
                Frequency in Hz

        Returns:
            mel: array-like
                Mel scale position
    """
    return 2595 * np.log10(1 + freq / 700)

def mel_to_hz(mel):
    """ Convert position on Mel scale to frequency 
    
        Args:
            mel: array-like
                Mel scale position

        Returns:
            freq: array-like
                Frequency in Hz
    """
    return 700 * (10**(mel / 2595) - 1)

def mel_filter_bank(num_fft, rate, num_filters):
    """ Compute Mel-scale filter bank

        Args:
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
            num_filters: int
                The number of filters in the filter bank.

        Returns:
            fbank: numpy.array
                Mel filter bank, has shape (num_filters, int(np.floor(num_fft / 2 + 1))
    """
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(rate / 2)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
    hz_points = mel_to_hz(mel_points)  # Convert Mel to Hz
    bin = np.floor((num_fft + 1) * hz_points / rate)

    bin_edges = hz_points[1:] - 0.5 * np.concatenate([np.diff(hz_points[1:]),[0]])
    bin_edges[0] = 0

    fbank = np.zeros((num_filters, int(np.floor(num_fft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def mag2mel(img, num_fft, rate, num_filters):
    """ Convert a Magnitude spectrogram to a Mel spectrogram.

        Args:
            img: numpy.array
                Magnitude spectrogram image, linear axis
            num_fft: int
                Number of points used for the FFT.
            rate: float
                Sampling rate in Hz.
            num_filters: int
                The number of filters in the filter bank.
        
        Returns:
            mel_spec: numpy.array
                Mel spectrogram image
    """
    power_spec = mag2pow(img, num_fft)
    fbank = mel_filter_bank(num_fft, rate, num_filters)
    mel_spec = np.dot(power_spec, fbank.T)
    return mel_spec

def mag2mfcc(img, num_fft, rate, num_filters, num_ceps, cep_lifter):
    """ Convert a Magnitude spectrogram to a Mel-frequency cepstrum.

        TODO: Check that the formulas used to compute the MFCCs are 
        correct and consistent with those used in librosa:
        https://librosa.org/doc/main/generated/librosa.feature.mfcc.html#  

        Args:
            img: numpy.array
                Magnitude spectrogram image, linear axis
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
            mfcc: numpy.array
                Mel frequency cepstral coefficients
    """
    img = mag2mel(img, num_fft, rate, num_filters)
    img = np.where(img == 0, np.finfo(float).eps, img)  # Numerical Stability
    img = to_decibel(img)  # dB
    
    mfcc = dct(img, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  
    
    return mfcc

def spec2wave(image, phase_angle, num_fft, step_len, num_iters, window_func):
    """ Estimate audio signal from magnitude spectrogram.

        Implements the algorithm described in 

            D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,” IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.            

        Follows closely the implentation of https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py

        Args:
            image: 2d numpy array
                Magnitude spectrogram, linear scale
            phase_angle: 
                Initial condition for phase in radians
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
            >>> #Create a simple sinusoidal audio signal with frequency of 10 Hz
            >>> import numpy as np
            >>> x = np.arange(1000)
            >>> audio = 32600 * np.sin(2 * np.pi * 10 * x / 1000) 
            >>> #Compute the Short Time Fourier Transform of the audio signal 
            >>> #using a window size of 200, step size of 40, and a Hamming window,
            >>> from ketos.audio.utils.misc import stft
            >>> win_fun = 'hamming'
            >>> mag, freq_max, num_fft, _, _ = stft(x=audio, rate=1000, seg_args={'win_len':200, 'step_len':40}, window_func=win_fun)
            >>> #Estimate the original audio signal            
            >>> from ketos.audio.utils.misc import spec2wave
            >>> audio_est = spec2wave(image=mag, phase_angle=0, num_fft=num_fft, step_len=40, num_iters=25, window_func=win_fun)
            >>> #plot the original and the estimated audio signal
            >>> import matplotlib.pyplot as plt
            >>> plt.clf()
            >>> _ = plt.plot(audio)
            >>> plt.savefig("ketos/tests/assets/tmp/sig_orig.png")
            >>> _ = plt.plot(audio_est)
            >>> plt.savefig("ketos/tests/assets/tmp/sig_est.png")

            .. image:: ../../../ketos/tests/assets/tmp/sig_est.png
    """
    # swap axis to conform with librosa 
    image = np.swapaxes(image, 0, 1)
    if np.ndim(phase_angle) == 2: phase_angle = np.swapaxes(phase_angle, 0, 1)

    # settings for FFT and inverse FFT    
    fft_config = dict(n_fft=num_fft, win_length=num_fft, hop_length=step_len, center=False, window=window_func)
    ifft_config = dict(win_length=num_fft, hop_length=step_len, center=False, window=window_func)

    # initial spectrogram for iterative algorithm
    complex_specgram = complex_value(image, phase_angle)

    # Griffin-Lim iterative algorithm
    for i in range(num_iters):
        audio = librosa.istft(complex_specgram, **ifft_config)
        if i != num_iters - 1:
            complex_specgram = librosa.stft(audio, **fft_config)
            _, phase = librosa.magphase(complex_specgram)
            angle = np.angle(phase)
            complex_specgram = complex_value(image, angle)

    # Cut 

    return audio

