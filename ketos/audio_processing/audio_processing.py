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
    operations on audio data.

    Contents:
        FrameMakerForBinaryCNN class: 
"""
import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy.fftpack import dct
from scipy import interpolate
from collections import namedtuple
from numpy import seterr
from sklearn.utils import shuffle
from sys import getsizeof
from psutil import virtual_memory


def make_frames(x, win_len, step_len, pad=True, center=True):
    """ Split sequential data into frames of length 'win_len' with consecutive 
        frames being shifted by an amount 'step_len'.

        Args: 
            x: numpy.array
                The data to be framed.
            win_len: int
                Window length.
            winstep: float
                Step size.
            pad: bool
                If necessary, pad the data with zeros to make sure that all frames 
                have an equal number of samples. 
            center: bool
                When padding, apply the same amount of padding before and after to 
                ensure that the data is centered, instead of only padding at the 
                end. Default is True.

        Returns:
            frames: numpy.array
                Framed data.

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

    mem = virtual_memory()

    siz = getsizeof(x) * winlen / winstep

    if siz > 0.1 * mem.total:
        print("Warning: size of output frames exceeds 10% of memory")
        print("Consider reducing the data size and/or increasing the step size and/or reducing the window length")

    totlen = x.shape[0]

    if zero_padding:
        n_frames = int(np.ceil(totlen / winstep))
        n_zeros = max(0, int((n_frames-1) * winstep + winlen - totlen))
        if np.ndim(x) == 1:
            z_shape = n_zeros
        else:
            z_shape = (n_zeros, x.shape[1])
        z = np.zeros(shape=z_shape)
        padded_signal = np.concatenate((x, z))
    else:
        padded_signal = x
        if winlen > totlen:
            n_frames = 1
            winlen = totlen
        else:
            n_frames = int(np.floor((totlen-winlen) / winstep)) + 1

    indices = np.tile(np.arange(0, winlen), (n_frames, 1)) + np.tile(np.arange(0, n_frames * winstep, winstep), (winlen, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    return frames

def stft(self, x, rate, window, step, window_func='hamming', even_len=True):
    """ Compute Short Time Fourier Transform (STFT).

        The window size and step size are rounded to the nearest integer 
        number of samples. The number of points used for the Fourier Transform 
        is equal to the number of samples in the window.
    
        Args:
            x: numpy.array
                Audio signal 
            rate: float
                Sampling rate in Hz
            window: float
                Window length in seconds
            step: float
                Step size in seconds 
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning
            even_len: bool
                If necessary, increase the window length to make it an even number 
                of samples. Default is True.

        Returns:
            img: numpy.array
                Short Time Fourier Transform of the input signal.
            freq_max: float
                Maximum frequency in Hz
            num_fft: int
                Number of points used for the Fourier Transform.
    """
    win_len = int(round(window * rate)) #convert to number of samples
    step_len = int(round(step * rate))
    if even_len:
        win_len += win_len%2 #ensure even window length

    frames = make_frames(win_len=win_len, step_len=step_len) #segment audio signal 
    if window_func:
        frames *= eval("np.{0}".format(window_func))(frames.shape[1]) #apply Window function

    fft = np.fft.rfft(frames) #Compute fast fourier transform
    img = to_decibel(np.abs(fft)) #Compute magnitude on dB scale
    num_fft = frames.shape[1] #Number of points used for the Fourier Transform        
    freq_max = audio_signal.rate / 2. #Maximum frequency
    return img, freq_max, num_fft

def cqt(self, x, rate, step, bins_per_oct, freq_min, freq_max=None):
    """ Compute the CQT spectrogram of an audio signal.

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

def append_specs(specs):
    """ Append spectrograms in the order in which they are provided.

        The spectrograms must have the same time and frequency resolutions
        and share the same frequency axis.

        Args:
            specs: list(Spectrogram)
                Input spectrograms

        Returns:
            s: Spectrogram
                Output spectrogram

            Example:
                >>> # read audio file
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> aud = AudioSignal.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # compute the spectrogram
                >>> from ketos.audio_processing.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram(aud, winlen=0.2, winstep=0.02, decibel=True)
                >>> # keep only frequencies below 800 Hz
                >>> spec.crop(fhigh=800)
                >>> # append spectrogram to itself two times 
                >>> from ketos.audio_processing.audio_processing import append_specs
                >>> merged = append_specs([spec, spec, spec])
                >>> # show orignal spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/grunt1_orig.png")
                >>> # show the merged spectrogram
                >>> fig = merged.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/grunt1_append_to_itself.png")

                .. image:: ../../../../ketos/tests/assets/tmp/grunt1_orig.png

                .. image:: ../../../../ketos/tests/assets/tmp/grunt1_append_to_itself.png

    """
    s = specs[0].copy()
    for i in range(1,len(specs)):
        s.append(specs[i])

    return s

class FrameMakerForBinaryCNN():
    """ Make frames suitable for training a binary CNN.

        Args:
            specs : list
                Spectograms
            label : int
                Label that we want the CNN to learn to detect
            frame_width: int
                Frame width (pixels).
            step_size: int
                Step size (pixels) used for framing. 
            signal_width: int
                Number of time bins that must have the label for the entire 
                frame to be assigned the label.
            rndm: bool
                Randomize the order of the frames
            seed: int
                Seed for random number generator
            equal_rep: bool
                Ensure equal representation of 0s and 1s by removing the 
                more abundant class until there are equally many.
            discard_mixed: bool
                Discard frames which have time bins with different labels

        Attributes:
            idx : int
                Index of current spectrogram. Used for internal book keeping.

        Example:

            >>> # extract saved spectrogram from database file
            >>> import tables
            >>> import ketos.data_handling.database_interface as di
            >>> db = tables.open_file("ketos/tests/assets/cod.h5", "r") 
            >>> table = di.open_table(db, "/sig") 
            >>> spec = di.load_specs(table)[0]
            >>> db.close()
            >>> # create an instance of FrameMakerForBinaryCNN with a single spectrogram as input and frame width of 8
            >>> from ketos.audio_processing.audio_processing import FrameMakerForBinaryCNN
            >>> fmaker = FrameMakerForBinaryCNN(specs=[spec], label=1, frame_width=8)
            >>> # make frames 
            >>> x, y, _ = fmaker.make_frames()
            >>> print(x.shape)
            (143, 8, 224)
            >>> print(y)
            [False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False  True  True  True  True  True  True  True  True  True  True
              True  True  True  True  True  True  True  True  True  True  True  True
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False]
            >>> # show the input spectrogram
            >>> fig = spec.plot(1)
            >>> fig.savefig("ketos/tests/assets/tmp/cod_w_label.png")

            .. image:: ../../../../ketos/tests/assets/tmp/cod_w_label.png

    """
    def __init__(self, specs, label, frame_width, step_size=1, signal_width=1, rndm=False, seed=1, equal_rep=False, discard_mixed=False):

        self.idx = 0
        self.specs = specs
        self.label = label
        self.frame_width = frame_width
        self.step_size = step_size
        self.signal_width = signal_width
        self.rndm = rndm
        self.seed = seed
        self.equal_rep = equal_rep
        self.discard_mixed = discard_mixed

    def eof(self):
        """ Inquire if all data have been read (end of file, eof)

            Returns:
                res : bool
                    True, if all data has been read. False, otherwise               
        """
        res = (self.idx >= len(self.specs))
        return res

    def make_frames(self, max_frames=10000):
        """ Make frames for training a binary CNN.

            Args:
                max_frames : int
                    Return at most this many frames

            Returns:
                x : 3D numpy array
                    Input data for the CNN.
                    x.shape[0] = number of frames
                    x.shape[1] = frame_width
                    x.shape[2] = number of frequency bins (y axis)        
                y : 1D numpy array
                    Labels for input data.
                    y.shape[0] = number of frames
                spec: Spectrogram
                    Merged spectrogram
        """
        x, y = None, None

        # append specs until limit is reached
        num_frames = 0
        spec = None
        while num_frames < max_frames and self.idx < len(self.specs):
            s = self.specs[self.idx]
            self.idx += 1
            if spec is None:
                spec = s
            else:
                spec.append(s)

            num_frames += int(s.image.shape[0] / self.step_size)  # this is only approximate

        x = spec.get_data()
        y = spec.get_label_vector(self.label)

        x = make_frames(x, winlen=self.frame_width, winstep=self.step_size)
        y = make_frames(y, winlen=self.frame_width, winstep=self.step_size)

        Nx = (x.shape[0] - 1) * self.step_size + x.shape[1]
        spec.image = spec.image[:Nx,:]

        y = np.sum(y, axis=1)

        # discard mixed
        if self.discard_mixed:
            x = x[np.logical_or(y==0, y==self.frame_width)]
            y = y[np.logical_or(y==0, y==self.frame_width)]
            y = (y > 0)
        else:
            y = (y >= self.signal_width)

        if self.rndm:
            x, y = shuffle(x, y, random_state=self.seed)

        # ensure equal representation of 0s and 1s
        if self.equal_rep:
            idx0 = pd.Index(np.squeeze(np.where(y == 0)))
            idx1 = pd.Index(np.squeeze(np.where(y == 1)))
            n0 = len(idx0)
            n1 = len(idx1)
            if n0 > n1:
                idx0 = np.random.choice(idx0, n1, replace=False)
                idx0 = pd.Index(idx0)
            else:
                idx1 = np.random.choice(idx1, n0, replace=False) 
                idx1 = pd.Index(idx1)

            idx = idx0.union(idx1)
            x = x[idx]
            y = y[idx]

        return x, y, spec

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


def filter_isolated_spots(img, struct=np.array([[1,1,1],[1,1,1],[1,1,1]])):
    """ Remove isolated spots from the image.

        Args:
            img : numpy array
                An array like object representing an image. 
            struct : numpy array
                A structuring pattern that defines feature connections.
                Must be symmetric.

        Returns:
            filtered_array : numpy array
                An array containing the input image without the isolated spots.

        Example:

            >>> from ketos.audio_processing.audio_processing import filter_isolated_spots
            >>> img = np.array([[0,0,1,1,0,0],
            ...                 [0,0,0,1,0,0],
            ...                 [0,1,0,0,0,0],
            ...                 [0,0,0,0,0,0],
            ...                 [0,0,0,1,0,0]])
            >>> # remove pixels without neighbors
            >>> img_fil = filter_isolated_spots(img)
            >>> print(img_fil)
            [[0 0 1 1 0 0]
             [0 0 0 1 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]
    """
    filtered_array = np.copy(img)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(img, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

def blur_image(img, size=20, sigma=5, gaussian=True):
    """ Smooth the input image using a median or Gaussian blur filter.
        
        Note that the input image is recasted as np.float32.

        This is essentially a wrapper around the scipy.ndimage.median_filter 
        and scipy.ndimage.gaussian_filter methods. 

        For further details, see https://docs.scipy.org/doc/scipy/reference/ndimage.html

        Args:
            img : numpy array
                Image to be processed. 
            size: int
                Only used by the median filter. Describes the shape that is taken from the input array,
                at every element position, to define the input to the filter function.
            sigma: float or array
                Only used by the Gaussian filter. Standard deviation for Gaussian kernel. May be given as a 
                single number, in which case all axes have the same standard deviation, or as an array, allowing 
                for the axes to have different standard deviations.
            Gaussian: bool
                Switch between median and Gaussian (default) filter

        Returns:
            blur_img: numpy array
                Blurred image.

        Example:

            >>> from ketos.audio_processing.audio_processing import blur_image
            >>> img = np.array([[0,0,0],
            ...                 [0,1,0],
            ...                 [0,0,0]])
            >>> # blur using Gaussian filter with sigma of 0.5
            >>> img_blur = blur_image(img, sigma=0.5)
            >>> img_blur = np.around(img_blur, decimals=2) # only keep up to two decimals
            >>> print(img_blur)
            [[0.01 0.08 0.01]
             [0.08 0.62 0.08]
             [0.01 0.08 0.01]]
    """

    try:
        assert img.dtype == "float32", "img type {0} shoult be 'float32'".format(img.dtype)
    except AssertionError:
        img = img.astype(dtype = np.float32)    
    
    if (gaussian):
        img_blur = ndimage.gaussian_filter(img, sigma=sigma)
    else:
        img_blur = ndimage.median_filter(img, size=size)

    return img_blur

def apply_median_filter(img, row_factor=3, col_factor=4):
    """ Discard pixels that are lower than the median threshold. 

        The resulting image will have 0s for pixels below the threshold and 1s for the pixels above the threshold.

        Note: Code adapted from Kahl et al. (2017)
            Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
            Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py 

        Args:
            img : numpy array
                Array containing the img to be filtered. 
                OBS: Note that contents of img are modified by call to function.
            row_factor: int or float
                Factor by which the row-wise median pixel value will be multiplied in orther to define the threshold.
            col_factor: int or float
                Factor by which the col-wise median pixel value will be multiplied in orther to define the threshold.

        Returns:
            filtered_img: numpy array
                The filtered image with 0s and 1s.

        Example:

            >>> from ketos.audio_processing.audio_processing import apply_median_filter
            >>> img = np.array([[1,4,5],
            ...                 [3,5,1],
            ...                 [1,0,9]])
            >>> img_fil = apply_median_filter(img, row_factor=1, col_factor=1)
            >>> print(img_fil)
            [[0 0 0]
             [0 1 0]
             [0 0 1]]
    """

    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    img[img <= row_median * row_factor] = 0
    img[img <= col_median * col_factor] = 0 
    filtered_img = img
    filtered_img[filtered_img > 0] = 1

    return filtered_img

def apply_preemphasis(sig, coeff=0.97):
    """ Apply pre-emphasis to signal

        Args:
            sig : numpy array
                1-d array containing the signal.
            coeff: float
                The preemphasis coefficient. If set to 0,
                no preemphasis is applied (the output will be the same as the input).

        Returns:
            emphasized_signal : numpy array
                The filtered signal.

        Example:

            >>> from ketos.audio_processing.audio_processing import apply_preemphasis
            >>> sig = np.array([1,2,3,4,5])
            >>> sig_new = apply_preemphasis(sig, coeff=0.95)
            >>> print(sig_new)
            [1.   1.05 1.1  1.15 1.2 ]
    """
    emphasized_signal = np.append(sig[0], sig[1:] - coeff * sig[:-1])
    
    return emphasized_signal


def inv_magphase(mag, angle):
    """ Computes complex value from magnitude and phase angle.

        Args:
            mag: numpy array
                Magnitude
            angle: float or numpy array
                Phase angle

        Returns:
            c: numpy array
                Complex value
    """
    phase = np.cos(angle) + 1.j * np.sin(angle)
    c = mag * phase
    return c  


def estimate_audio_signal(image, phase_angle, n_fft, hop, num_iters, window):
    """ Estimate audio signal from magnitude spectrogram.

        Implements the algorithm described in 

            D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,” IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.            

        Follows closely the implentation of https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py

        Args:
            image: 2d numpy array
                Magnitude spectrogram
            phase_angle: 
                Initial condition for phase.
            n_fft: int
                Number of points used for the Fast-Fourier Transform. Same as window size.
            hop: int
                Step size.
            num_iters: 
                Number of iterations to perform.
            window: string, tuple, number, function, np.ndarray [shape=(n_fft,)]
                - a window specification (string, tuple, or number); see `scipy.signal.get_window`
                - a window function, such as `scipy.signal.hamming`
                - a user-specified window vector of length `n_fft`

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
            >>> sig_est = estimate_audio_signal(image=mag, phase_angle=0, n_fft=200, hop=40, num_iters=25, window=np.hamming(frames.shape[1]))
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
    fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=False, window=window)
    ifft_config = dict(win_length=n_fft, hop_length=hop, center=False, window=window)

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
