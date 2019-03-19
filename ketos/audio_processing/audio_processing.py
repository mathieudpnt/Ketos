""" Audio processing module within the ketos library

    This module provides utilities to perform various types of 
    operations on audio data.

    Contents:
        FrameMakerForBinaryCNN class: 

    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
             Project goal: The ketos library provides functionalities for handling data, processing audio signals and
             creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
import pandas as pd
import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy.fftpack import dct
from scipy import interpolate
from collections import namedtuple
from numpy import seterr
from sklearn.utils import shuffle


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
                    :width: 250px
                    :align: left

                .. image:: ../../../../ketos/tests/assets/tmp/grunt1_append_to_itself.png
                    :width: 250px
                    :align: left

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
            image_width: int
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
            >>> # create an instance of FrameMakerForBinaryCNN with default args
            >>> from ketos.audio_processing.audio_processing import FrameMakerForBinaryCNN
            >>> fmaker = FrameMakerForBinaryCNN(specs=[spec], label=1, image_width=8)
            >>> # make frames 
            >>> x, y, _ = fmaker.make_frames()
            >>> print(x.shape)
            (143, 8, 224)
            >>> print(y)
            [False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False False False False False False False False False False False False
             False  True  True  True  True  True  True  True  True  True  True  True
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
                :width: 550px
                :align: center

    """
    def __init__(self, specs, label, image_width, step_size=1, signal_width=1, rndm=False, seed=1, equal_rep=False, discard_mixed=False):

        self.idx = 0
        self.specs = specs
        self.label = label
        self.image_width = image_width
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
                    x.shape[1] = image_width
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

        x = make_frames(x, winlen=self.image_width, winstep=self.step_size)
        y = make_frames(y, winlen=self.image_width, winstep=self.step_size)

        Nx = (x.shape[0] - 1) * self.step_size + x.shape[1]
        spec.image = spec.image[:Nx,:]

        y = np.sum(y, axis=1)

        # discard mixed
        if self.discard_mixed:
            x = x[np.logical_or(y==0, y==self.image_width)]
            y = y[np.logical_or(y==0, y==self.image_width)]
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
    """

    x = np.power(10., y/20.)
    return x

def make_frames(sig, winlen, winstep, zero_padding=False):
    """ Split the signal into frames of length 'winlen' with consecutive 
        frames being shifted by an amount 'winstep'. 
        
        If 'winstep' < 'winlen', the frames overlap.

    Args: 
        sig: numpy array
            The signal to be frame.
        winlen: float
            The window length in bins.
        winstep: float
            The window step (or stride) in bins.
        zero_padding: bool
            If necessary, pad the signal with zeros at the end to make sure that all frames have equal number of samples.
            This assures that sample are not truncated from the original signal.

    Returns:
        frames: numpy array
            2-d array with padded frames.
    """

    totlen = sig.shape[0]

    if zero_padding:
        n_frames = int(np.ceil(totlen / winstep))
        n_zeros = max(0, int((n_frames-1) * winstep + winlen - totlen))
        z = np.zeros(n_zeros)
        padded_signal = np.append(sig, z)
    else:
        padded_signal = sig
        if winlen > totlen:
            n_frames = 1
            winlen = totlen
        else:
            n_frames = int(np.floor((totlen-winlen) / winstep)) + 1

    indices = np.tile(np.arange(0, winlen), (n_frames, 1)) + np.tile(np.arange(0, n_frames * winstep, winstep), (winlen, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    return frames


def normalize_spec(spec):
    """Normalize spectogram so that values range from 0 to 1

    Args:
        spec : numpy array
            Spectogram to be normalized.

    Returns:
        normalized_spec : numpy array
            The normalized spectogram, with same shape as the input

    """
    normalized_spec = spec - spec.min(axis=None)
    normalized_spec = normalized_spec / spec.max(axis=None)

    return normalized_spec


def crop_high_freq(spec, index_max):
    """ Discard high frequencies

    Args:
        spec : numpy array
            Spectogram.
        index_max: int
            Remove rows with index >= index_max from spectogram.

    Returns:
        cropped_spec: numpy array
            Spectogram without high frequencies. 
            Note that the dimension of the array is reduced by the number 
            of rows removed.
    """
    if (index_max < 0):
        index_max = 0

    if (index_max >= spec.shape[1]):
        index_max = spec.shape[1] - 1

    cropped_spec = spec[:, :index_max]

    return cropped_spec

def filter_isolated_spots(img, struct):
    """Remove isolated spots from the img

    Args:
        img : numpy array
            An array like object representing an image. 
        struct : numpy array
            A structuring pattern that defines feature connections.
            Must be symmetric.
    Returns:
        filtered_array : numpy array
            An array containing the input image without the isolated spots.
    """
    filtered_array = np.copy(img)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(img, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

def blur_image(img, size=20, sigma=5, Gaussian=True):
    """ Smooth the input image using a median or Gaussian blur filter.
        Note that the input image is recasted as np.float32.

    Args:
        img : numpy array
            Image to be processed. 
        size: int
            Only used by the median filter. Describes  the shape that is taken from the input array,
            at every element position, to define the input to the filter function
        sigma: scalar of sequence of scalars
            Standard deviation for Gaussian kernel (Only used if Gaussian=True). The standard deviations of the Gaussian
            filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
        Gaussian: bool
            Switch between median (default) and Gaussian filter


    Returns:
        blur_img: numpy array
            Blurred image.
    """

    try:
        assert img.dtype == "float32", "img type {0} shoult be 'float32'".format(img.dtype)
    except AssertionError:
        numpy.ndarray.astype(dtype = np.float32)    
    
    if (Gaussian):
        img_blur = ndimage.gaussian_filter(img,sigma=sigma)
    else:
        img_blur = ndimage.median_filter(img,size=size)

    return img_blur

def apply_median_filter(img,row_factor=3, col_factor=4):
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
    """

    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    img[img <= row_median * row_factor] = 0
    img[img <= col_median * col_factor] = 0 
    filtered_img = img
    filtered_img[filtered_img > 0] = 1

    return filtered_img

def apply_preemphasis(sig,coeff=0.97):
    """Apply pre-emphasis to signal

        Args:
            sig : numpy array
                1-d array containing the signal.
            coeff: float
                The preemphasis coefficient. If set to 0,
                 no preemphasis is applied (the output will be the same as the input).
        Returns:
            emphasized_signal : numpy array
                The filtered signal.
    """
    emphasized_signal = np.append(sig[0], sig[1:] - coeff * sig[:-1])
    
    return emphasized_signal


def extract_mfcc_features(mag_frames, NFFT, rate, n_filters=40, n_ceps=20, cep_lifter=20):
    """ Extract MEL-frequency cepstral coefficients (mfccs) from signal.
    
        Args:
            pow_frames : numpy array
                Power spectrogram.
            NFFT : int
                The number of points used for creating the magnitude spectrogram.
            rate : int
                The sampling rate of the signal (in Hz).                
            n_filters: int
                The number of filters in the filter bank.
            n_ceps: int
                The number of Mel-frequency cepstrums.
            cep_lifters: int
                The number of cepstum filters.

        Returns:
            filter_banks : numpy array
                Array containing the filter banks.
            mfcc : numpy array
                Array containing the MFCCs.
    """
    #check that NFFT has sensible value
    n = mag_frames.shape[1] - 1
    assert (NFFT == 2*n or NFFT == 2*n+1), "NFFT does not agree with size of magnitude spectrogram"

    #make Power Spectrogram
    pow_frames = (1.0 / NFFT) * (mag_frames**2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    #num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_ceps + 1)] # Keep 2-13
    
    #cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  
    
    return filter_banks, mfcc

def apply_narrowband_filter(spec, time_res, time_const):
    """ Subtract the running mean from the rows.
    
        The weights used to calculate the running mean decrease exponentially over the time elapsed since the current time. 

        The horizontal subtraction results in a reduction of narrowband
        long-duration noise.

    Args:
        spec : numpy array
            Spectrogram
        time_res : float
            Time resolution of the spectrogram
        time_const : float
            The time constant indicates the time (in seconds) at which weight becomes 15% of the weight applied to the current value.
        
    Returns:
        filtered_spec: numpy array
            The noise filtered spectogram.
    """
    dt = time_res
    T = time_const
    eps = 1 - np.exp((np.log(0.15) * dt / T))
    nx,ny = spec.shape
#    rmean = np.zeros(ny)
    rmean = np.average(spec, 0)
    filtered_spec = np.zeros(shape=(nx,ny))
    for ix in range(nx):
        for iy in range(ny):    
            filtered_spec[ix,iy] = spec[ix,iy] - rmean[iy] # subtract running mean
            rmean[iy] = (1 - eps) * rmean[iy] + eps * spec[ix,iy] # update running mean

    return filtered_spec

def apply_broadband_filter(spec):
    """ Subtract the median from the columns

        The vertical subtraction results in a reduction of broadband
        sort-duration noise.

    Args:
        spec : numpy array
            Spectrogram

    Returns:
        filtered_spec: numpy array
            The noise filtered spectogram.
    """
    nx,ny = spec.shape
    rmean = np.median(spec, 1)
    filtered_spec = np.zeros(shape=(nx, ny))
    for iy in range(ny): # loop over time bins
        for ix in range(nx): # loop over frequency bins
            filtered_spec[ix,iy] = spec[ix,iy] - rmean[ix]

    return filtered_spec

