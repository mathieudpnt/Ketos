import os
import traceback
import operator

import numpy as np
import cv2

import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy import interpolate

import python_speech_features as psf
from pydub import AudioSegment

def standardize_sample_rate(sig, orig_rate, new_rate):
    """ Resample the signal sig to the sampling rate new_rate.

    Note: Code adapted from Kahl et al. (2017)
          Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
          Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py  

    Args:
        sig : numpy array
            The signal to be resampled.
        orig_rate: int
            Sampling rate of sig.
        new_rate: int
            New sampling rate.
    
    Returns:
        rate : int
            New sampling rate.
        sig : mumpy array
            resampled signal.
    """



    duration = sig.shape[0] / orig_rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * new_rate / orig_rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return new_rate, sig


#TODO: Confirm the meaning of winlen and winstep in the paper. Are these in seconds?
def magnitude_spec(sig, rate, winlen, winstep, NFFT):
    """ Create a magnitute spectogram.

        First, the signal is framed into overlapping frames.
        Second, creates the spectogram 

        Note: Code adapted from Kahl et al. (2017)
            Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
            Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py 

    Args:
        sig : numpy array
            Audio signal.
        rate : int
            Sampling rate of the audio signal.
        winlen : float
            Length of each frame (in seconds)
        winstep : float
            Time (in seconds) after the start of the previous frame that the next frame should start.
        NFTT : int
            The FFT (Fast Fourier Transform) length to use.

    Returns:
        spec: numpy array
            Magnitude spectogram.
    """    
    #get frames
    winfunc = lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)        

    #Magnitude Spectrogram
    spec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    return spec


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


def crop_high_freq_spec(spec, threshhold):
    """ Discard high frequencies

    Args:
        spec : numpy array
            Spectogram.
        threshold: int
            Number of rows (starting from the top) to exclude from spectogram.

    Returns:
        cropped_spec: numpy array
            Spectogram without high frequencies. Shape will be (spec.shape[0]-threshold,spec.shape[1])
    """
    cropped_spec = spec[(spec.shape[0] - threshold):, :]

    return cropped_spec


def filter_isolated_cells(img, struct):
    """Remove isolated spots from the img

    Args:
        img : numpy array
            An array like object representing an image. 
        struct :numpy array
            A structuring pattern that defines feature connections.
            Must be symmetric.
    Returns:
        filtered_array : numpy array
            An array containing the input image without the isolated spots.
    """
    filtered_array = np.copy(img)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

#TODO: Check if it's necessary to create a deep copy of the input array.
def blur_img(img,ksize=5):
    """ Smooth the input image using a median blur filter.

    Args:
        img : numpy array
            Image to be processed.
        ksize: int 
            Aperture linear size. Must be ood and greater than 1 (3,5,7,...)

    Returns:
        blur_img: numpy array
            Blurred image
    """
    try:
        assert img.dtype == "float32"#, "img type {0} shoult be 'float32'".format(img.dtype)
    except AssertionError:
        img.dtype = np.float32    
    
    blur_img = cv2.medianBlur(img,ksize)

    return blur_img


def apply_median_thresh(img,row_factor=3, col_factor=4):
    """ Discard pixels that are lower than the median threshold. 

        The resulting image will have 0s for pixels below the threshold and 1s for the pixels above the threshold.

        Note: Code adapted from Kahl et al. (2017)
            Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
            Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py 
    Args:
        img : numpy array
            Array containing the img to be filtered.blur_img
        row_factor: int or float
            Factor by which the row-wise median pixel value will be multiplied in orther to define the threshold.blur_img
        col_factor: int or float
            Factor by which the col-wise median pixel value will be multiplied in orther to define the threshold.

    Returns:
        filtered_img: numpy array
            The filtered image with 0s and 1s.
    """

    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    filtered_img = img[img < row_median * row_factor] = 0
    filtered_img = filtered_img[ filtered_img < col_median * col_factor] = 0 
    filtered_img[filtered_img > 0] = 1

    return filtered_img



