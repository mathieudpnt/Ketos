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

    Args:
        sig : numpy array
            The signal to be resampled.
        orig_rate: int
            Sampling rate of sig.
        new_rate: int
            New sampling rate
    
    Returns:
        rate : int
            New sampling rate
        sig : mumpy array
            resampled signal
    """



    duration = sig.shape[0] / orig_rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * new_rate / orig_rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return new_rate, sig

    