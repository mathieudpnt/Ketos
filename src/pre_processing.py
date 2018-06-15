import os
import traceback
import operator

import numpy as np
import cv2

import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy.fftpack import dct
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


def crop_high_freq_spec(spec, threshold):
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
    id_sizes = np.array(ndimage.sum(img, id_regions, range(num_ids + 1)))
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


    img[img < row_median * row_factor] = 0
    img[img < col_median * col_factor] = 0 
    filtered_img = img
    filtered_img[filtered_img > 0] = 1


    return filtered_img

def preemphasis(sig,coeff=0.97):
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



#TODO: Refactor. Break this function into smaller functions
#  and possibly reuse some of the functions already defined in this module
#TODO: Improve docstring
def extract_mfcc_features(rate,sig, frame_size=0.05, frame_stride=0.03, preemphasis_coeff = 0.97, NFFT=512, n_filters=40, n_ceps=20, cep_lifter=20):
    """ Extract MEL-frequency cepstral coefficients (mfccs) from signal.
    
        Args:
            rate : int
                The sampling rate of the signal (in Hz).                
            sig : numpy array
                The input signal.
            frame_size : float
                Length of each frame (in seconds).
            frame_stride : float
                The length od the stride (in seconds).
            preemphasis_coeff : float
                The preemphasis coefficient. If 0, preemphasis is not applied.
            NFFT : int
                The FFT (Fast Fourier Transform) length to use.
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
    #sample_rate, signal = wavfile.read(path_file)
    emphasized_signal = preemphasis(sig, preemphasis_coeff)

    # params
    '''frame_size = 0.025
    frame_stride = 0.01'''
    frame_length, frame_step = frame_size * rate, frame_stride * rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # hamming window
    frames *= np.hamming(frame_length)

    #NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    #nfilt = 40
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



# Subtract running mean (removes narrowband long-duration noise)

def filter_narrowband_noise(spec, frame_stride, T):
    """ Subtract the running mean from the rows

        The horizontal subtraction results in a reduction of narrowband
        long-duration noise.

    Args:
        spec : numpy array

    Returns:
        filtered_spec: numpy array
            The noise filtered spectogram.
    """
    dt  = frame_stride
    eps = 1 - np.exp((np.log(0.15) * dt / T))
    nx,ny = spec.shape
    rmean = np.zeros(ny)
    rmean = np.average(spec, 0)
    filtered_spec = np.zeros(shape=(nx,ny))
    for ix in range(nx):
        for iy in range(ny):    
            filtered_spec[ix,iy] = spec[ix,iy] - rmean[iy] # subtract running mean
            rmean[iy] = (1 - eps) * rmean[iy] + eps * spec[ix,iy] # update running mean

    return filtered_spec


def filter_broadband_noise(spec, frame_stride, T):
    """ Subtract the running mean from the columns 

        The vertical subtraction results in a reduction of broadband
        sort-duration noise.

    Args:
        spec : numpy array

    Returns:
        filtered_spec: numpy array
            The noise filtered spectogram.
    """

    eps = 0.01
    nx,ny = spec.shape
    rmean = np.zeros(nx)
    rmean = np.average(spec, 1)
    filtered_spec = np.zeros(shape=(nx,ny))
    for iy in range(ny):
        for ix in range(nx):    
            filtered_spec[ix,iy] = spec[ix,iy] - rmean[ix] # subtract running mean
            rmean[ix] = (1 - eps) * rmean[ix] + eps * spec[ix,iy] # update running mean

    return filtered_spec



# Smoothing 

def apply_smoothing(spec):
    """ Apply a smoothing filter to the spectogram

        Args:
            spec : numpy array
                The spectogram to be filtered.

        Returns:
            filtered_spec : numpy array
                The filtered spectogram.
    """


    nx, ny = spec.shape
    smooth_spec = np.zeros(shape=(nx,ny))
    M = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.int32)
    for ix in range(nx):
        for iy in range(ny):    
            for di in range(-1,2):
                for dj in range(-1,2):
                    ix1 = ix+di
                    iy1 = iy+dj
                    if (ix1 >= 0 and ix1 < nx and iy1 >= 0 and iy1 < ny):
                        i = 1 + di
                        j = 1 + dj
                        smooth_spec[ix,iy] += spec[ix1,iy1] * M[i,j]

    return smooth_spec

