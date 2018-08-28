import numpy as np
import cv2
import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy.fftpack import dct
from scipy import interpolate
from collections import namedtuple
from sound_classification.audio_signal import AudioSignal, TimeStampedAudioSignal
from numpy import seterr


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

def resample(signal, new_rate):
    """ Resample the acoustic signal with an arbitrary sampling rate.

    Note: Code adapted from Kahl et al. (2017)
          Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
          Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py  

    Args:
        signal : AudioSignal
            The signal to be resampled.
        new_rate: int
            New sampling rate.
    
    Returns:
        new_signal : AudioSignal
            resampled signal.
    """

    orig_rate = signal.rate
    sig = signal.data

    duration = sig.shape[0] / orig_rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * new_rate / orig_rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    new_sig = np.round(new_audio).astype(sig.dtype)

    if isinstance(signal, TimeStampedAudioSignal):
        new_signal = TimeStampedAudioSignal(rate=new_rate, data=new_sig, time_stamp=signal.time_stamp, tag=signal.tag)    
    else:
        new_signal = AudioSignal(rate=new_rate, data=new_sig)    

    return new_signal

def make_frames(signal, winlen, winstep, zero_padding=False):
    """ Split the signal into frames of length 'winlen' with consecutive 
        frames being shifted by an amount 'winstep'. 
        
        If 'winstep' < 'winlen', the frames overlap.

    Args: 
        signal: AudioSignal
            The signal to be framed.
        winlen: float
            The window length in seconds.
        winstep: float
            The window step (or stride) in seconds.
        zero_padding: bool
            If necessary, pad the signal with zeros at the end to make sure that all frames have equal number of samples.
            This assures that sample are not truncated from the original signal.

    Returns:
        frames: numpy array
            2-d array with padded frames.
    """

    rate = signal.rate
    sig = signal.data

    totlen = len(sig)
    winlen = int(round(winlen * rate))
    winstep = int(round(winstep * rate))

    if zero_padding:
        n_frames = int(np.ceil(totlen / winstep))
        n_zeros = max(0, int((n_frames-1) * winstep + winlen - totlen))
        z = np.zeros(n_zeros)
        padded_signal = np.append(sig, z)
    else:
        if winlen > totlen:
            n_frames = 1
            winlen = totlen
        else:
            n_frames = int(np.floor((totlen-winlen) / winstep)) + 1

        l = (n_frames - 1) * winstep + winlen
        padded_signal = np.delete(sig, np.s_[l:])

    indices = np.tile(np.arange(0, winlen), (n_frames, 1)) + np.tile(np.arange(0, n_frames * winstep, winstep), (winlen, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    duration = len(padded_signal) / signal.rate
    
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

def blur_image(img,ksize=3,Gaussian=True):
    """ Smooth the input image using a median or Gaussian blur filter.
        Note that the input image is recasted as np.float32.

    Args:
        img : numpy array
            Image to be processed. 
        ksize: int 
            Aperture linear size. Must be odd integer greater than or equal to 1. For the median filter, the only allowed values are 1, 3, 5. 
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
        img_blur = cv2.GaussianBlur(img,(ksize,ksize),0)
    else:
        try:
            assert ksize < 6, "ksize must be 1, 3, or 5"
        except AssertionError:
            ksize = 5

        img_blur = cv2.medianBlur(img,ksize)

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

