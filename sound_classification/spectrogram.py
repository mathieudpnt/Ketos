from abc import ABC
import numpy as np
from scipy.fftpack import dct
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import cv2
from sound_classification.pre_processing import make_frames


class Spectrogram():
    """ Spectrogram

        Parent class for spectogram subclasses.
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Attributes:
            signal: AudioSignal object
                Audio signal 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)
"""
    def __init__(self):
        
        self.image = np.zeros((2,2))
        self.shape = self.image.shape
        self.NFFT = 0
        self.tres = 0
        self.tmin = 1
        self.fres = 1
        self.fmin = 0
        self.timestamp = None
        self.flabels = None
    
    def _find_tbin(self, t):
        """ Find bin corresponding to given time.
            Returns -1, if t < t_min
            Returns N, if t > t_max, where N is the number of time bins

            Args:
                t: float
                   Time since spectrogram start in duration

            Returns:
                bin : int
                    Bin number
        """
        bin = int((t - self.tmin) / self.tres)
        
        if bin < 0:
            bin = -1
        if bin >= self.tbins():
            bin = self.tbins()
        
        return bin


    def _find_fbin(self, f):
        """ Find bin corresponding to given frequency
            Returns -1, if f < f_min
            Returns N, if f > f_max, where N is the number of frequency bins

            Args:
                f: float
                   Frequency in Hz 

            Returns:
                bin: int
                     Bin number
        """
        bin = int((f - self.fmin) / self.fres)

        if bin < 0:
            bin = -1
        if bin >= self.fbins():
            bin = self.fbins()

        return bin


    def tbins(self):
        return self.image.shape[0]


    def fbins(self):
        return self.image.shape[1]


    def fmax(self):
        return self.fmin + self.fres * self.fbins()

        
    def duration(self):
        return self.tbins() * self.tres

        
    #TODO: handle datetime=None
    def taxis(self):
        if self.timestamp is not None:
            times = list()
            delta = datetime.timedelta(seconds=self.tres)
            t = self.timestamp + datetime.timedelta(seconds=self.tmin)
            for _ in range(self.tbins()):
                times.append(t)
                t += delta
            
            return times


    def faxis(self):
        if self.flabels == None:
            self.flabels = ['f{0}'.format(x) for x in range(self.fbins())]
        
        return self.flabels


    def _crop_image(self, tlow=None, thigh=None, flow=None, fhigh=None):
        """ Crop spectogram along time axis, frequency axis, or both.
            
            If the cropping box extends beyond the boarders of the spectrogram, 
            the cropped spectrogram is the overlap of the two.
            
            In general, the cuts will not coincide with bin divisions.             
            For the lower cuts, the entire lower bin is included.
            For the higher cuts, the entire upper bin is excluded.

            Args:
                tlow: float
                    Lower limit of time cut, measured in duration from the beginning of the spectrogram
                thigh: float
                    Upper limit of time cut, measured in duration from the beginning of the spectrogram start 
                flow: float
                    Lower limit on frequency cut in Hz
                fhigh: float
                    Upper limit on frequency cut in Hz

            Returns:
                img: 2d numpy array
                     Cropped image
        """
        Nt = self.tbins()
        Nf = self.fbins()
        
        t1 = 0
        t2 = Nt
        f1 = 0
        f2 = Nf

        if tlow != None:
            t1 = max(0, self._find_tbin(tlow))
        if thigh != None:
            t2 = min(Nt, self._find_tbin(thigh))
        if flow != None:
            f1 = max(0, self._find_fbin(flow))
        if fhigh != None:
            f2 = min(Nf, self._find_fbin(fhigh))
            
        if t2 <= t1 or f2 <= f1:
            img = None
        else:
            img = self.image[t1:t2, f1:f2]

        return img, t1, f1


    def crop(self, tlow=None, thigh=None, flow=None, fhigh=None):
        """ Crop spectogram along time axis, frequency axis, or both.
            
            If the cropping box extends beyond the boarders of the spectrogram, 
            the cropped spectrogram is the overlap of the two.
            
            In general, the cuts will not coincide with bin divisions.             
            For the lower cuts, the entire lower bin is included.
            For the higher cuts, the entire upper bin is excluded.

            Args:
                tlow: float
                    Lower limit of time cut, measured in duration from the beginning of the spectrogram
                thigh: float
                    Upper limit of time cut, measured in duration from the beginning of the spectrogram start 
                flow: float
                    Lower limit on frequency cut in Hz
                fhigh: float
                    Upper limit on frequency cut in Hz
        """
        self.image, t1, f1 = self._crop_image(tlow, thigh, flow, fhigh)
        
        self.tmin += self.tres * t1
        self.fmin += self.fres * f1
        
        if self.flabels != None:
            self.flabels = self.flabels[f1:f1+self.shape[1]]


    def average(self, axis=None, tlow=None, thigh=None, flow=None, fhigh=None):
        """ Compute average magnitude within specified time and frequency regions.
            
            If the region extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap, None is returned.

            Args:
                axis: bool
                    Axis along which average is computed, where 0 is the time axis and 1 is the frequency axis. If axis=None, the average is computed along both axes.
                tlow: float
                    Lower limit of time cut, measured in duration from the beginning of the spectrogram
                thigh: float
                    Upper limit of time cut, measured in duration from the beginning of the spectrogram start 
                flow: float
                    Lower limit on frequency cut in Hz
                fhigh: float
                    Upper limit on frequency cut in Hz

            Returns:
                avg : float or numpy array
                    Average magnitude
        """
        m, _, _ = self._crop_image(tlow, thigh, flow, fhigh)

        if m is None or m.size == 0: 
            return np.nan

        avg = np.average(m, axis=axis)

        return avg


    def median(self, axis=None, tlow=None, thigh=None, flow=None, fhigh=None):
        """ Compute median magnitude within specified time and frequency regions.
            
            If the region extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap, None is returned.

            Args:
                axis: bool
                    Axis along which median is computed, where 0 is the time axis and 1 is the frequency axis. If axis=None, the average is computed along both axes.
                tlow: float
                    Lower limit of time cut, measured in duration from the beginning of the spectrogram
                thigh: float
                    Upper limit of time cut, measured in duration from the beginning of the spectrogram start 
                flow: float
                    Lower limit on frequency cut in Hz
                fhigh: float
                    Upper limit on frequency cut in Hz

            Returns:
                med : float or numpy array
                    Median magnitude
        """
        m, _, _ = self._crop_image(tlow, thigh, flow, fhigh)

        if m is None or m.size == 0: 
            return np.nan

        med = np.median(m, axis=axis)

        return med
            
    def blur_gaussian(self, tsigma, fsigma):
        """ Blur the spectrogram using a Gaussian filter.

            This uses the GaussianBlur method from the cv2 package:
            
                https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#gaussianblur

            Args:
                tsigma: float
                    Gaussian kernel standard deviation along time axis. Must be strictly positive.
                fsigma: float
                    Gaussian kernel standard deviation along frequency axis.

            Examples:
            
            >>> from sound_classification.spectrogram import Spectrogram
            >>> from sound_classification.audio_signal import AudioSignal
            >>> import matplotlib.pyplot as plt
            >>> # create audio signal
            >>> s = AudioSignal.morlet(rate=1000, frequency=300, width=1)
            >>> # create spectrogram
            >>> spec = Spectrogram.from_signal(s, winlen=0.2, winstep=0.05)
            >>> # show image
            >>> spec.plot()
            >>> plt.show()
            >>> # apply very small amount (0.01 sec) of horizontal blur
            >>> # and significant amount of vertical blur (30 Hz)  
            >>> spec.blur_gaussian(tsigma=0.01, fsigma=30)
            >>> # show blurred image
            >>> spec.plot()
            >>> plt.show()

            .. image:: _static/morlet_spectrogram.png
                :width: 300px
                :align: left
            .. image:: _static/morlet_spectrogram_blurred.png
                :width: 300px
                :align: right
        """
        assert tsigma > 0, "tsigma must be strictly positive"

        if fsigma < 0:
            fsigma = 0
        
        sigmaX = tsigma / self.tres
        sigmaY = fsigma / self.fres
        
        self.image = cv2.GaussianBlur(src=self.image, ksize=(0,0), sigmaX=sigmaY, sigmaY=sigmaX)
        

class MagSpectrogram(Spectrogram):
    """ Magnitude Spectrogram
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            tres: float
                Time resolution in Hz 
            fres: float
                Frequency resolution in Hz
            fmin: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins. 
                        
    """


    def __init__(self, audio_signal, winlen, winstep, tres, fmin, tmin, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, timestamp=None):

        self.image, self. NFFT, self.fres = self.make_mag_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp)
        self.shape = self.image.shape
        self.tres = winstep
        self.tmin = 0
        self.fmin = 0
        self.timestamp = timestamp
        self.flabels = flabels


    def make_mag_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None) )
            """ Create spectrogram from audio signal
        
            Args:
                signal: AudioSignal
                    Audio signal 
                winlen: float
                    Window size in seconds
                winstep: float
                    Step size in seconds 
                hamming: bool
                    Apply Hamming window
                NFFT: int
                    Number of points for the FFT. If None, set equal to the number of samples.
                timestamp: datetime
                    Spectrogram time stamp (default: None)

            Returns:
                (image, NFFT, fres):numpy.array,int, int
                A tuple with the resulting magnitude spectrogram, the NFFT and the frequency resolution
        """

         # Make frames
        frames = make_frames(signal, winlen, winstep) 

        # Apply Hamming window    
        if hamming:
            frames *= np.hamming(frames.shape[1])

        # Compute fast fourier transform
        image = np.abs(np.fft.rfft(frames, n=NFFT))

        # Number of points used for FFT
        if NFFT is None:
            NFFT = frames.shape[1]
        
        # Frequency resolution
        fres = signal.rate / 2. / image.shape[1]

        return image, NFFT, fres


class PowerSpectrogram(Spectrogram):
    """ Creates a Power Spectrogram from an :class:`audio_signal.AudioSignal`
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            tres: float
                Time resolution in Hz 
            fres: float
                Frequency resolution in Hz
            fmin: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: ?
                ??
                        
    """


    def __init__(self, audio_signal, winlen, winstep, tres, fmin, tmin, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, timestamp=None):

        self.image, self. NFFT, self.fres = self.make_power_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp)
        self.shape = self.image.shape
        self.tres = tres
        self.tmin = tmin
        self.fmin = fmin
        self.timestamp = timestamp
        self.flabels = flabels


    def make_power_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None) )
            """ Create spectrogram from audio signal
        
            Args:
                signal: AudioSignal
                    Audio signal 
                winlen: float
                    Window size in seconds
                winstep: float
                    Step size in seconds 
                hamming: bool
                    Apply Hamming window
                NFFT: int
                    Number of points for the FFT. If None, set equal to the number of samples.
                timestamp: datetime
                    Spectrogram time stamp (default: None)

            Returns:
                (power_spec, NFFT, fres):numpy.array,int, int
                A tuple with the resulting power spectrogram, the NFFT and the frequency resolution
        """

         # Make frames
        frames = make_frames(signal, winlen, winstep) 


        # Apply Hamming window    
        if hamming:
            frames *= np.hamming(frames.shape[1])

        # Compute fast fourier transform
        image = np.abs(np.fft.rfft(frames, n=NFFT))
        

        # Number of points used for FFT
        if NFFT is None:
            NFFT = frames.shape[1]
        
        # Frequency resolution
        fres = signal.rate / 2. / image.shape[1]
        power_spec = image = (1.0/NFFT) * (image ** 2)
        
        return power_spec, NFFT, fres

       
    
class MelSpectrogram(Spectrogram):
    """ Creates a Mel Spectrogram from an :class:`audio_signal.AudioSignal`
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            signal: AudioSignal
                    And instance of the :class:`audio_signal.AudioSignal` class 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            tres: float
                Time resolution in Hz 
            fres: float
                Frequency resolution in Hz
            fmin: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: ?
                ??
                        
    """


    def __init__(self, audio_signal, winlen, winstep, tres, fmin, tmin, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, timestamp=None):

        self.image, self.filter_banks, self. NFFT, self.fres = self.make_mel_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp)
        self.shape = self.image.shape
        self.tres = tres
        self.tmin = tmin
        self.fmin = fmin
        self.timestamp = timestamp
        self.flabels = flabels

    def make_mel_spec(self, audio_signal, winlen, winstep, n_filters=40,
                         n_ceps=20, cep_lifter=20, hamming=True, NFFT=None, timestamp=None) )
        """ Create a Mel spectrogram from audio signal
    
        Args:
            signal: AudioSignal
                Audio signal 
            winlen: float
                Window size in seconds
            winstep: float
                Step size in seconds 
            n_filters: int
                The number of filters in the filter bank.
            n_ceps: int
                The number of Mel-frequency cepstrums.
            cep_lifters: int
                The number of cepstum filters.
            hamming: bool
                Apply Hamming window
            NFFT: int
                Number of points for the FFT. If None, set equal to the number of samples.
            timestamp: datetime
                Spectrogram time stamp (default: None)

        Returns:
            mel_spec: numpy.array
                Array containing the Mel spectrogram
            filter_banks: numpy.array
                Array containing the filter banks
            NFFT: int
                The number of points used for creating the magnitude spectrogram
                (Calculated if not given)
            fres: int
                The calculated frequency resolution
            
            
    """

        # Make frames
    frames = make_frames(signal, winlen, winstep) 


    # Apply Hamming window    
    if hamming:
        frames *= np.hamming(frames.shape[1])

    # Compute fast fourier transform
    image = np.abs(np.fft.rfft(frames, n=NFFT))
    

    # Number of points used for FFT
    if NFFT is None:
        NFFT = frames.shape[1]
    
    # Frequency resolution
    fres = signal.rate / 2. / image.shape[1]
    power_spec = image = (1.0/NFFT) * (image ** 2)
    
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
    
    
    mel_spec = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_ceps + 1)] # Keep 2-13
    
    
    (nframes, ncoeff) = mel_spec.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mel_spec *= lift  
    


    return mel_spec, filter_banks NFFT, fres
