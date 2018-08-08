import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt


class Spectrogram():
    """ Spectrogram generated from an audio segment

        Args:
            image: 2d numpy array
                Spectrogram image 
            NFFT: int
                Number of points used for the Fast-Fourier Transform
            duration: float
                duration of audio segment in duration 
            fres: float
                Frequency resolution in Hz
            fmin: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
    """

    def __init__(self, image, NFFT, duration, fres, fmin=0, timestamp=None):

        self.image = image
        self.NFFT = NFFT
        self.tres = duration / image.shape[0]
        self.time = 0
        self.fres = fres
        self.fmin = fmin
        self.duration = duration
        self.timestamp = timestamp

    def _find_freq_bin(self, freq):
        """ Find bin corresponding to given frequency in Hz

            Args:
                freq: float
                    Frequency in Hz 

            Returns:
                bin : int
                    Bin number
        """
        bin = int((freq - self.fmin) / self.fres)
        return bin

    def _find_tbin(self, t):
        """ Find bin corresponding to given time

            Args:
                t: float
                   Time since spectrogram start in duration

            Returns:
                bin : int
                    Bin number
        """
        bin = int((t - self.tmin) / self.tres)
        return bin

    def _find_fbin(self, f):
        """ Find bin corresponding to given frequency

            Args:
                f: float
                   Frequency in Hz 

            Returns:
                bin: int
                     Bin number
        """
        bin = int((f - self.fmin) / self.fres)
        return bin

    def _crop_freq_image(self, freq_interval):
        """ Crop image along frequency axis.
            
            If the frequency interval extends beyond the boarders of the image, 
            only the overlap region is returned.

            If there is no overlap between the frequency interval and the image, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 

            Returns:
                cropped_image : 2d numpy array
                    Cropped image
        """
        if freq_interval is None:
            return self.image 

        low = self._find_freq_bin(freq_interval.low)
        high = self._find_freq_bin(freq_interval.high)

        # ensure lower and upper limits are within axis range
        low = max(0, low)
        high = min(self.image.shape[1], high)

        if low >= high:
            return None

        cropped_image = self.image[:,low:high]
        return cropped_image

    def fbins(self):
        return self.image.shape[1]

    def tbins(self):
        return self.image.shape[0]

    def fmax(self):
        return self.fmin + self.fres * self.fbins()
        
    def duration(self):
        return self.tbins() * self.tres

    def crop(self, tlow=None, thigh=None, flow=None, fhigh=None):
        """ Crop spectogram along time axis, frequency axis, or both.
            
            If the cropping box extends beyond the boarders of the spectrogram, 
            the cropped spectrogram is the overlap of the two. 

            Args:
                tlow: float
                    Lower limit of time cut, measured in duration from the beginning of the spectrogram start
                thigh: float
                    Upper limit of time cut, measured in duration from the beginning of the spectrogram start 
                flow: float
                    Lower limit on frequency cut in Hz
                fhigh: float
                    Upper limit on frequency cut in Hz
        """
        

    def crop_freq(self, freq_interval):
        """ Crop spectogram along frequency axis.
            
            If the frequency interval extends beyond the boarders of the spectrogram, 
            only the overlap region is returned.

            If there is no overlap between the frequency interval and the spectrogram, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 

            Returns:
                cropped_spec : Spectrogram
                    Cropped spectrogram
        """
        cropped_image = self._crop_freq_image(freq_interval)

        cropped_spec = self.__class__(cropped_image, self.NFFT, self.duration, self.fres, fmin=freq_interval.low, timestamp=self.timestamp)

        return cropped_spec

    def average(self, freq_interval=None, integrate=True):
        """ Compute average magnitude within specified frequency interval.
            
            If the frequency interval extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap between the frequency interval and the spectrogram, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 
                integrate: bool
                    Integrate over frequencies. If 'False' an array is returned instead of a number.

            Returns:
                avg : float
                    Average magnitude
        """
        m = self._crop_freq_image(freq_interval)

        if m is None: 
            return np.nan

        if integrate is True:
            avg = np.average(m)
        else:
            avg = np.average(m, axis=0)

        return avg

    def median(self, freq_interval=None, integrate=True):
        """ Compute median magnitude within specified frequency interval.
            
            If the frequency interval extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap between the frequency interval and the spectrogram, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 
                integrate: bool
                    Integrate over frequencies. If 'False' an array is returned instead of a number.

            Returns:
                med : float or numpy array
                    Average magnitude
        """
        m = self._crop_freq_image(freq_interval)

        if m is None: 
            return np.nan

        if integrate is True:
            med = np.median(m)
        else:
            med = np.median(m, axis=0)

        return med

    def create_plot(self, decibel=False):
        """ Plot the spectrogram with proper axes ranges and labels

            Args:
                decibel: bool
                Use linear or logarithmic scale
        """
        img = self.image
        if decibel:
            from sound_classification.pre_processing import to_decibel
            img = to_decibel(img)

        plt.imshow(img.T,aspect='auto',origin='lower',extent=(0,self.duration,self.fmin,self.freq_max()))
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar()
