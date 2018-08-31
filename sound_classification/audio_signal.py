import numpy as np
import datetime
import scipy.io.wavfile as wave
from scipy import interpolate
from sound_classification.data_handling import read_wave
from sound_classification.util import morlet_func
import matplotlib.pyplot as plt
from scipy.integrate import quadrature
from scipy.stats import norm
from tqdm import tqdm


class AudioSignal:
    """ Audio signal

        Args:
            rate: float
                Sampling rate in Hz
            data: 1d numpy array
                Audio data 
            tag: str
                Optional meta data string
    """
    def __init__(self, rate, data, tag=""):
        self.rate = float(rate)
        self.data = data.astype(dtype=np.float32)
        self.tag = tag

    @classmethod
    def from_wav(cls, path):
        """ Generate audio signal from wave file

            Args:
                path: str
                    Path to input wave file

            Returns:
                Instance of AudioSignal
                    Audio signal from wave file
        """        
        rate, data = read_wave(path)
        return cls(rate, data, path[path.rfind('/')+1:])

    @classmethod
    def gaussian_noise(cls, rate, sigma, samples):
        """ Generate Gaussian noise signal

            Args:
                rate: float
                    Sampling rate in Hz
                sigma: float
                    Standard deviation of the signal amplitude
                samples: int
                    Length of the audio signal given as the number of samples

            Returns:
                Instance of AudioSignal
                    Audio signal sampling of Gaussian noise
        """        
        assert sigma > 0, "sigma must be strictly positive"

        y = np.random.normal(loc=0, scale=sigma, size=samples)
        return cls(rate=rate, data=y, tag="Gaussian_noise_s{0:.3f}s".format(sigma))

    @classmethod
    def morlet(cls, rate, frequency, width, samples=None, height=1, displacement=0):
        """ Audio signal with the shape of the Morlet wavelet

            Uses :func:`util.morlet_func` to compute the Morlet wavelet.

            Args:
                rate: float
                    Sampling rate in Hz
                frequency: float
                    Frequency of the Morlet wavelet in Hz
                width: float
                    Width of the Morlet wavelet in seconds (sigma of the Gaussian envelope)
                samples: int
                    Length of the audio signal given as the number of samples (if no value is given, samples = 6 * width * rate)
                height: float
                    Peak value of the audio signal
                displacement: float
                    Peak position in seconds

            Returns:
                Instance of AudioSignal
                    Audio signal sampling of the Morlet wavelet 
        """        
        if samples is None:
            samples = int(6 * width * rate)

        N = int(samples)

        # compute Morlet function at N equally spaced points
        dt = 1. / rate
        stop = (N-1.)/2. * dt
        start = -stop
        time = np.linspace(start, stop, N)
        y = morlet_func(time=time, frequency=frequency, width=width, displacement=displacement, norm=False)        
        y *= height
        
        tag = "Morlet_f{0:.0f}Hz_s{1:.3f}s".format(frequency, width) # this is just a string with some helpful info

        return cls(rate=rate, data=np.array(y), tag=tag)

    def copy(self):
        """ Makes a copy of the audio signal.

            Returns:
                Instance of AudioSignal
                    Copied signal
        """        
        data = np.copy(self.data)
        return self.__class__(rate=self.rate, data=data, tag=self.tag)

    def to_wav(self, path):
        """ Save audio signal to wave file

            Args:
                path: str
                    Path to output wave file
        """        
        wave.write(filename=path, rate=int(self.rate), data=self.data.astype(dtype=np.int16))

    def empty(self):
        """ Check if the signal contains any data

            Returns:
                res: bool
                     True if the length of the data array is zero
        """    
        res = len(self.data) == 0    
        return res

    def seconds(self):
        """ Signal duration in seconds

            Returns:
                s: float
                   Signal duration in seconds
        """    
        s = float(len(self.data)) / float(self.rate)
        return s

    def max(self):
        """ Maximum value of the signal

            Returns:
                v: float
                   Maximum value of the data array
        """    
        v = max(self.data)
        return v

    def min(self):
        """ Minimum value of the signal

            Returns:
                v: float
                   Minimum value of the data array
        """    
        v = min(self.data)
        return v

    def std(self):
        """ Standard deviation of the signal

            Returns:
                v: float
                   Standard deviation of the data array
        """   
        v = np.std(self.data) 
        return v

    def average(self):
        """ Average value of the signal

            Returns:
                v: float
                   Average value of the data array
        """   
        v = np.average(self.data)
        return v

    def median(self):
        """ Median value of the signal

            Returns:
                v: float
                   Median value of the data array
        """   
        v = np.median(self.data)
        return v

    def plot(self):
        """ Plot the signal with proper axes ranges and labels
            
            Examples:
            
            >>> from sound_classification.audio_signal import AudioSignal
            >>> import matplotlib.pyplot as plt
            >>> s = AudioSignal.morlet(rate=100, frequency=5, width=1)
            >>> s.plot()
            >>> plt.show() 

            .. image:: _static/morlet.png
                :width: 500px
                :align: center
        """
        start = 0.5 / self.rate
        stop = self.seconds() - 0.5 / self.rate
        num = len(self.data)
        plt.plot(np.linspace(start=start, stop=stop, num=num), self.data)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')

    def _cropped_data(self, begin=None, end=None):
        """ Select a portion of the audio data

            Args:
                begin: float
                    Start time of selection window in seconds
                end: float
                    End time of selection window in seconds

            Returns:
                cropped_data: numpy array
                   Selected portion of the audio data
        """   
        i1 = 0
        i2 = len(self.data)

        if begin is not None:
            begin = max(0, begin)
            i1 = int(begin * self.rate)
            i1 = max(i1, 0)

        if end is not None:
            end = min(self.seconds(), end)
            i2 = int(end * self.rate)
            i2 = min(i2, len(self.data))

        cropped_data = list()
        if i2 > i1:
            cropped_data = self.data[i1:i2] # crop data

        cropped_data = np.array(cropped_data)
        return cropped_data        

    def crop(self, begin=None, end=None):
        """ Clip audio signal

            Args:
                begin: float
                    Start time of selection window in seconds
                end: float
                    End time of selection window in seconds
        """   
        self.data = self._cropped_data(begin, end)

    def append(self, signal, delay=0):
        """ Merge with another audio signal.

            The two audio signals must have the same samling rate.

            If delay < 0, a smooth transition is made between the two signals 
            in the overlap region.

            Note that the current implementation of the smoothing procedure is 
            quite slow, so it is advisable to use small overlap regions.

            If delay == 0, the two signals are joint without any smoothing.

            If delay > 0, a signal with zero sound intensity and duration 
            delay is added between the two audio signals. 

            Args:
                signal: AudioSignal
                    Audio signal to be merged
                delay: float
                    Delay between the two audio signals in seconds.
        """   
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        new_data = signal.data

        # compute overlap (can be negative)
        len_tot = self.merged_length(signal, delay)
        overlap = len(self.data) + len(signal.data) - len_tot

        # extract data from overlap region
        if overlap > 0:

            overlap = min(overlap, len(self.data))
            overlap = min(overlap, len(new_data))

            # signal 1
            a = np.copy(self.data[-overlap:])
            self.data = np.delete(self.data, np.s_[-overlap:])

            # signal 2
            b = np.copy(new_data[:overlap])
            new_data = np.delete(new_data, np.s_[:overlap])

            # superimpose a and b
            # TODO: If possible, vectorize this loop for faster execution
            # TODO: Cache values returned by smoothclamp to avoid repeated calculation
            # TODO: Use coarser binning for smoothing function to speed things up even more
            c = np.empty(overlap)
            for i in range(overlap):
                w = smoothclamp(i, 0, overlap-1)
                c[i] = (1.-w) * a[i] + w * b[i]

            # append
            self.data = np.append(self.data, c)

        elif overlap < 0:
            z = np.zeros(-overlap)
            self.data = np.append(self.data, z)

        self.data = np.append(self.data, new_data) 
        
        assert len(self.data) == len_tot # check that length of merged signal is as expected

    def merged_length(self, signal, delay=0):
        """ Compute sample size of merged signal.

            Args:
                signal: AudioSignal
                    Audio signal to be merged
                delay: float
                    Delay between the two audio signals in seconds.
        """   
        if signal is None:
            return len(self.data)

        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        m = len(self.data)
        n = len(signal.data)
        overlap = -int(delay * self.rate)
        l = m + n - overlap
        return l

    def add_gaussian_noise(self, sigma):
        """ Add Gaussian noise to the signal

            Args:
                sigma: float
                    Standard deviation of the gaussian noise
        """
        noise = AudioSignal.gaussian_noise(rate=self.rate, sigma=sigma, samples=len(self.data))
        self.add(noise)

    def add(self, signal, delay=0, scale=1):
        """ Add the amplitudes of the two audio signals.
        
            The audio signals must have the same sampling rates.

            The summed signal always has the same length as the original signal.

            If the audio signals have different lengths and/or a non-zero delay is selected, 
            only the overlap region will be affected by the operation.
            
            If the overlap region is empty, the original signal is unchanged.

            Args:
                signal: AudioSignal
                    Audio signal to be added
                delay: float
                    Shift the audio signal by this many seconds
                scale: float
                    Scaling factor for signal to be added
        """
        assert self.rate == signal.rate, "Cannot add audio signals with different sampling rates."

        if delay >= 0:
            i_min = int(delay * self.rate)
            j_min = 0
            i_max = min(self.data.shape[0], signal.data.shape[0] + i_min)
            j_max = min(signal.data.shape[0], self.data.shape[0] - i_min)
            
        else:
            i_min = 0
            j_min = int(-delay * self.rate)
            i_max = min(self.data.shape[0], signal.data.shape[0] - j_min)
            j_max = min(signal.data.shape[0], self.data.shape[0] + j_min)
            
        if i_max > i_min and i_max > 0 and j_max > j_min and j_max > 0:
            self.data[i_min:i_max] += scale * signal.data[j_min:j_max]

    def resample(self, new_rate):
        """ Resample the acoustic signal with an arbitrary sampling rate.

        Note: Code adapted from Kahl et al. (2017)
              Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
              Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py  

        Args:
            new_rate: int
                New sampling rate in Hz
        """

        orig_rate = self.rate
        sig = self.data

        duration = sig.shape[0] / orig_rate

        time_old  = np.linspace(0, duration, sig.shape[0])
        time_new  = np.linspace(0, duration, int(sig.shape[0] * new_rate / orig_rate))

        interpolator = interpolate.interp1d(time_old, sig.T)
        new_audio = interpolator(time_new).T

        new_sig = np.round(new_audio).astype(sig.dtype)

        self.rate = new_rate
        self.data = new_sig

def smoothclamp(x, mi, mx): 
        """ Smoothing function
        """    
        return (lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )


class TimeStampedAudioSignal(AudioSignal):
    """ Audio signal with global time stamp and optionally a tag 
        that may be used to indicate the source.

        Args:
            rate: int
                Sampling rate in Hz
            data: 1d numpy array
                Audio data 
            time_stamp: datetime
                Global time stamp marking start of audio recording
            tag: str
                Optional argument that may be used to indicate the source.
    """

    def __init__(self, rate, data, time_stamp, tag=""):
        AudioSignal.__init__(self, rate, data, tag)
        self.time_stamp = time_stamp

    @classmethod
    def from_audio_signal(cls, audio_signal, time_stamp):
        """ Initialize time stamped audio signal from regular audio signal.

            Args:
                audio_signal: AudioSignal
                    Audio signal
                time_stamp: datetime
                    Global time stamp marking start of audio recording
        """
        return cls(rate=audio_signal.rate, data=audio_signal.data, time_stamp=time_stamp, tag=audio_signal.tag)

    @classmethod
    def from_wav(cls, path, time_stamp):
        """ Generate time stamped audio signal from wave file

            Args:
                path: str
                    Path to input wave file
                time_stamp: datetime
                    Global time stamp marking start of audio recording

            Returns:
                Instance of TimeStampedAudioSignal
                    Time stamped audio signal from wave file
        """        
        signal = super(TimeStampedAudioSignal, cls).from_wav(path=path)
        return cls.from_audio_signal(audio_signal=signal, time_stamp=time_stamp)

    def copy(self):
        """ Makes a copy of the time stamped audio signal.

            Returns:
                Instance of TimeStampedAudioSignal
                    Copied signal
        """                
        data = np.copy(self.data)
        return self.__class__(rate=self.rate, data=data, tag=self.tag, time_stamp=self.time_stamp)

    def begin(self):
        """ Get global time stamp marking the start of the audio signal.

            Returns:
                t: datetime
                Global time stamp marking the start of audio the recording
        """
        t = self.time_stamp
        return t

    def end(self):
        """ Get global time stamp marking the end of the audio signal.

            Returns:
                t: datetime
                Global time stamp marking the end of audio the recording
        """
        duration = len(self.data) / self.rate
        delta = datetime.timedelta(seconds=duration)
        t = self.begin() + delta
        return t 
    
    def crop(self, begin=None, end=None):
        """ Clip audio signal

            Args:
                begin: datetime
                    Start data and time of selection window
                end: datetime
                    End date and time of selection window
        """   
        begin_sec, end_sec = None, None
        
        if begin is not None:
            begin_sec = (begin - self.begin()).total_seconds()
        if end is not None:
            end_sec = (end - self.begin()).total_seconds()
        
        self.data = self._cropped_data(begin_sec, end_sec)

        if begin_sec > 0 and len(self.data) > 0:
            self.time_stamp += datetime.timedelta(seconds=begin_sec) # update time stamp

    def append(self, signal, delay=None):
        """ Merge with another time stamped audio signal.

            If delay is None (default), the delay will be determined from the 
            two audio signals' time stamps.

            See :meth:`audio_signal.append` for more details.

            Args:
                signal: AudioSignal
                    Audio signal to be merged
                delay: float
                    Delay between the two audio signals in seconds.
        """   
        if delay is None:
            delay = self.delay(signal)
            if delay is None:
                return super(TimeStampedAudioSignal, self).append(signal=signal)

        return super(TimeStampedAudioSignal, self).append(signal=signal, delay=delay)

    def delay(self, signal):
        """ Compute delay between two time stamped audio signals, defined 
            as the time difference between the end of the first signal and 
            the beginning of the second signal.

            Args:
                signal: TimeStampedAudioSignal
                    Audio signal

            Returns:
                d: float
                    Delay in seconds
        """   
        d = None
        
        if isinstance(signal, TimeStampedAudioSignal):
            d = (signal.begin() - self.end()).total_seconds()
        
        return d