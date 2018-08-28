import numpy as np
import datetime
import scipy.io.wavfile as wave
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
    """
    def __init__(self, rate, data, tag=""):
        self.rate = float(rate)
        self.data = data.astype(dtype=np.float32)
        self.tag = tag

    @classmethod
    def from_wav(cls, path):
        rate, data = read_wave(path)
        return cls(rate, data, path[path.rfind('/')+1:])

    @classmethod
    def gaussian_noise(cls, rate, sigma, samples):
        """ Gaussian noise

            Args:
                rate: float
                    Sampling rate in Hz
                sigma: float
                    Standard deviation of the signal amplitude
                samples: int
                    Length of the audio signal given as the number of samples
        """        
        assert sigma > 0, "sigma must be strictly positive"

        y = np.random.normal(loc=0, scale=sigma, size=samples)
        return cls(rate=rate, data=y, tag="Gaussian_noise_s{0:.3f}s".format(sigma))

    @classmethod
    def morlet(cls, rate, frequency, width, samples=None, height=1, displacement=0, fspread=0):
        """ Audio signal with the shape of the Morlet wavelet

            Note: The computation of the Morlet wavelet signal is very slow for fspread > 0.

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
                fspread: float
                    Frequency spread (standard dev) in Hz
        """        
        if samples is None:
            samples = int(6 * width * rate)

        N = int(samples)

        # compute Morlet function at N equally spaced points
        dt = 1. / rate
        stop = (N-1.)/2. * dt
        start = -stop
        time = np.linspace(start, stop, N)

        if fspread == 0:
            y = morlet_func(time=time, frequency=frequency, width=width, displacement=displacement, norm=False)
        
        else:
            def integrand(x, time, frequency, width, displacement, norm, fspread):
                morlet = morlet_func(time, frequency=x, width=width, displacement=displacement, norm=norm)
                gauss = norm.pdf(x, loc=frequency, scale=fspread)
                return morlet * gauss

            # TODO: Use C function to speed up this step?!
            y = list()
            for t in tqdm(time):
                I = quadrature(func=integrand, a=max(1E-3, frequency-2*fspread), b=frequency+2*fspread, args=(t, frequency, width, displacement, norm, fspread), rtol=0.001, maxiter=1000, vec_func=False)
                y.append(I[0])

            y = np.array(y)

        y *= height
        
        tag = "Morlet_f{0:.0f}Hz_s{1:.3f}s".format(frequency, width)

        return cls(rate=rate, data=np.array(y), tag=tag)


    def to_wav(self, path):
        wave.write(filename=path, rate=int(self.rate), data=self.data.astype(dtype=np.int16))

    def empty(self):
        return len(self.data) == 0

    def seconds(self):
        return float(len(self.data)) / float(self.rate)

    def max(self):
        return max(self.data)

    def min(self):
        return min(self.data)

    def std(self):
        return np.std(self.data)

    def average(self):
        return np.average(self.data)

    def median(self):
        return np.median(self.data)

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

        return np.array(cropped_data)        

    def crop(self, begin=None, end=None):
        self.data = self._cropped_data(begin,end)

    def append(self, signal, overlap_sec=0):
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        # make hard copy
        d = signal.data[:]

        overlap = int(overlap_sec * self.rate)

        # extract data from overlap region
        if overlap > 0:

            overlap = min(overlap, len(self.data))
            overlap = min(overlap, len(d))

            # signal 1
            a = np.empty(overlap)
            np.copyto(a, self.data[-overlap:])
            self.data = np.delete(self.data, np.s_[-overlap:])

            # signal 2
            b = np.empty(overlap)
            np.copyto(b, d[:overlap])
            d = np.delete(d, np.s_[:overlap])

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

        self.data = np.append(self.data, d) 

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


def smoothclamp(x, mi, mx): 
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
    def from_audio_signal(cls, audio_signal, time_stamp, tag=""):
        return cls(audio_signal.rate, audio_signal.data, time_stamp, tag)

    def begin(self):
        return self.time_stamp

    def end(self):
        duration = len(self.data) / self.rate
        delta = datetime.timedelta(seconds=duration)
        end = self.begin() + delta
        return end 
    
    def crop(self, begin=None, end=None):
        begin_sec, end_sec = None, None
        
        if begin is not None:
            begin_sec = (begin - self.begin()).total_seconds()
        if end is not None:
            end_sec = (end - self.begin()).total_seconds()
        
        self.data = self._cropped_data(begin_sec, end_sec)

        if begin_sec > 0 and len(self.data) > 0:
            self.time_stamp += datetime.timedelta(seconds=begin_sec) # update time stamp
