import numpy as np
import datetime
import scipy.io.wavfile as wave
from sound_classification.data_handling import read_wave
from sound_classification.util import morlet_func


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
    def morlet(cls, rate, frequency, width, samples=None, height=1, displacement=0):
        """ Audio signal with the shape of the Morlet wavelet

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
        """
        
        if samples is None:
            samples = int(6 * width * rate)
        
        # compute Morlet function at N equally spaced points
        N = int(samples)
        dt = 1. / rate
        stop = (N-1.)/2. * dt
        start = -stop
        t = np.linspace(start, stop, N)
        y = morlet_func(t, frequency=frequency, width=width, displacement=displacement, norm=False)
        y *= height
        
        return cls(rate=rate, data=np.array(y), tag="Morlet_f{0:.0f}Hz_s{1:.3f}s")
        

    def to_wav(self, path):
        wave.write(filename=path, rate=int(self.rate), data=self.data.astype(dtype=np.int16))

    def empty(self):
        return len(self.data) == 0

    def seconds(self):
        return float(len(self.data)) / float(self.rate)

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
