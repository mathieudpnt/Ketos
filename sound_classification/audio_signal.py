import numpy as np
import datetime
import scipy.io.wavfile as wave
from sound_classification.data_handling import read_wave


class AudioSignal:
    """ Audio signal

        Args:
            rate: int
                Sampling rate in Hz
            data: 1d numpy array
                Audio data 
    """
    def __init__(self, rate, data, tag=""):
        self.rate = rate
        self.data = data
        self.tag = tag

    @classmethod
    def from_wav(cls, path):
        rate, data = read_wave(path)
        return cls(rate, data, path[path.rfind('/')+1:])

    def to_wav(self, path):
        wave.write(filename=path, rate=self.rate, data=self.data)

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

        return cropped_data        

    def crop(self, begin=None, end=None):
        cropped_data = self._cropped_data(begin,end)
        cropped_signal = self.__class__(rate=self.rate, data=cropped_data)
        return cropped_signal        

    def append(self, signal, overlap_sec=0):
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        # make hard copy
        d = signal.data[:]

        overlap = int(overlap_sec * self.rate)

        # extract data from overlap region
        if overlap > 0:

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
            c = np.empty(overlap)
            for i in range(overlap):
                w = smoothclamp(i, 0, overlap-1)
                c[i] = (1.-w) * a[i] + w * b[i]

            # append
            self.data = np.append(self.data, c)

        self.data = np.append(self.data, d) 

def smoothclamp(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )


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
        cropped_data = self._cropped_data(begin_sec, end_sec)

        time_stamp = self.time_stamp
        if begin_sec > 0 and len(cropped_data) > 0:
            time_stamp += datetime.timedelta(seconds=begin_sec) # update time stamp

        cropped_signal = self.__class__(rate=self.rate, data=cropped_data, time_stamp=time_stamp, tag=self.tag)
        return cropped_signal        


