import numpy as np
import datetime


class AudioSignal:
    """ Audio signal

        Args:
            rate: int
                Sampling rate in Hz
            data: 1d numpy array
                Audio data 
    """
    def __init__(self, rate, data):
        self.rate = rate
        self.data = data

    def empty(self):
        return len(self.data) == 0

    def seconds(self):
        return float(len(self.data)) / float(self.rate)


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

    def __init__(self, rate, data, time_stamp, tag=None):
        AudioSignal.__init__(self, rate, data)
        self.time_stamp = time_stamp
        self.tag = tag

    @classmethod
    def from_audio_signal(cls, audio_signal, time_stamp, tag=None):
        return cls(audio_signal.rate, audio_signal.data, time_stamp, tag)

    def begin(self):
        return self.time_stamp

    def end(self):
        duration = len(self.data) / self.rate
        delta = datetime.timedelta(seconds=duration)
        end = self.begin() + delta
        return end 
    
    def crop(self, begin, end):
        i1 = int((begin - self.begin()).total_seconds() * self.rate)
        i2 = int((end - self.begin()).total_seconds() * self.rate)
        i1 = max(i1, 0)
        i2 = min(i2, len(self.data))
        cropped_data = list()
        time_stamp = begin
        if i2 > i1:
            cropped_data = self.data[i1:i2] # crop data
            time_stamp = self.time_stamp + datetime.timedelta(seconds=float(i1)/self.rate) # update time stamp
        
        cropped_signal = self.__class__(rate=self.rate, data=cropped_data, time_stamp=time_stamp, tag=self.tag)
        return cropped_signal

    def append(self, signal):
        assert self.rate == signal.rate, "Cannot merge audio signals with different sampling rates."

        extended_data = np.append(self.data, signal.data) 
    
        tag = self.tag
        if isinstance(signal, TimeStampedAudioSignal):
            tag += ", " + signal.tag
    
        extended_signal = self.__class__(rate=self.rate, data=extended_data, time_stamp=self.time_stamp, tag=tag)
        return extended_signal        

