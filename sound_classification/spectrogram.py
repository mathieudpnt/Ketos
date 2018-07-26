import numpy as np

class Spectrogram():
    """ Spectrogram generated from an audio segment

        Args:
            image: 2d numpy array
                Spectrogram image 
            NFFT: int
                Number of points used for the Fast-Fourier Transform
            length: float
                Length of audio segment in seconds 
            freq_res: float
                Frequency resolution in Hz
            freq_min: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
            
        Attributes:
            freq_max: float
                Upper limit of frequency axis in Hz
    """

    def __init__(self, image, NFFT, length, freq_res, freq_min=0, timestamp=None):

        self.image = image
        self.NFFT = NFFT
        self.length = length
        self.freq_res = freq_res
        self.freq_min = freq_min
        self.timestamp = timestamp
        self.freq_max = freq_min + freq_res * image.shape[1]

    def _find_freq_bin(self, freq):
        bin = int(freq / self.freq_res)
        return bin

    def _crop_freq_image(self, freq_interval):
        low = self._find_freq_bin(freq_interval.low)
        high = self._find_freq_bin(freq_interval.high)

        low = max(0, low)
        high = min(self.image.shape[1], high)

        if low >= high:
            return None

        cropped_image = self.image[:,low:high]
        return cropped_image

    def crop_freq(self, freq_interval):
        cropped_image = self._crop_freq_image(freq_interval)

        cropped_spec = self.__class__(cropped_image, self.NFFT, self.length, self.freq_res, freq_min=freq_interval.low, timestamp=self.timestamp)

        return cropped_spec

    def average(self, freq_interval):
        m = self._crop_freq_image(freq_interval)
        return np.average(m)

    def median(self, freq_interval):
        m = self._crop_freq_image(freq_interval)
        return np.median(m)