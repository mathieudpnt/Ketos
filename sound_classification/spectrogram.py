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
        """ Find bin corresponding to given frequency in Hz

            Args:
                freq: float
                    Frequency in Hz 

            Returns:
                bin : int
                    Bin number
        """
        bin = int(freq / self.freq_res)
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
        low = self._find_freq_bin(freq_interval.low)
        high = self._find_freq_bin(freq_interval.high)

        low = max(0, low)
        high = min(self.image.shape[1], high)

        if low >= high:
            return None

        cropped_image = self.image[:,low:high]
        return cropped_image

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

        cropped_spec = self.__class__(cropped_image, self.NFFT, self.length, self.freq_res, freq_min=freq_interval.low, timestamp=self.timestamp)

        return cropped_spec

    def average(self, freq_interval):
        """ Compute average magnitude within specified frequency interval.
            
            If the frequency interval extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap between the frequency interval and the spectrogram, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 

            Returns:
                avg : float
                    Average magnitude
        """
        m = self._crop_freq_image(freq_interval)

        if m is None: 
            return None

        avg = np.average(m)
        return avg

    def median(self, freq_interval):
        """ Compute median magnitude within specified frequency interval.
            
            If the frequency interval extends beyond the boarders of the spectrogram, 
            only the overlap region is used for the computation.

            If there is no overlap between the frequency interval and the spectrogram, 
            None is returned.

            Args:
                freq_interval: Interval
                    Frequency interval with limits given in Hz 

            Returns:
                med : float
                    Average magnitude
        """
        m = self._crop_freq_image(freq_interval)

        if m is None: 
            return None

        med = np.average(m)
        return med