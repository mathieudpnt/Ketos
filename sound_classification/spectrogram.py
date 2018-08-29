import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import cv2
from sound_classification.pre_processing import make_frames


class Spectrogram():
    """ Spectrogram
    
        The 0th axis is the time axis (t-axis).
        The 1st axis is the frequency axis (f-axis).
        
        Each axis is characterized by a starting value (tmin and fmin)
        and a resolution or bin size (tres and fres).

        Args:
            image: 2d numpy array
                Spectrogram image 
            NFFT: int
                Number of points used for the Fast-Fourier Transform
            tres: float
                Time resolution in Hz 
            fres: float
                Frequency resolution in Hz
            fmin: float
                Lower limit of frequency axis in Hz (default: 0)
            timestamp: datetime
                Spectrogram time stamp (default: None)
    """


    def __init__(self, image, NFFT, tres, fres, fmin=0, timestamp=None, flabels=None):

        self.image = image
        self.NFFT = NFFT
        self.tres = tres
        self.tmin = 0
        self.fres = fres
        self.fmin = fmin
        self.timestamp = timestamp
        self.flabels = flabels

    @classmethod
    def cropped(cls, spec, tlow=None, thigh=None, flow=None, fhigh=None):
        cropped_spec = cls(image=spec.image, NFFT=spec.NFFT, tres=spec.tres, fres=spec.fres, fmin=spec.fmin, timestamp=spec.timestamp, flabels=spec.flabels)
        cropped_spec.crop(tlow, thigh, flow, fhigh)
        return cropped_spec

    @classmethod
    def from_signal(cls, signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None):
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
                Instance of Spectrogram
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

        # Create spectrogram instance
        spec = cls(image=image, NFFT=NFFT, tres=winstep, fres=fres, timestamp=timestamp)

        return spec

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

        
    def shape(self):
        return self.image.shape


    def taxis(self):
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

    def plot(self, decibel=False):
        """ Plot the spectrogram with proper axes ranges and labels

            Args:
                decibel: bool
                Use linear or logarithmic scale
        """
        img = self.image
        if decibel:
            from sound_classification.pre_processing import to_decibel
            img = to_decibel(img)

        plt.imshow(img.T,aspect='auto',origin='lower',extent=(0,self.duration(),self.fmin,self.fmax()))
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar()
        
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
        
