from abc import ABC
import numpy as np
from scipy.fftpack import dct
from scipy import ndimage
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
from sound_classification.pre_processing import make_frames
from sound_classification.audio_signal import AudioSignal


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
    def __init__(self, image=np.zeros((2,2)), NFFT=0, tres=1, tmin=0, fres=1, fmin=0, timestamp=None, flabels=None, tag=''):
        
        self.image = image
        self.NFFT = NFFT
        self.tres = tres
        self.tmin = tmin
        self.fres = fres
        self.fmin = fmin
        self.timestamp = timestamp
        self.flabels = flabels
        self.tag = tag

    def make_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False):
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
                compute_phase: bool
                    Compute phase spectrogram in addition to magnitude spectrogram

            Returns:
                image: numpy.array
                    Magnitude spectrogram
                NFFT: int, int
                    Number of points used for the FFT
                fres: int
                    Frequency resolution
                phase_change: numpy.array
                    Phase change spectrogram. Only computed if compute_phase=True.
        """

         # Make frames
        frames = make_frames(audio_signal, winlen, winstep) 

        # Apply Hamming window    
        if hamming:
            frames *= np.hamming(frames.shape[1])

        # Compute fast fourier transform
        fft = np.fft.rfft(frames, n=NFFT)

        # Compute magnitude
        image = np.abs(fft)

        # Compute phase
        if compute_phase:
            phase = np.angle(fft)

            # phase discontinuity due to mismatch between step size and bin frequency
            N = int(round(winstep * audio_signal.rate))
            T = N / audio_signal.rate
            f = np.arange(image.shape[1], dtype=np.float64)
            f += 0.5
            f *= audio_signal.rate / 2. / image.shape[1]
            p = f * T
            jump = 2*np.pi * (p - np.floor(p))
            corr = np.repeat([jump], image.shape[0], axis=0)
            
            # observed phase difference
            shifted = np.append(phase, [phase[-1,:]], axis=0)
            shifted = shifted[1:,:]
            diff = shifted - phase

            # observed phase difference corrected for discontinuity
            diff[diff < 0] = diff[diff < 0] + 2*np.pi
            diff -= corr
            diff[diff < 0] = diff[diff < 0] + 2*np.pi

            # mirror at pi
            diff[diff > np.pi] = 2*np.pi - diff[diff > np.pi]

        else:
            diff = None

        # Number of points used for FFT
        if NFFT is None:
            NFFT = frames.shape[1]
        
        # Frequency resolution
        fres = audio_signal.rate / 2. / image.shape[1]

        phase_change = diff
        return image, NFFT, fres, phase_change

    def get_data(self):
        return self.image

    def _find_bin(self, x, bins, x_min, x_max, truncate=False):
        """ Find bin corresponding to given value.

            Returns -1, if x < x_min
            Returns N, if x > x_max, where N is the number of bins

            Args:
                x: float
                   Value

            Returns:
                bin : int
                    Bin number
        """
        if np.ndim(x) == 0:
            scalar = True
            x = [x]
        else:
            scalar = False

        x = np.array(x)
        dx = (x_max - x_min) / bins
        b = (x - x_min) / dx
        b = b.astype(dtype=int, copy=False)

        if truncate:
            b[b < 0] = 0
            b[b >= bins] = bins - 1
        else:
            b[b < 0] = -1
            b[b >= bins] = bins

        if scalar:
            b = b[0]

        return b


    def _find_tbin(self, t, truncate=True):
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
        tmax = self.tmin + self.tbins() * self.tres
        bin = self._find_bin(x=t, bins=self.tbins(), x_min=self.tmin, x_max=tmax, truncate=truncate)
        return bin


    def _find_fbin(self, f, truncate=True):
        """ Find bin corresponding to given frequency.

            Returns -1, if f < f_min.
            Returns N, if f > f_max, where N is the number of frequency bins.

            Args:
                f: float
                   Frequency in Hz 

            Returns:
                bin: int
                     Bin number
        """
        bin = self._find_bin(x=f, bins=self.fbins(), x_min=self.fmin, x_max=self.fmax(), truncate=truncate)
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
                t1: int
                    Lower time bin
                f1: int
                    Lower frequency bin
        """
        Nt = self.tbins()
        Nf = self.fbins()
        
        t1 = 0
        t2 = Nt
        f1 = 0
        f2 = Nf

        if tlow != None:
            t1 = self._find_tbin(tlow, truncate=True)
        if thigh != None:
            t2 = self._find_tbin(thigh, truncate=True)
        if flow != None:
            f1 = self._find_fbin(flow, truncate=True)
        if fhigh != None:
            f2 = self._find_fbin(fhigh, truncate=True)
            
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
            self.flabels = self.flabels[f1:f1+self.image.shape[1]]


    def clip(self, boxes, fpad=False):
        """ Extract boxed areas from spectrogram.

            After clipping, this instance contains the remaining part of the spectrogram.

            Args:
                boxes: numpy array
                    2d numpy array with shape=(?,4)   

            Returns:
                specs: list(Spectrogram)
                    List of clipped spectrograms.                
        """
        if np.ndim(boxes) == 1:
            boxes = [boxes]

        # sort boxes in chronological order
        sorted(boxes, key=lambda box: box[0])

        boxes = np.array(boxes)
        N = boxes.shape[0]

        # get cuts
        t1 = self._find_tbin(boxes[:,0]) # start time
        t2 = self._find_tbin(boxes[:,1]) # end time
        if boxes.shape[1] == 4:
            f1 = self._find_fbin(boxes[:,2]) # lower frequency
            f2 = self._find_fbin(boxes[:,3]) # upper frequency
        else:
            f1 = np.zeros(N, dtype=int)
            f2 = np.ones(N, dtype=int) * self.fbins()
        
        specs = list()

        # loop over boxes
        for i in range(N):
            
            fmin = self.fmin + self.fres * f1[i]
            fmin = max(fmin, self.fmin) # min frequency in Hz

            # select box
            if fpad is True:
                img = self.image[t1[i]:t2[i],:]
                img[:,0:f1] = 0
                img[:,f2:] = 0 
            else:
                img = self.image[t1[i]:t2[i], f1[i]:f2[i]]

            spec = Spectrogram(image=img, tres=self.tres, fmin=fmin, fres=self.fres)

            specs.append(spec)

        # complement
        t2 = np.insert(t2, 0, 0)
        t1 = np.append(t1, self.tbins())
        t2max = 0
        for i in range(len(t1)):
            t2max = max(t2[i], t2max)
            if t2max < t1[i]:
                if t2max == 0:
                    img_c = self.image[t2max:t1[i]]
                else:
                    img_c = np.append(img_c, self.image[t2max:t1[i]], axis=0)

        self.image = img_c
        self.tmin = 0

        return specs

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

            This uses the Gaussian filter method from the scipy.ndimage package:
            
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

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
        
        self.image = ndimage.gaussian_filter(input=self.image, sigma=(sigmaX,sigmaY))
    
    def add(self, spec, delay=0, scale=1):
        """ Add another spectrogram to this spectrogram.
            The spectrograms must have the same time and frequency resolution.
            The output spectrogram always has the same dimensions (time x frequency) as the original spectrogram.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
                delay: float
                    Shift the audio signal by this many seconds
                scale: float
                    Scaling factor for signal to be added        
        """
        assert self.tres == spec.tres, 'It is not possible to add spectrograms with different time resolutions'
        assert self.fres == spec.fres, 'It is not possible to add spectrograms with different frequency resolutions'

        # crop spectrogram
        if delay < 0:
            tlow = spec.tmin - delay
        else:
            tlow = spec.tmin
        thigh = spec.tmin + self.duration() - delay  
        spec.crop(tlow, thigh, self.fmin, self.fmax())

        # add
        nt = spec.tbins()
        nf = spec.fbins()
        t1 = self._find_tbin(self.tmin + delay)
        f1 = self._find_fbin(spec.fmin)
        self.image[t1:t1+nt,f1:f1+nf] += scale * spec.image

    def append(self, spec):
        """ Append another spectrogram to this spectrogram.
            The spectrograms must have the same dimensions and resolutions.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
        """
        assert self.tres == spec.tres, 'It is not possible to add spectrograms with different time resolutions'
        assert self.fres == spec.fres, 'It is not possible to add spectrograms with different frequency resolutions'

        assert np.all(self.image.shape == spec.image.shape), 'It is not possible to add spectrograms with different shapes'

        self.image = np.append(self.image, spec.image, axis=0)


    def plot(self, decibel=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                decibel: bool
                Use linear (if False) or logarithmic scale (if True)
            
            Returns:
            fig: matplotlib.figure.Figure
            A figure object.

        """
        img = self.image
        if decibel:
            from sound_classification.pre_processing import to_decibel
            img = to_decibel(img)

        fig, ax = plt.subplots()
        img_plot = ax.imshow(img.T,aspect='auto',origin='lower',extent=(0,self.duration(),self.fmin,self.fmax()))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        if decibel:
            fig.colorbar(img_plot, format='%+2.0f dB')
        else:
            fig.colorbar(img_plot,format='%+2.0f')  
        return fig


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
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins.     
            compute_phase: bool
                Compute phase spectrogram in addition to magnitude spectrogram
    """
    def __init__(self, audio_signal, winlen, winstep, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, compute_phase=False):

        super(MagSpectrogram, self).__init__()
        self.image, self. NFFT, self.fres, self.phase_change = self.make_mag_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase)
        self.tres = winstep
        self.timestamp = timestamp
        self.flabels = flabels


    def make_mag_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False):
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
                compute_phase: bool
                    Compute phase spectrogram in addition to magnitude spectrogram

            Returns:
                (image, NFFT, fres):numpy.array,int, int
                A tuple with the resulting magnitude spectrogram, the NFFT, the frequency resolution
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase)
        
        return image, NFFT, fres, phase_change

    def audio_signal(self):
        """ Generate audio signal from magnitude spectrogram
            
            Returns:
                a: AudioSignal
                    Audio signal

            TODO: Check that this implementation is correct!
                  (For example, the window function is not being 
                  taken into account which I believe it should be)
        """
        y = np.fft.irfft(self.image)
        d = self.tres * self.fres * (y.shape[1] + 1)
        N = int(np.ceil(y.shape[0] * d))
        s = np.zeros(N)
        for i in range(y.shape[0]):
            i0 = i * d
            for j in range(0, y.shape[1]):
                k = int(np.ceil(i0 + j))
                if k < N:
                    s[k] += y[i,j]
        rate = int(np.ceil((N+1) / self.duration()))
        a = AudioSignal(rate=rate, data=s[:N])
        return a


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
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels:list of strings
                List of labels for the frequency bins.
            compute_phase: bool
                Compute phase spectrogram in addition to power spectrogram                        
    """
    def __init__(self, audio_signal, winlen, winstep,flabels=None,
                 hamming=True, NFFT=None, timestamp=None, compute_phase=False):

        super(PowerSpectrogram, self).__init__()
        self.image, self. NFFT, self.fres, self.phase_change = self.make_power_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase)
        self.tres = winstep
        self.timestamp = timestamp
        self.flabels = flabels


    def make_power_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False):
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
                compute_phase: bool
                    Compute phase spectrogram in addition to power spectrogram

            Returns:
                (power_spec, NFFT, fres, phase):numpy.array,int,int,numpy.array
                A tuple with the resulting power spectrogram, the NFFT, the frequency resolution, 
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase)
        power_spec = (1.0/NFFT) * (image ** 2)
        
        return power_spec, NFFT, fres, phase_change

       
    
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
            timestamp: datetime
                Spectrogram time stamp (default: None)
            flabels: list of strings
                List of labels for the frequency bins.
    """


    def __init__(self, audio_signal, winlen, winstep,flabels=None, hamming=True, 
                 NFFT=None, timestamp=None, **kwargs):

        super(MelSpectrogram, self).__init__()
        self.image, self.filter_banks, self.NFFT, self.fres = self.make_mel_spec(audio_signal, winlen, winstep,
                                                                                 hamming=hamming, NFFT=NFFT, timestamp=timestamp, **kwargs)
        self.tres = winstep
        self.timestamp = timestamp
        self.flabels = flabels

    def make_mel_spec(self, audio_signal, winlen, winstep, n_filters=40,
                         n_ceps=20, cep_lifter=20, hamming=True, NFFT=None, timestamp=None):
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

        image, NFFT, fres, _ = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp)
        power_spec = (1.0/NFFT) * (image ** 2)
        
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (audio_signal.rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / audio_signal.rate)

        fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, n_filters + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(power_spec, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        
        
        mel_spec = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_ceps + 1)] # Keep 2-13
                
        (nframes, ncoeff) = mel_spec.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mel_spec *= lift  
        
        return mel_spec, filter_banks, NFFT, fres

    def plot(self,filter_bank=False, decibel=False):
        """ Plot the spectrogram with proper axes ranges and labels.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                decibel: bool
                    Use linear (if False) or logarithmic scale (if True)
                filter_bank: bool
                    Plot the filter banks if True. If false (default) print the mel spectrogram.
            
            Returns:
            fig: matplotlib.figure.Figure
            A figure object.

        """
        if filter_bank:
            img = self.filter_banks
        else:
            img = self.image

        if decibel:
            from sound_classification.pre_processing import to_decibel
            img = to_decibel(img)

        fig, ax = plt.subplots()
        img_plot = ax.imshow(img.T,aspect='auto',origin='lower',extent=(0,self.duration(),self.fmin,self.fmax()))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        if decibel:
            fig.colorbar(img_plot,format='%+2.0f dB')
        else:
            fig.colorbar(img_plot,format='%+2.0f')  
        return fig
