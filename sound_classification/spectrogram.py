from abc import ABC
import numpy as np
from scipy.fftpack import dct
from scipy import ndimage
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import math
from sound_classification.pre_processing import make_frames, to_decibel
from sound_classification.audio_signal import AudioSignal
from sound_classification.annotation import AnnotationHandler


def ensure_same_length(specs, pad=False):
    """ Ensure that all spectrograms have the same length/duration.
        
        Args:
            specs: list
                Input spectrograms.
            pad: bool
                If True, the shorter spectrograms will be padded with zeros
                to achieve the same duration as the longest spectrogram. 
                if False, the longer spectrograms will be cropped to achieve 
                the same duration as the shortest spectrogram.
    
        Returns:   
            specs: list
                List of uniform length spectrograms.
    """

    nt = list()
    for s in specs:
        nt.append(s.tbins())

    nt = np.array(nt)
    if pad: 
        n = np.max(nt)
        for s in specs:
            ns = s.tbins()
            if n-ns > 0:
                s.image = np.pad(s.image, pad_width=((0,n-ns),(0,0)), mode='constant')
    else: 
        n = np.min(nt)
        for s in specs:
            s.image = s.image[:n]

    return specs


def interbreed(specs1, specs2, num, scale_min=1, scale_max=1, smooth=True, smooth_par=5, shuffle=False):
    """ Create new spectrograms by superimposing spectrograms from two different groups.

        If the spectrograms have different lengths, the shorter of the two will be placed 
        randomly within the larger one.

        Args:
            specs1: list
                First group of input spectrograms.
            specs2: list
                Second group of input spectrograms.
            num: int
                Number of spectrograms that will be created
            scale_min, scale_max: float, float
                Scale the spectrogram that is being added by a random 
                number between scale_min and scale_max
            smooth: bool
                If True, a smoothing operation will be applied 
                to avoid sharp edges in the result spetrogram
            smooth_par: int
                Smoothing parameter. The larger the value, the less 
                smoothing.
            shuffle: bool
                Select spectrograms from the two groups in random 
                order instead of the order in which they are provided.

        Returns:   
            specs: list
                List of uniform length spectrograms.
    """
    M = len(specs1)
    N = len(specs2)
    x = np.arange(M)
    y = np.arange(N)

    specs = list()
    while len(specs) < num:
        
        if shuffle:
            np.random.shuffle(x)
            np.random.shuffle(y)

        for i in x:
            for j in y:

                # scaling factor
                if scale_max > scale_min:
                    rndm = np.random.random_sample()
                    scale = scale_min + (scale_max - scale_min) * rndm
                else:
                    scale = scale_max

                # placement
                dt = specs1[i].duration() - specs2[j].duration()
                if dt != 0:
                    rndm = np.random.random_sample()
                    delay = np.abs(dt) * rndm
                else:
                    delay = 0

                if dt >= 0:
                    spec_long = specs1[i]
                    spec_short = specs2[j]
                else:
                    spec_short = specs1[i]
                    spec_long = specs2[j]

                spec = spec_long.copy()
                spec.add(spec=spec_short, delay=delay, scale=scale, make_copy=True, smooth=smooth, smooth_par=smooth_par)
                specs.append(spec)

                if len(specs) >= num:
                    return specs

    return specs

class Spectrogram(AnnotationHandler):
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
        super().__init__() # initialize AnnotationHandler

    def copy(self):
        """ Make a deep copy of the spectrogram.

            Returns:
                spec: Spectrogram
                    Spectrogram copy.
        """
        spec = self.__class__()
        spec.image = np.copy(self.image)
        spec.NFFT = self.NFFT
        spec.tres = self.tres
        spec.tmin = self.tmin
        spec.fres = self.fres
        spec.fmin = self.fmin
        spec.timestamp = self.timestamp
        spec.flabels = self.flabels
        spec.tag = self.tag
        spec.labels = self.labels.copy()
        spec.boxes = list()
        for b in self.boxes:
            spec.boxes.append(b.copy())
        return spec

    def _make_spec_from_cut(self, tbin1=None, tbin2=None, fbin1=None, fbin2=None, fpad=False):
        """ Create a new spectrogram from an existing spectrogram by 
            cropping in time and/or frequency.
        
            Args:
                tbin1: int
                    Lower time bin
                tbin2: int
                    Upper time bin
                fbin1: int
                    Lower frequency bin
                fbin2: int
                    Upper frequency bin
                fpad: bool
                    If True, the new spectrogram will have the same 
                    frequency range as the original, but bins outside 
                    the cropping range will be set to zero.

            Returns:
                spec: Spectrogram
                    Created spectrogram.
        """
        # cut image
        if fpad:
            img = self.image[tbin1:tbin2]
            img[:,:fbin1] = 0
            img[:,fbin2:] = 0 
            fmin = self.fmin
        else:
            img = self.image[tbin1:tbin2, fbin1:fbin2]
            fmin = self.fmin + self.fres * fbin1
            fmin = max(fmin, self.fmin)

        # cut labels and boxes
        t1 = self._tbin_low(tbin1)
        t2 = self._tbin_low(tbin2)
        f1 = self._fbin_low(fbin1)
        f2 = self._fbin_low(fbin2)

        labels, boxes = self.cut_annotations(t1=t1, t2=t2, f1=f1, f2=f2)

        spec = Spectrogram(image=img, NFFT=self.NFFT, tres=self.tres, tmin=0, fres=self.fres, fmin=fmin, timestamp=self.timestamp, flabels=None, tag='')
        spec.annotate(labels=labels, boxes=boxes)
        return spec

    def make_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):
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
                decibel: bool
                    Use logarithmic (decibel) scale.

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
        frames = audio_signal.make_frames(winlen, winstep) 

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

        # use logarithmic axis
        if decibel:
            image = to_decibel(image)

        phase_change = diff
        return image, NFFT, fres, phase_change

    def get_data(self):
        """ Get the underlying data numpy array

            Returns:
                self.image: numpy array
                    Image
        """
        return self.image

    def annotate(self, labels, boxes):
        """ Add a set of annotations

            Args:
                labels: list(int)
                    Annotation labels
                boxes: list(tuple)
                    Annotation boxes, specifying the start and stop time of the annotation 
                    and, optionally, the minimum and maximum frequency.
        """
        super().annotate(labels, boxes)
        for b in self.boxes:
            if b[3] == math.inf:
                b[3] = self.fmax()

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
        epsilon = 1E-12

        if np.ndim(x) == 0:
            scalar = True
            x = [x]
        else:
            scalar = False

        x = np.array(x)
        dx = (x_max - x_min) / bins
        b = (x - x_min) / dx
        b[b % 1 == 0.0] += epsilon
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

    def _tbin_low(self, bin):
        """ Get the lower time value of the specified time bin.

            Args:
                bin: int
                    Bin number
        """
        t = self.tmin + bin * self.tres
        return t

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

    def _fbin_low(self, bin):
        """ Get the lower frequency value of the specified frequency bin.

            Args:
                bin: int
                    Bin number
        """
        f = self.fmin + bin * self.fres
        return f

    def tbins(self):
        return self.image.shape[0]


    def fbins(self):
        return self.image.shape[1]


    def fmax(self):
        return self.fmin + self.fres * self.fbins()

        
    def duration(self):
        return self.tbins() * self.tres


    def get_label_vector(self, label):
        """ Get a vector indicating presence/absence (1/0) 
            of the specified annotation label for each 
            time bin.

            Args:
                label: int
                    Label of interest.

            Returns:
                y: numpy array
                    Vector of 0s and 1s with length equal to the number of 
                    time bins.
        """
        y = np.zeros(self.tbins())
        boi, _ = self._select_boxes(label)
        for b in boi:
            t1 = self._find_tbin(b[0])
            t2 = self._find_tbin(b[1]) 
            y[t1:t2] = 1

        return y

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


    def crop(self, tlow=None, thigh=None, flow=None, fhigh=None, keep_time=False):
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
                keep_time: bool
                    Keep the existing time axis. If false, the time axis will be shifted so t=0 corresponds to 
                    the first bin of the cropped spectrogram.
        """
        # crop image
        self.image, tbin1, fbin1 = self._crop_image(tlow, thigh, flow, fhigh)

        # update t_min and f_min
        if keep_time: self.tmin += self.tres * tbin1
        self.fmin += self.fres * fbin1

        # crop labels and boxes
        self.labels, self.boxes = self.cut_annotations(t1=tlow, t2=thigh, f1=flow, f2=fhigh)
        
        if self.flabels != None:
            self.flabels = self.flabels[fbin1:fbin1+self.image.shape[1]]

    def extract(self, label, min_length=None, center=False, fpad=False, make_copy=False):
        """ Extract those segments of the spectrogram where the specified label occurs. 

            After the selected segments have been extracted, this instance contains the 
            remaining part of the spectrogram.

            Args:
                label: int
                    Annotation label of interest. 
                min_length: float
                    If necessary, extend the annotation boxes so that all extracted 
                    segments have a duration of at least min_length (in seconds) or 
                    longer.  
                center: bool
                    Place the annotation box at the center of the extracted segment 
                    (instead of placing it randomly).                     
                fpad: bool
                    If necessary, pad with zeros along the frequency axis to ensure that 
                    the extracted spectrogram had the same frequency range as the source 
                    spectrogram.
                make_copy: bool
                    If true, the extracted portion of the spectrogram is copied rather 
                    than cropped, so this instance is unaffected by the operation.

            Returns:
                specs: list(Spectrogram)
                    List of clipped spectrograms.                
        """
        if make_copy:
            s = self.copy()
        else:
            s = self

        # select boxes of interest (BOI)
        boi, idx = s._select_boxes(label)
        # strech to minimum length, if necessary
        boi = s._stretch(boxes=boi, min_length=min_length, center=center)
        # extract
        res = s._clip(boxes=boi, fpad=fpad)
        # remove extracted labels
        s.delete_annotations(idx)
        
        return res

    def segment(self, number=1, length=None, pad=False):
        """ Split the spectrogram into a number of equally long segments, 
            either by specifying number of segments or segment duration.

            Args:
                number: int
                    Number of segments.
                length: float
                    Duration of each segment in seconds (only applicable if number=1)
                pad: bool
                    If True, pad spectrogram with zeros if necessary to ensure 
                    that bins are used.

            Returns:
                segs: list
                    List of segments
        """        
        epsilon = 1E-12

        if pad:
            f = np.ceil
        else:
            f = np.floor

        if number > 1:
            bins = int(f(self.tbins() / number))
            dt = bins * self.tres
        
        elif length is not None:
            bins = int(np.ceil(self.tbins() * length / self.duration()))
            number = int(f(self.tbins() / bins))
            dt = bins * self.tres

        else:
            return [self]

        t1 = np.arange(number) * dt + epsilon
        t2 = (np.arange(number) + 1) * dt + epsilon
        boxes = np.array([t1,t2])
        boxes = np.swapaxes(boxes, 0, 1)
        boxes = np.pad(boxes, ((0,0),(0,1)), mode='constant', constant_values=0)
        boxes = np.pad(boxes, ((0,0),(0,1)), mode='constant', constant_values=self.fmax()+0.5*self.fres)
        segs = self._clip(boxes=boxes)
        
        return segs

    def _select_boxes(self, label):
        """ Select boxes corresponding to a specified label.

            Args:
                label: int
                    Label of interest

            Returns:
                res: list
                    Selected boxes
                idx: list
                    Indices of selected boxes
        """  
        res, idx = list(), list()
        if len(self.labels) == 0:
            return res, idx

        for i, (b, l) in enumerate(zip(self.boxes, self.labels)):
            if l == label:
                res.append(b)
                idx.append(i)

        return res, idx

    def _stretch(self, boxes, min_length, center=False):
        """ Stretch boxes to ensure that all have a 
            minimum time length.

            Args:
                boxes: list
                    Input boxes
                min_length: float
                    Minimum time length of each box
                center: bool
                    If True, box is streched equally on both sides.
                    If False, the distribution of strech is random.

            Returns:
                res: list
                    Strechted boxes
        """ 
        res = list()
        for b in boxes:
            b = b.copy()
            t1 = b[0]
            t2 = b[1]
            dt = min_length - (t2 - t1)
            if dt > 0:
                if center:
                    r = 0.5
                else:
                    r = np.random.random_sample()
                t1 -= r * dt
                t2 += (1-r) * dt
                if t1 < 0:
                    t2 -= t1
                    t1 = 0
            b[0] = t1
            b[1] = t2
            res.append(b)

        return res

    def _clip(self, boxes, fpad=False):
        """ Extract boxed areas from spectrogram.

            After clipping, this instance contains the remaining part of the spectrogram.

            Args:
                boxes: numpy array
                    2d numpy array with shape=(?,4) 
                fpad: bool
                    If necessary, pad with zeros along the frequency axis to ensure that 
                    the extracted spectrogram had the same frequency range as the source 
                    spectrogram.

            Returns:
                specs: list(Spectrogram)
                    List of clipped spectrograms.                
        """
        if boxes is None or len(boxes) == 0:
            return list()

        if np.ndim(boxes) == 1:
            boxes = [boxes]

        # sort boxes in chronological order
        boxes = sorted(boxes, key=lambda box: box[0])

        boxes = np.array(boxes)
        N = boxes.shape[0]

        # convert from time/frequency to bin numbers
        t1 = self._find_tbin(boxes[:,0]) # start time bin
        t2 = self._find_tbin(boxes[:,1], truncate=False) # end time bin
        f1 = self._find_fbin(boxes[:,2]) # lower frequency bin
        f2 = self._find_fbin(boxes[:,3], truncate=False) # upper frequency bin

        specs = list()

        # loop over boxes
        for i in range(N):            
            spec = self._make_spec_from_cut(tbin1=t1[i], tbin2=t2[i], fbin1=f1[i], fbin2=f2[i], fpad=fpad)
            specs.append(spec)

        # complement
        t2 = np.insert(t2, 0, 0)
        t1 = np.append(t1, self.tbins())
        t2max = 0
        for i in range(len(t1)):
            t2max = max(t2[i], t2max)
            if t2max <= t1[i]:
                if t2max == 0:
                    img_c = self.image[t2max:t1[i]]
                else:
                    img_c = np.append(img_c, self.image[t2max:t1[i]], axis=0)

        self.image = img_c
        self.tmin = 0

        return specs

    def subtract_background(self):
        """ Subtract the median value from each row (frequency bin) 

        """
        self.image = self.image - np.median(self.image, axis=0)

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
    
    def add(self, spec, delay=0, scale=1, make_copy=False, smooth=False, smooth_par=5):
        """ Add another spectrogram to this spectrogram.
            The spectrograms must have the same time and frequency resolution.
            The output spectrogram always has the same dimensions (time x frequency) as the original spectrogram.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
                delay: float
                    Shift the spectrograms by this many seconds
                scale: float
                    Scaling factor for spectrogram to be added 
                make_copy: bool
                    Make a copy of the spectrogram that is being added, so that the instance provided as input argument 
                    to the method is unchanged.       
                smooth: bool
                    Smoothen the edges of the spectrogram that is being added.
                smooth_par: int
                    This parameter can be used to control the amount of smoothing.
                    A value of 1 gives the largest effect. The larger the value, the smaller 
                    the effect.
        """
        assert self.tres == spec.tres, 'It is not possible to add spectrograms with different time resolutions'
        assert self.fres == spec.fres, 'It is not possible to add spectrograms with different frequency resolutions'

        # make copy
        if make_copy:
            sp = spec.copy()
        else:
            sp = spec

        # crop spectrogram
        if delay < 0:
            tlow = sp.tmin - delay
        else:
            tlow = sp.tmin
        thigh = sp.tmin + self.duration() - delay  
        sp.crop(tlow, thigh, self.fmin, self.fmax())

        # fade-in/fade-out
        if smooth:
            sigmas = 3
            p = 2 * np.ceil(smooth_par)
            nt = sp.tbins()
            if nt % 2 == 0:
                mu = nt / 2.
            else:
                mu = (nt - 1) / 2
            sigp = np.power(mu, p) / np.power(sigmas, 2)
            t = np.arange(nt)
            envf = np.exp(-np.power(t-mu, p) / (2 * sigp)) # envelop function = exp(-x^p/2*a^p)
            for i in range(sp.fbins()):
                sp.image[:,i] *= envf # multiply rows by envelope function

        # add
        nt = sp.tbins()
        nf = sp.fbins()
        t1 = self._find_tbin(self.tmin + delay)
        f1 = self._find_fbin(sp.fmin)
        self.image[t1:t1+nt,f1:f1+nf] += scale * sp.image

        # add annotations
        sp._shift_annotations(delay=delay)
        self.annotate(labels=sp.labels, boxes=sp.boxes)

    def append(self, spec):
        """ Append another spectrogram to this spectrogram.
            The spectrograms must have the same dimensions and resolutions.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
        """
        assert self.tres == spec.tres, 'It is not possible to append spectrograms with different time resolutions'
        assert self.fres == spec.fres, 'It is not possible to append spectrograms with different frequency resolutions'

        assert np.all(self.image.shape[1] == spec.image.shape[1]), 'It is not possible to add spectrograms with different frequency range'

        # add annotations
        spec._shift_annotations(delay=self.duration())
        self.annotate(labels=spec.labels, boxes=spec.boxes)

        # append image
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
            decibel: bool
                Use logarithmic (decibel) scale.
    """
    def __init__(self, audio_signal, winlen, winstep, timestamp=None,
                 flabels=None, hamming=True, NFFT=None, compute_phase=False, decibel=False):

        super(MagSpectrogram, self).__init__()
        self.image, self. NFFT, self.fres, self.phase_change = self.make_mag_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        self.tres = winstep
        self.timestamp = timestamp
        self.flabels = flabels


    def make_mag_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):
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
                decibel: bool
                    Use logarithmic (decibel) scale.

            Returns:
                (image, NFFT, fres):numpy.array,int, int
                A tuple with the resulting magnitude spectrogram, the NFFT, the frequency resolution
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        
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
            decibel: bool
                Use logarithmic (decibel) scale.
    """
    def __init__(self, audio_signal, winlen, winstep,flabels=None,
                 hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):

        super(PowerSpectrogram, self).__init__()
        self.image, self. NFFT, self.fres, self.phase_change = self.make_power_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        self.tres = winstep
        self.timestamp = timestamp
        self.flabels = flabels


    def make_power_spec(self, audio_signal, winlen, winstep, hamming=True, NFFT=None, timestamp=None, compute_phase=False, decibel=False):
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
                decibel: bool
                    Use logarithmic (decibel) scale.

            Returns:
                (power_spec, NFFT, fres, phase):numpy.array,int,int,numpy.array
                A tuple with the resulting power spectrogram, the NFFT, the frequency resolution, 
                and the phase spectrogram (only if compute_phase=True).
        """

        image, NFFT, fres, phase_change = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
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

        image, NFFT, fres, _ = self.make_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, decibel=False)
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
