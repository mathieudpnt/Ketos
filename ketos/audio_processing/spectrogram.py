""" Spectrogram module within the ketos library

    This module provides utilities to work with spectrograms.

    Spectrograms are two-dimensional visual representations of 
    sound waves, in which time is shown along the horizontal 
    axis, frequency along the vertical axis, and color is used 
    to indicate the sound amplitude. Read more on Wikipedia:

    https://en.wikipedia.org/wiki/Spectrogram

    Contents:
        Spectrogram class
        MagSpectrogram class
        PowerSpectrogram class
        MelSpectrogram class

    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
    Project goal: The ketos library provides functionalities for handling data, processing audio signals and
    creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np
from scipy.fftpack import dct
from scipy import ndimage
from skimage.transform import rescale
import matplotlib.pyplot as plt
import time
import datetime
import math
from ketos.audio_processing.audio_processing import make_frames, to_decibel
from ketos.audio_processing.audio import AudioSignal
from ketos.audio_processing.annotation import AnnotationHandler
from ketos.utils import random_floats


def ensure_same_length(specs, pad=False):
    """ Ensure that all spectrograms have the same length

        Note that all spectrograms must have the same time resolution.
        If this is not the case, an assertion error will be thrown.
        
        Args:
            specs: list
                Input spectrograms
            pad: bool
                If True, the shorter spectrograms will be padded with zeros. If False, the longer spectrograms will be cropped (removing late times)
    
        Returns:   
            specs: list
                List of same-length spectrograms

        Example:
        >>> from ketos.audio_processing.audio import AudioSignal
        >>> 
        >>> # Create two audio signals with different lengths
        >>> audio1 = AudioSignal.morlet(rate=100, frequency=5, width=1)   
        >>> audio2 = AudioSignal.morlet(rate=100, frequency=5, width=1.5)
        >>>
        >>> # Compute spectrograms
        >>> spec1 = MagSpectrogram(audio1, winlen=0.2, winstep=0.05)
        >>> spec2 = MagSpectrogram(audio2, winlen=0.2, winstep=0.05)
        >>> 
        >>> # Print the durations
        >>> print('{0:.2f}, {1:.2f}'.format(spec1.duration(), spec2.duration()))
        5.85, 8.85

        >>> # Ensure all spectrograms have same duration as the shortest spectrogram
        >>> specs = ensure_same_length([spec1, spec2])
        >>> print('{0:.2f}, {1:.2f}'.format(specs[0].duration(), specs[1].duration()))
        5.85, 5.85
    """
    if len(specs) == 0:
        return specs

    tres = specs[0].tres # time resolution of 1st spectrogram

    nt = list()
    for s in specs:
        assert s.tres == tres, 'Spectrograms must have the same time resolution' 
        nt.append(s.tbins())

    nt = np.array(nt)
    if pad: 
        n = np.max(nt)
        for s in specs:
            ns = s.tbins()
            if n-ns > 0:
                s.image = np.pad(s.image, pad_width=((0,n-ns),(0,0)), mode='constant')
                s.time_vector = np.append(s.time_vector, np.zeros(n-ns))
                s.file_vector = np.append(s.file_vector, np.zeros(n-ns))
    else: 
        n = np.min(nt)
        for s in specs:
            s.image = s.image[:n]
            s.time_vector = s.time_vector[:n]
            s.file_vector = s.file_vector[:n]

    return specs


def interbreed(specs1, specs2, num, smooth=True, smooth_par=5, shuffle=True,\
            scale_min=1, scale_max=1, t_scale_min=1, t_scale_max=1,\
            f_scale_min=1, f_scale_max=1, seed=1, validation_function=None):
    """ Interbreed spectrograms to create new ones.

        Interbreeding consists in adding/superimposing two spectrograms on top of each other.

        If the spectrograms have different lengths, the shorter of the two will be placed 
        within the larger one with a randomly generated time offset.

        The shorter spectrogram may also be subject to re-scaling along any of its dimensions, as 
        specified via the arguments t_scale_min, t_scale_max, f_scale_min, f_scale_max, scale_min, scale_max.

        Note that the spectrograms must have the same time and frequency resolution. Otherwise an assertion error will be thrown.

        Args:
            specs1: list
                First group of input spectrograms.
            specs2: list
                Second group of input spectrograms.
            num: int
                Number of spectrograms that will be created
            smooth: bool
                If True, a smoothing operation will be applied 
                to avoid sharp discontinuities in the resulting spetrogram
            smooth_par: int
                Smoothing parameter. The larger the value, the less 
                smoothing. Only applicable if smooth is set to True
            shuffle: bool
                Select spectrograms from the two groups in random 
                order instead of the order in which they are provided.
            scale_min, scale_max: float, float
                Scale the spectrogram that is being added by a random 
                number between scale_min and scale_max
            t_scale_min, t_scale_max: float, float
                Scale the time axis of the spectrogram that is being added 
                by a random number between t_scale_min and t_scale_max
            f_scale_min, f_scale_max: float, float
                Scale the frequency axis of the spectrogram that is being added 
                by a random number between f_scale_min and f_scale_max
            seed: int
                Seed for numpy's random number generator
            validation_function:
                This function is applied to each new spectrogram. The function must accept 'spec1', 'spec2', and 'new_spec'; returns True or False. If True, the new spectrogram is accepted; if False, it gets discarded.

        Returns:   
            specs: Spectrogram or list of Spectrograms
                Created spectrogram(s)

        Example:
            >>> # extract saved spectrograms from database file
            >>> import tables
            >>> import ketos.data_handling.database_interface as di
            >>> db = tables.open_file("ketos/tests/assets/morlet.h5", "r") 
            >>> spec1 = di.get_objects(di.open(db, "/spec1"))[0]
            >>> spec2 = di.get_objects(di.open(db, "/spec2"))[0]
            >>> db.close()
            >>> 
            >>> # interbreed the two spectrograms once to make one new spectrogram
            >>> from ketos.audio_processing.spectrogram import interbreed
            >>> new_spec = interbreed([spec1], [spec2], num=1)
            >>>
            >>> # plot the original spectrograms and the new one
            >>> import matplotlib.pyplot as plt
            >>> fig = spec1.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec1.png")
            >>> fig = spec2.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec2.png")
            >>> fig = new_spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/new_spec.png")

            .. image:: ../../../../ketos/tests/assets/tmp/spec1.png
                :width: 300px
                :align: left
                

            .. image:: ../../../../ketos/tests/assets/tmp/spec2.png
                :width: 300px
                :align: left

            .. image:: ../../../../ketos/tests/assets/tmp/new_spec.png
                :width: 300px
                :align: center

            >>> # Interbreed the two spectrograms to make 3 new spectrograms.
            >>> # Apply a random scaling factor between 0.0 and 5.0.
            >>> # Only accept spectrograms with peak value at least two times 
            >>> # larger than either of the two parent spectrograms
            >>> def func(spec1, spec2, new_spec):
            ...     m1 = np.max(spec1.image)
            ...     m2 = np.max(spec2.image)
            ...     m = np.max(new_spec.image)
            ...     return m >= 2 * max(m1, m2)
            >>> new_specs = interbreed([spec1], [spec2], num=3, scale_min=0, scale_max=5, validation_function=func)
            >>>
            >>> # plot the first of the new spectrograms
            >>> fig = new_specs[0].plot()
            >>> fig.savefig("ketos/tests/assets/tmp/new_spec_x.png")

            .. image:: ../../../../ketos/tests/assets/tmp/new_spec_x.png
                :width: 300px
                :align: left
    """
    if validation_function is None:
        def always_true(spec1, spec2, new_spec):
            return True
        validation_function = always_true

    N = min(len(specs1), len(specs2))

    specs = list()
    while len(specs) < num:
        
        # randomly sampled scaling factors
        M = num - len(specs)
        t_scale = random_floats(size=M, low=t_scale_min, high=t_scale_max, seed=seed)
        f_scale = random_floats(size=M, low=f_scale_min, high=f_scale_max, seed=seed)
        scale = random_floats(size=M, low=scale_min, high=scale_max, seed=seed)
        seed += 1

        if M == 1:
            t_scale = [t_scale]
            f_scale = [f_scale]
            scale = [scale]

        if shuffle:
            _specs1 = np.random.choice(specs1, N, replace=False)
            _specs2 = np.random.choice(specs2, N, replace=False)
        else:
            _specs1 = specs1[:N]
            _specs2 = specs2[:N]

        for i in range(N):
            
            s1 = _specs1[i]
            s2 = _specs2[i]

            # time offset
            dt = s1.duration() - s2.duration()
            if dt != 0:
                rndm = np.random.random_sample()
                delay = np.abs(dt) * rndm
            else:
                delay = 0

            if dt >= 0:
                spec_long = s1
                spec_short = s2
            else:
                spec_short = s1
                spec_long = s2

            spec = spec_long.copy() # make a copy

            # add the two spectrograms
            spec.add(spec=spec_short, delay=delay, scale=scale[i], make_copy=True,\
                    smooth=smooth, smooth_par=smooth_par, t_scale=t_scale[i], f_scale=f_scale[i])
            
            if validation_function(spec_long, spec_short, spec):
                specs.append(spec)

            if len(specs) >= num:
                break

    # if list has length 1, return the element rather than the list
    if len(specs) == 1:
        specs = specs[0]

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

        self.file_dict, self.file_vector, self.time_vector = self._make_file_and_time_vectors()        

    def _make_file_and_time_vectors(self):
        n = self.image.shape[0]
        time_vector = self.tres * np.arange(n) + self.tmin
        file_vector = np.zeros(n)
        file_dict = {0: self.tag}
        return file_dict, file_vector, time_vector

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
        spec.time_vector = self.time_vector.copy()
        spec.file_vector = self.file_vector.copy()
        spec.file_dict = self.file_dict.copy()
        spec.labels = self.labels.copy()
        spec.boxes = list()
        for b in self.boxes:
            spec.boxes.append(b.copy())

        return spec

    def _make_spec_from_cut(self, tbin1=None, tbin2=None, fbin1=None, fbin2=None, fpad=False, preserve_time=False):
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

        if preserve_time:
            tmin = t1
        else:
            tmin = 0

        # handle annotations
        labels, boxes = self.get_cropped_annotations(t1=t1, t2=t2, f1=f1, f2=f2)
        ann = AnnotationHandler(labels, boxes)
        ann._shift_annotations(delay=tmin)

        # create cropped spectrogram
        spec = Spectrogram(image=img, NFFT=self.NFFT, tres=self.tres, tmin=tmin, fres=self.fres, fmin=fmin, timestamp=self.timestamp, flabels=None, tag='')

        # add annotations
        spec.annotate(labels=labels, boxes=boxes)

        # handle time vector, file vector and file dict
        spec.time_vector, spec.file_vector, spec.file_dict = self._cut_time_and_file_vectors(tbin1, tbin2)

        return spec

    def _cut_time_and_file_vectors(self, tbin1, tbin2):
        if tbin1 is None:
            tbin1 = 0
        if tbin2 is None:
            tbin2 = len(self.time_vector)
        
        time_vector = self.time_vector.copy()
        file_vector = self.file_vector.copy()

        time_vector = time_vector[tbin1:tbin2]
        file_vector = file_vector[tbin1:tbin2]

        file_dict = {}
        new_file_vector = file_vector.copy()
        new_key = 0
        for it in self.file_dict.items():
            key = it[0]
            val = it[1]
            if np.any(file_vector == key):
                file_dict[new_key] = val
                new_file_vector[file_vector == key] = new_key
                new_key += 1

        return time_vector, new_file_vector, file_dict

    def get_time_vector(self):
        return self.time_vector

    def get_file_vector(self):
        return self.file_vector

    def get_file_dict(self):
        return self.file_dict

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

        if tlow != None and tlow > self.tmin:
            t1 = self._find_tbin(tlow, truncate=True)
        if thigh != None and thigh < self.tmin + self.duration():
            t2 = self._find_tbin(thigh, truncate=True)
        if flow != None and flow > self.fmin:
            f1 = self._find_fbin(flow, truncate=True)
        if fhigh != None and fhigh < self.fmax():
            f2 = self._find_fbin(fhigh, truncate=True)
            
        if t2 <= t1 or f2 <= f1:
            img = None
        else:
            img = self.image[t1:t2, f1:f2]

        return img, t1, f1


    def crop(self, tlow=None, thigh=None, flow=None, fhigh=None, preserve_time=False):
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
                preserve_time: bool
                    Keep the existing time axis. If false, the time axis will be shifted so t=0 corresponds to 
                    the first bin of the cropped spectrogram.
        """
        # crop image
        self.image, tbin1, fbin1 = self._crop_image(tlow, thigh, flow, fhigh)

        # update t_min and f_min
        if preserve_time: 
            self.tmin += self.tres * tbin1
        
        self.fmin += self.fres * fbin1

        # crop labels and boxes
        self.labels, self.boxes = self.get_cropped_annotations(t1=tlow, t2=thigh, f1=flow, f2=fhigh)
        
        if self.flabels != None:
            self.flabels = self.flabels[fbin1:fbin1+self.image.shape[1]]

    def extract(self, label, min_length=None, center=False, fpad=False, make_copy=False, preserve_time=False):
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
        # stretch to minimum length, if necessary
        boi = s._stretch(boxes=boi, min_length=min_length, center=center)
        # extract
        res = s._clip(boxes=boi, fpad=fpad, preserve_time=preserve_time)
        # remove extracted labels
        s.delete_annotations(idx)
        
        return res

    def segment(self, number=1, length=None, pad=False, preserve_time=False):
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
        segs = self._clip(boxes=boxes, preserve_time=preserve_time)
        
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
                    If True, box is stretched equally on both sides.
                    If False, the distribution of stretch is random.

            Returns:
                res: list
                    stretchted boxes
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

    def _clip(self, boxes, fpad=False, preserve_time=False):
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
            spec = self._make_spec_from_cut(tbin1=t1[i], tbin2=t2[i], fbin1=f1[i], fbin2=f2[i], fpad=fpad, preserve_time=preserve_time)
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
                    time_vector = self.time_vector[t2max:t1[i]]
                    file_vector = self.file_vector[t2max:t1[i]]
                else:
                    img_c = np.append(img_c, self.image[t2max:t1[i]], axis=0)
                    time_vector = np.append(time_vector, self.time_vector[t2max:t1[i]])
                    file_vector = np.append(file_vector, self.file_vector[t2max:t1[i]])

        self.image = img_c
        self.time_vector = time_vector
        self.file_vector = file_vector
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
            
            >>> from ketos.audio_processing.spectrogram import Spectrogram
            >>> from ketos.audio_processing.audio import AudioSignal
            >>> import matplotlib.pyplot as plt
            >>> # create audio signal
            >>> s = AudioSignal.morlet(rate=1000, frequency=300, width=1)
            >>> # create spectrogram
            >>> spec = MagSpectrogram(s, winlen=0.2, winstep=0.05)
            >>> # show image
            >>> spec.plot()
            <Figure size 640x480 with 2 Axes>
            
            >>> plt.show()
            >>> # apply very small amount (0.01 sec) of horizontal blur
            >>> # and significant amount of vertical blur (30 Hz)  
            >>> spec.blur_gaussian(tsigma=0.01, fsigma=30)
            >>> # show blurred image
            >>> spec.plot()
            <Figure size 640x480 with 2 Axes>

            >>> plt.show()
            
            .. image:: ../../_static/morlet_spectrogram.png
                :width: 300px
                :align: left
            .. image:: ../../_static/morlet_spectrogram_blurred.png
                :width: 300px
                :align: right
        """
        assert tsigma > 0, "tsigma must be strictly positive"

        if fsigma < 0:
            fsigma = 0
        
        sigmaX = tsigma / self.tres
        sigmaY = fsigma / self.fres
        
        self.image = ndimage.gaussian_filter(input=self.image, sigma=(sigmaX,sigmaY))
    
    def add(self, spec, delay=0, scale=1, make_copy=False, smooth=False, smooth_par=5, preserve_time=False, t_scale=1, f_scale=1):
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

        # stretch/squeeze
        sp.scale_time_axis(scale=t_scale, preserve_shape=False)
        sp.scale_freq_axis(scale=f_scale, preserve_shape=True)

        # crop spectrogram
        if delay < 0:
            tlow = sp.tmin - delay
        else:
            tlow = sp.tmin
        thigh = sp.tmin + self.duration() - delay  

        sp.crop(tlow, thigh, self.fmin, self.fmax(), preserve_time=preserve_time)

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

        n = self.image.shape[0]
        self.time_vector = self.tmin + self.tres * np.arange(n)
        self.file_vector = np.zeros(n)
        self.file_dict = {0: 'fake'}

    def scale_time_axis(self, scale, preserve_shape=True):

        flip_pad = False

        if scale == 1:
            return
        
        else:
            n = self.image.shape[0]
            scaled_image = rescale(self.image, (scale, 1), anti_aliasing=True, multichannel=False)
            dn = n - scaled_image.shape[0]

            if not preserve_shape:
                self.image = scaled_image

            else:
                if dn < 0:
                    self.image = scaled_image[:n,:]
                
                elif dn > 0:
                    pad = self.image[n-dn:,:]
                    if flip_pad: 
                        pad = np.flip(pad, axis=0)
                    
                    self.image = np.concatenate((scaled_image, pad), axis=0)

                assert self.image.shape[0] == n, 'Ups. Something went wrong while attempting to rescale the time axis.'                

        # update annotations
        self._scale_annotations(scale)

    def scale_freq_axis(self, scale, preserve_shape=True):

        pad_with_gaussian_noise = False
        flip_pad = False

        if scale == 1:
            return
        
        else:
            n = self.image.shape[1]
            scaled_image = rescale(self.image, (1, scale), anti_aliasing=True, multichannel=False)
            dn = n - scaled_image.shape[1]

            if not preserve_shape:
                self.image = scaled_image

            else:
                if dn < 0:
                    self.image = scaled_image[:,:n]
                
                elif dn > 0:
                    pad = self.image[:,n-dn:]
                    if flip_pad: 
                        pad = np.flip(pad, axis=1)

                    if pad_with_gaussian_noise:
                        mean = np.mean(self.image, axis=1)
                        std = np.std(self.image, axis=1)
                        pad = np.zeros(shape=(self.image.shape[0],dn))
                        for i, (m,s) in enumerate(zip(mean, std)):
                            pad[i,:] = np.random.normal(loc=m, scale=s, size=(1,dn))

                    # pad image
                    self.image = np.concatenate((scaled_image, pad), axis=1)

                assert self.image.shape[1] == n, 'Ups. Something went wrong while attempting to rescale the frequency axis.'                

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

        # add time and file info
        self.time_vector = np.append(self.time_vector, spec.time_vector)

        # join dictionaries
        new_keys = {}
        for it in spec.file_dict.items():
            key = it[0]
            value = it[1]
            if value not in self.file_dict.values():
                n = len(self.file_dict)
                self.file_dict[n] = value
                new_keys[key] = n
            else:
                existing_key = self._get_key(file=value)
                new_keys[key] = existing_key

        # update keys
        file_vec = list()
        for f in spec.file_vector:
            file_vec.append(new_keys[f])

        # join file vectors
        self.file_vector = np.append(self.file_vector, file_vec)

        # append image
        self.image = np.append(self.image, spec.image, axis=0)

    def _get_key(self, file):
        res = None
        for it in self.file_dict.items():
            key = it[0]
            value = it[1]
            if file == value:
                res = key

        return res

    def plot(self, label=None, pred=None, feat=None, conf=None):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display selected label, binary predictions, features, and confidence levels.

            All plotted quantities share the same time axis, and are assumed to span the 
            same period of time as the spectrogram.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                spec: Spectrogram
                    spectrogram to be plotted
                label: int
                    Label of interest
                pred: 1d array
                    Binary prediction for each time bin in the spectrogram
                feat: 2d array
                    Feature vector for each time bin in the spectrogram
                conf: 1d array
                    Confidence level of prediction for each time bin in the spectrogram
            
            Returns:
            fig: matplotlib.figure.Figure
            A figure object.

            Example:
                >>> # extract saved spectrogram from database file
                >>> import tables
                >>> import ketos.data_handling.database_interface as di
                >>> db = tables.open_file("ketos/tests/assets/cod.h5", "r") 
                >>> table = di.open(db, "/sig") 
                >>> spectrogram = di.get_objects(table)[0]
                >>> db.close()
                >>> 
                >>> # plot the spectrogram and label '1'
                >>> import matplotlib.pyplot as plt
                >>> fig = spectrogram.plot(label=1)
                >>> plt.show()
        """
        nrows = 1
        if (label is not None): 
            nrows += 1
        if (pred is not None): 
            nrows += 1
        if (feat is not None): 
            nrows += 1
        if (conf is not None): 
            nrows += 1

        if nrows == 1:
            figsize=(6.4, 4.8)
        else:
            figsize=(8, 1+1.5*nrows)            
        
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True)

        if nrows == 1:
            ax0 = ax
        else:
            ax0 = ax[-1]

        # spectrogram
        x = self.image
        img_plot = ax0.imshow(x.T, aspect='auto', origin='lower', extent=(0, self.duration(), self.fmin, self.fmax()))
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Frequency (Hz)')
        fig.colorbar(img_plot, ax=ax0, format='%+2.0f dB')

        row = -2

        # labels
        if label is not None:
            labels = self.get_label_vector(label)
            n = len(labels)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt
            ax[row].plot(t_axis, labels, color='C1')
            ax[row].set_xlim(0, self.duration())
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('label')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)
            row -= 1

        # predictions
        if pred is not None:
            n = len(pred)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt
            ax[row].plot(t_axis, pred, color='C2')
            ax[row].set_xlim(0, self.duration())
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('prediction')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  
            row -= 1

        # feat
        if feat is not None:
            n = len(feat)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt
            m = np.mean(feat, axis=0)
            idx = np.argwhere(m != 0)
            idx = np.squeeze(idx)
            x = feat[:,idx]
            x = x / np.max(x, axis=0)
            img_plot = ax[row].imshow(x.T, aselft='auto', origin='lower', extent=(0, self.duration(), 0, 1))
            ax[row].set_ylabel('feature #')
            fig.colorbar(img_plot, ax=ax[row])
            row -= 1

        # confidence
        if conf is not None:
            n = len(conf)
            t_axis = np.arange(n, dtype=float)
            dt = self.duration() / n
            t_axis *= dt 
            t_axis += 0.5 * dt
            ax[row].plot(t_axis, conf, color='C3')
            ax[row].set_xlim(0, self.duration())
            ax[row].set_ylim(-0.1, 1.1)
            ax[row].set_ylabel('confidence')
            fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  
            row -= 1

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
                 flabels=None, hamming=True, NFFT=None, compute_phase=False, decibel=False, tag=''):

        super(MagSpectrogram, self).__init__(timestamp=timestamp, flabels=flabels, tag=tag)
        self.image, self.NFFT, self.fres, self.phase_change = self.make_mag_spec(audio_signal, winlen, winstep, hamming, NFFT, timestamp, compute_phase, decibel)
        self.tres = winstep
        self.file_dict, self.file_vector, self.time_vector = self._make_file_and_time_vectors()        

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
            from ketos.pre_processing import to_decibel
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
