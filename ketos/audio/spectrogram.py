# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" 'audio.spectrogram' module within the ketos library.

    This module provides utilities to work with spectrograms.

    Spectrograms are two-dimensional visual representations of 
    sound waves, in which time is shown along the horizontal 
    axis, frequency along the vertical axis, and color is used 
    to indicate the sound amplitude. Read more on Wikipedia:
    https://en.wikipedia.org/wiki/Spectrogram

    The module contains the parent class Spectrogram, and four
    child classes (MagSpectrogram, PowerSpectrogram, MelSpectrogram, 
    CQTSpectrogram), which inherit methods and attributes from the 
    parent class.

    Note, however, that not all methods (e.g. crop) work for all 
    child classes. See the documentation of the individual methods 
    for further details.

    Contents:
        Spectrogram class:
        MagSpectrogram class:
        PowerSpectrogram class:
        MelSpectrogram class:
        CQTSpectrogram class
"""
import os
import copy
import librosa
import warnings
import numpy as np
from scipy.signal import get_window
from scipy.fftpack import dct
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ketos.audio.waveform import Waveform, get_duration, get_sampling_rate, _validate_wf_args
import ketos.audio.utils.misc as aum
from ketos.audio.utils.axis import LinearAxis, Log2Axis, MelAxis
from ketos.audio.annotation import AnnotationHandler
from ketos.audio.utils.filter import enhance_signal, reduce_tonal_noise
from ketos.audio.base_audio import BaseAudioTime, segment_data


def add_specs(a, b, offset=0, scale=1, make_copy=False):
    """ Place two spectrograms on top of one another by adding their 
        pixel values.

        The spectrograms must be of the same type, and share the same 
        time resolution. 
        
        The spectrograms must have consistent frequency axes. 
        For linear frequency axes, this implies having the same 
        resolution; for logarithmic axes with base 2, this implies having 
        the same number of bins per octave minimum values that differ by 
        a factor of :math:`2^{n/m}` where :math:`m` is the number of bins 
        per octave and :math:`n` is any integer. No check is made for the 
        consistency of the frequency axes.

        Note that the attributes filename, offset, and label of spectrogram 
        `b` is being added are lost.

        The sum spectrogram has the same dimensions (time x frequency) as 
        spectrogram `a`.

        Args:
            a: Spectrogram
                Spectrogram
            b: Spectrogram
                Spectrogram to be added
            offset: float
                Shift spectrogram `b` by this many seconds relative to spectrogram `a`.
            scale: float
                Scaling factor applied to signal that is added
            make_copy: bool
                Make copies of both spectrograms, leaving the orignal instances 
                unchanged by the addition operation.

        Returns:
            ab: Spectrogram
                Sum spectrogram
    """
    assert a.type == b.type, "It is not possible to add spectrograms with different types"
    assert a.time_res() == b.time_res(), 'It is not possible to add spectrograms with different time resolutions'

    # make copy
    if make_copy:
        ab = a.deepcopy()
    else:
        ab = a

    # compute cropping boundaries for time axis
    end = a.duration() - offset

    # determine position of b within a
    pos_x = a.time_ax.bin(offset, truncate=True) #lower left corner time bin
    pos_y = a.freq_ax.bin(b.freq_min(), truncate=True) #lower left corner frequency bin

    # crop spectrogram b
    b = b.crop(start=-offset, end=end, freq_min=a.freq_min(), freq_max=a.freq_max(), make_copy=make_copy)

    # add the two images
    bins_x = b.data.shape[0]
    bins_y = b.data.shape[1]
    ab.data[pos_x:pos_x+bins_x, pos_y:pos_y+bins_y] += scale * b.data

    return ab

def load_audio_for_spec(path, channel, rate, window, step, offset, duration, 
    resample_method, id=None, normalize_wav=False, waveform_transforms=None, 
    smooth=0.01):
    """ Load audio data from a wav file for the specific purpose of computing 
        the spectrogram.

        The loaded audio covers a time interval that extends slightly beyond 
        that specified, [offset, offset+duration], as needed to compute the 
        full spectrogram without padding with zeros at either end. 

        Moreover, the returned instance has two extra class attributes 
        not usually associated with instances of the Waveform class,

            * stft_args: dict
                Parameters to be used for the computation of the 
                Short-Time Fourier transform

            * len_extend: tuple(int,int) 
                Length (no. samples) by which the time interval has been 
                extended at both ends (left, right).
        
        Returns None if the requested data segment is empty.

        Args:
            path: str
                Path to wav file
            channel: int
                Channel to read from. Only relevant for stereo recordings
            rate: float
                Desired sampling rate in Hz. If None, the original sampling rate will be used
            window: float
                Window size in seconds that will be used for computing the spectrogram
            step: float
                Step size in seconds that will be used for computing the spectrogram
            offset: float
                Start time of spectrogram in seconds, relative the start of the wav file.
            duration: float
                Length of spectrogrma in seconds.
            resample_method: str
                Resampling method. Only relevant if `rate` is specified. Options are
                    * kaiser_best
                    * kaiser_fast
                    * scipy (default)
                    * polyphase

                See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                for details on the individual methods.
            id: str
                Unique identifier (optional). If None, the filename will be used.
            normalize_wav: bool
                Normalize the waveform to have a mean of zero (mean=0) and a standard 
                deviation of unity (std=1). Default is False.
            smooth: float
                Width in seconds of the smoothing region used for stitching together audio files.

        Returns:
            audio: Waveform
                The audio signal
    """
    path, offset, duration = _validate_wf_args(path, offset, duration)

    # make copies so we don't change the input arguments
    offset_ext = offset.copy()
    duration_ext = duration.copy()

    if rate is None:
        rate = get_sampling_rate(path=path)

    duration_ext = get_duration(path=path, offset=offset, duration=duration_ext)
    total_duration = np.sum(duration_ext)

    if total_duration <= 0:
        return None

    nominal_offset = offset[0]

    # compute the arguments for the short-time fourier transform
    stft_args = aum.segment_args(rate=rate, offset=nominal_offset, window=window, step=step, duration=total_duration)

    # modify offset and duration to extend audio segment at both ends
    offset_ext[0] = stft_args['offset_len'] / rate
    left_ext = nominal_offset - offset_ext[0]
    total_duration_ext = int(stft_args['num_segs'] * stft_args['step_len'] + stft_args['win_len']) / rate
    right_ext = total_duration_ext - total_duration - left_ext
    duration_ext[0]  += left_ext
    duration_ext[-1] += right_ext

    # now load extended audio with from_wav method
    audio = Waveform.from_wav(path=path, rate=rate, channel=channel,
        offset=offset_ext, duration=duration_ext, resample_method=resample_method, 
        id=id, normalize_wav=normalize_wav, transforms=waveform_transforms,
        smooth=smooth)

    if len(audio.get_data()) == 0:
        return None, None

    # make sure we don't pad twice
    stft_args["offset_len"] = 0

    # use the correct offset value
    audio.offset = nominal_offset

    # create extra class attributes
    audio.stft_args = stft_args
    n_left_ext = aum.num_samples(left_ext, audio.rate)
    n_right_ext = aum.num_samples(right_ext, audio.rate)
    audio.len_extend = (n_left_ext, n_right_ext)

    return audio


class Spectrogram(BaseAudioTime):
    """ Spectrogram.

        Parent class for MagSpectrogram, PowerSpectrogram, MelSpectrogram, 
        and CQTSpectrogram.

        The Spectrogram class stores the spectrogram pixel values in a 
        numpy array, where the first axis (0) is the time dimension and 
        the second axis (1) is the frequency dimensions.

        Args:
            data: numpy array
                Spectrogram matrix. 
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            type: str
                Spectrogram type. Options include,
                    * 'Mag': Magnitude spectrogram
                    * 'Pow': Power spectrogram
                    * 'Mel': Mel spectrogram
                    * 'CQT': CQT spectrogram

            freq_ax: LinearAxis or Log2Axis
                Axis object for the frequency dimension
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to the spectrogram. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram
            
        Attributes:
            data: numpy array
                Spectrogram matrix. 
            time_ax: LinearAxis
                Axis object for the time dimension
            freq_ax: LinearAxis or Log2Axis
                Axis object for the frequency dimension
            type: str
                Spectrogram type. Options include,
                    * 'Mag': Magnitude spectrogram
                    * 'Pow': Power spectrogram
                    * 'Mel': Mel spectrogram
                    * 'CQT': CQT spectrogram

            filename: str or list(str)
                Name of the source audio file.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file.
            label: int
                Spectrogram label.
            annot: AnnotationHandler
                AnnotationHandler object.
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram
"""
    def __init__(self, data, time_res, type, freq_ax, filename=None, offset=0, label=None, 
        annot=None, transforms=None, transform_log=None, waveform_transform_log=None, **kwargs):

        super().__init__(data=data, time_res=time_res, filename=filename, offset=offset, label=label, 
            annot=annot, transform_log=transform_log, **kwargs)

        if waveform_transform_log is None: waveform_transform_log = []

        self.freq_ax = freq_ax
        self.type = type

        self.allowed_transforms.update({'blur': self.blur, 
                                        'enhance_signal': self.enhance_signal,
                                        'reduce_tonal_noise': self.reduce_tonal_noise,
                                        'resize': self.resize})
        
        self.apply_transforms(transforms)

        self.waveform_transform_log = waveform_transform_log

    @classmethod
    def infer_shape(cls, **kwargs):
        """ Infers the spectrogram shape that would result if the class were 
            instantiated with a specific set of parameter values.
            Returns a None value if the shape could not be inferred.
            Accepts the same list of arguments as the `from_wav` method, 
            which is implemented in the child classes.

            Note: The current implementation involves computing a dummy spectrogram.
            Therefore, if this method is called repeatedly the computational overhead 
            can become substantial.

            Returns:
                : tuple
                    Inferred shape. If the parameter value do not allow 
                    the shape be inferred, a None value is returned.
        """
        if 'duration' in kwargs.keys() and 'rate' in kwargs.keys() and hasattr(cls, 'from_waveform'):
            sr = kwargs['rate']
            num_samples = int(kwargs['duration'] * sr)
            y = np.zeros(num_samples)
            wf = Waveform(data=y, rate=sr)
            kwargs.pop('rate', None)
            kwargs.pop('duration', None)
            x = cls.from_waveform(wf, **kwargs)
            return x.get_data().shape
        else:
            return None

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'type':self.type, 'waveform_transform_log': self.waveform_transform_log})
        return attrs

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = super().get_kwargs()
        kwargs.update({'freq_ax': self.freq_ax})
        return kwargs

    def freq_min(self):
        """ Get spectrogram minimum frequency in Hz.

            Returns:
                : float
                    Frequency in Hz
        """
        return self.freq_ax.min()

    def freq_max(self):
        """ Get spectrogram maximum frequency in Hz.

            Returns:
                : float
                    Frequency in Hz
        """
        return self.freq_ax.max()

    def crop(self, start=None, end=None, length=None,\
        freq_min=None, freq_max=None, height=None, make_copy=False):
        """ Crop spectogram along time axis, frequency axis, or both.
            
            Args:
                start: float
                    Start time in seconds, measured from the left edge of spectrogram.
                end: float
                    End time in seconds, measured from the left edge of spectrogram.
                length: int
                    Horizontal size of the cropped image (number of pixels). If provided, 
                    the `end` argument is ignored. 
                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                height: int
                    Vertical size of the cropped image (number of pixels). If provided, 
                    the `freq_max` argument is ignored. 
                make_copy: bool
                    Return a cropped copy of the spectrogra. Leaves the present instance 
                    unaffected. Default is False.

            Returns:
                spec: Spectrogram
                    Cropped spectrogram

            Examples: 
                >>> import numpy as np
                >>> import matplotlib.pyplot as plt
                >>> from ketos.audio.spectrogram import Spectrogram
                >>> from ketos.audio.utils.axis import LinearAxis
                >>> # Create a spectrogram with shape (20,30), time resolution of 
                >>> # 0.5 s, random pixel values, and a linear frequency axis from 
                >>> # 0 to 300 Hz,
                >>> ax = LinearAxis(bins=30, extent=(0.,300.), label='Frequency (Hz)')
                >>> img = np.random.rand(20,30)
                >>> spec = Spectrogram(data=img, time_res=0.5, type='Mag', freq_ax=ax)
                >>> # Draw the spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_orig.png")
                >>> plt.close(fig)
                
                .. image:: ../../../ketos/tests/assets/tmp/spec_orig.png

                >>> # Crop the spectrogram along time axis
                >>> spec1 = spec.crop(start=2.0, end=4.2, make_copy=True)
                >>> # Draw the spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_cropped.png")
                >>> plt.close(fig)
                
                .. image:: ../../../ketos/tests/assets/tmp/spec_cropped.png
        """
        spec = super().crop(start=start, end=end, length=length, make_copy=make_copy) #crop time axis

        # crop frequency axis
        b1, b2 = self.freq_ax.cut(x_min=freq_min, x_max=freq_max, bins=height)

        # add frequency information to log
        if not make_copy:
            self.transform_log[-1]['freq_min'] = freq_min
            self.transform_log[-1]['freq_max'] = freq_max

        # crop image
        spec.data = spec.data[:, b1:b2+1]

        # crop annotations, if any
        if self.annot: self.annot.crop(freq_min=freq_min, freq_max=freq_max)

        return spec
                
    def add(self, spec, offset=0, scale=1, make_copy=False):
        """ Add another spectrogram on top of this spectrogram.

            The spectrograms must be of the same type, and share the same 
            time resolution. 
            
            The spectrograms must have consistent frequency axes. 
            For linear frequency axes, this implies having the same 
            resolution; for logarithmic axes with base 2, this implies having 
            the same number of bins per octave minimum values that differ by 
            a factor of :math:`2^{n/m}` where :math:`m` is the number of bins 
            per octave and :math:`n` is any integer. No check is made for the 
            consistency of the frequency axes.

            Note that the attributes filename, offset, and label of the spectrogram 
            that is being added are lost.

            The sum spectrogram has the same dimensions (time x frequency) as 
            the original spectrogram.

            Args:
                spec: Spectrogram
                    Spectrogram to be added
                offset: float
                    Shift the spectrograms that is being added by this many seconds 
                    relative to the original spectrogram.
                scale: float
                    Scaling factor applied to spectrogram that is added
                make_copy: bool
                    Make copies of both spectrograms so as to leave the original 
                    instances unchanged.

            Returns:
                : Spectrogram
                    Sum spectrogram
        """
        return add_specs(a=self, b=spec, offset=offset, scale=scale, make_copy=make_copy)

    def blur(self, sigma_time, sigma_freq=0):
        """ Blur the spectrogram using a Gaussian filter.

            Note that the spectrogram frequency axis must be linear if sigma_freq > 0.

            This uses the Gaussian filter method from the scipy.ndimage package:
            
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

            Args:
                sigma_time: float
                    Gaussian kernel standard deviation along time axis in seconds. 
                    Must be strictly positive.
                sigma_freq: float
                    Gaussian kernel standard deviation along frequency axis in Hz.

            Example:        
                >>> from ketos.audio.spectrogram import Spectrogram
                >>> from ketos.audio.waveform import Waveform
                >>> import matplotlib.pyplot as plt
                >>> # create audio signal
                >>> s = Waveform.morlet(rate=1000, frequency=300, width=1)
                >>> # create spectrogram
                >>> spec = MagSpectrogram.from_waveform(s, window=0.2, step=0.05)
                >>> # show image
                >>> fig = spec.plot()
                >>> plt.close(fig)
                >>> # apply very small amount (0.01 sec) of horizontal blur
                >>> # and significant amount of vertical blur (30 Hz)  
                >>> spec.blur(sigma_time=0.01, sigma_freq=30)
                >>> # show blurred image
                >>> fig = spec.plot()
                >>> plt.close(fig)
                
                .. image:: ../_static/morlet_spectrogram.png

                .. image:: ../_static/morlet_spectrogram_blurred.png
        """
        assert sigma_time > 0, "sigma_time must be strictly positive"
        sig_t = sigma_time / self.time_res()

        if sigma_freq > 0:
            assert isinstance(self.freq_ax, LinearAxis), "Frequency axis must be linear when sigma_freq > 0"
            sig_f = sigma_freq / self.freq_ax.bin_width()
        else:
            sig_f = 0

        self.data = ndimage.gaussian_filter(input=self.data, sigma=(sig_t, sig_f))
        self.transform_log.append({'name':'blur', 'sigma_time':sigma_time, 'sigma_freq':sigma_freq})

    def enhance_signal(self, enhancement=1.):
        """ Enhance the contrast between regions of high and low intensity.

            See :func:`audio.image.enhance_image` for implementation details.

            Args:
                enhancement: float
                    Parameter determining the amount of enhancement.
        """
        self.data = enhance_signal(self.data, enhancement=enhancement)
        self.transform_log.append({'name':'enhance_signal', 'enhancement':enhancement})

    def reduce_tonal_noise(self, method='MEDIAN', **kwargs):
        """ Reduce continuous tonal noise produced by e.g. ships and slowly varying 
            background noise

            See :func:`audio.image.reduce_tonal_noise` for implementation details.

            Currently, offers the following two methods:

                1. MEDIAN: Subtracts from each row the median value of that row.
                
                2. RUNNING_MEAN: Subtracts from each row the running mean of that row.
                
            The running mean is computed according to the formula given in 
            Baumgartner & Mussoline, JASA 129, 2889 (2011); doi: 10.1121/1.3562166

            Args:
                method: str
                    Options are 'MEDIAN' and 'RUNNING_MEAN'
            
            Optional args:
                time_constant: float
                    Time constant in seconds, used for the computation of the running mean.
                    Must be provided if the method 'RUNNING_MEAN' is chosen.

            Example:
                >>> # read audio file
                >>> from ketos.audio.waveform import Waveform
                >>> aud = Waveform.from_wav('ketos/tests/assets/grunt1.wav')
                >>> # compute the spectrogram
                >>> from ketos.audio.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram.from_waveform(aud, window=0.2, step=0.02)
                >>> # keep only frequencies below 800 Hz
                >>> spec = spec.crop(freq_max=800)
                >>> # show spectrogram as is
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_before_tonal.png")
                >>> plt.close(fig)
                >>> # tonal noise reduction
                >>> spec.reduce_tonal_noise()
                >>> # show modified spectrogram
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_after_tonal.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/spec_before_tonal.png

                .. image:: ../../../ketos/tests/assets/tmp/spec_after_tonal.png

        """
        if 'time_constant' in kwargs.keys():
            time_const_len = kwargs['time_constant'] / self.time_ax.bin_width()
        else:
            time_const_len = None

        self.data = reduce_tonal_noise(self.data, method=method, time_const_len=time_const_len)

        transf = {'name':'reduce_tonal_noise', 'method':method}
        if 'time_constant' in kwargs.keys(): transf.update({'time_constant': kwargs['time_constant']})
        self.transform_log.append(transf)

    def resize(self, shape=None, time_res=None, **kwargs):
        """ Resize the spectrogram.

            The resizing operation can be controlled either by specifying the 
            shape of the resized spectrogram or by specifying the desired time 
            resolution. In the latter case, the spectrogram is only resized along the time axis.

            The resizing operation is performed using the `resize` method of the 
            scikit-image package, which interpolates the pixel values:

                https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

            Use keyword arguments to control the behavior of scikit-image's resize 
            operation.

            Args:
                shape: tuple(int,int)
                    Shape of the resized spectrogram
                time_res: float
                    Time resolution of the resized spectrogram in seconds. Note that the actual time 
                    resolution of the resized spectrogram may differ slightly from that specified 
                    via the time_res argument, as required to produce an image with an integer number 
                    of time bins.

            Returns: 
                None

            Example:
                >>> from ketos.audio.spectrogram import MagSpectrogram
                >>> # load spectrogram
                >>> spec = MagSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window=0.2, step=0.02)
                >>> # add an annotation
                >>> spec.annotate(start=1.1, end=1.6, freq_min=70, freq_max=600, label=1)
                >>> # keep only frequencies below 800 Hz
                >>> spec = spec.crop(freq_max=800)
                >>> # make a copy of the current spectrogram, then reduce time resolution by a factor of eight
                >>> spec_orig = spec.deepcopy()
                >>> new_time_res = 8.0 * spec.time_res()
                >>> spec.resize(time_res=new_time_res)
                >>> # show spectrograms
                >>> fig = spec_orig.plot(show_annot=True)
                >>> fig.savefig("ketos/tests/assets/tmp/spec_w_annot_box.png")
                >>> plt.close(fig)
                >>> fig = spec.plot(show_annot=True)
                >>> fig.savefig("ketos/tests/assets/tmp/spec_w_annot_box_reduced_resolution.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/spec_w_annot_box.png

                .. image:: ../../../ketos/tests/assets/tmp/spec_w_annot_box_reduced_resolution.png
        """
        assert shape is not None or time_res is not None, "either shape or time_res must be specified"

        transf = {'name':'resize'} #log transform attributes

        # deduce new shape from time_res argument
        if shape is None:
            n_bins = int(self.time_res() / time_res * self.data.shape[0])
            shape = (n_bins, self.data.shape[1])
            transf.update({'time_res': time_res})
        else:
            transf.update({'shape': shape})

        if np.ndim(self.data) == 3:
            shape = (shape[0], shape[1], self.data.shape[2])

        # resize time axis
        if shape[0] != self.data.shape[0]:
            self.time_ax.resize(bins=shape[0])

        # resize frequency axis
        if shape[1] != self.data.shape[1]:
            self.freq_ax.resize(bins=shape[1])

        # resize data array
        self.data = resize(self.data, output_shape=shape, **kwargs)

        transf.update(kwargs)
        self.transform_log.append(transf)

    def plot(self, show_annot=False, figsize=(5,4), cmap='viridis', label_in_title=True, vmin=None, vmax=None):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the spectrogram.

            The colormaps available can be seen here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                show_annot: bool
                    Display annotations
                figsize: tuple
                    Figure size
                cmap: string
                    The colormap to be used
                label_in_title: bool
                    Include label (if available) in figure title
                vmin, vmax : scalar, optional
                    When using scalar data and no explicit norm, vmin and vmax define the data range that the colormap covers. 
                    By default, the colormap covers the complete value range of the supplied data. 
                    vmin, vmax are ignored if the norm parameter is used.            
            Returns:
                fig: matplotlib.figure.Figure
                A figure object.

            Example:
                >>> from ketos.audio.spectrogram import MagSpectrogram
                >>> # load spectrogram
                >>> spec = MagSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window=0.2, step=0.02)
                >>> # add an annotation
                >>> spec.annotate(start=1.1, end=1.6, freq_min=70, freq_max=600, label=1)
                >>> # keep only frequencies below 800 Hz
                >>> spec = spec.crop(freq_max=800)
                >>> # show spectrogram with annotation box
                >>> fig = spec.plot(show_annot=True)
                >>> fig.savefig("ketos/tests/assets/tmp/spec_w_annot_box.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/spec_w_annot_box.png
        """
        fig, ax = super().plot(figsize, label_in_title)

        x = self.get_data() # select image data        
        extent = (0., self.duration(), self.freq_min(), self.freq_max()) # axes ranges        
        img = ax.imshow(x.T, aspect='auto', origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)# draw image
        ax.set_ylabel(self.freq_ax.label) # axis label        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')# colobar

        # superimpose annotation boxes
        if show_annot: self._draw_annot_boxes(ax)
            
        #fig.tight_layout()
        return fig

    def _draw_annot_boxes(self, ax):
        """Draws annotations boxes on top of the spectrogram

            Args:
                ax: matplotlib.axes.Axes
                    Axes object
        """
        annots = self.get_annotations()
        if annots is None: return
        y1 = self.freq_min()
        y2 = self.freq_max()
        for idx,annot in annots.iterrows():
            x1 = annot['start']
            x2 = annot['end']
            if not np.isnan(annot['freq_min']): y1 = annot['freq_min']
            if not np.isnan(annot['freq_max']): y2 = annot['freq_max']
            box = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='C1',facecolor='none')
            ax.add_patch(box)
            ax.text(x1, y2, int(annot['label']), ha='left', va='bottom', color='C1')


class MagSpectrogram(Spectrogram):
    """ Magnitude Spectrogram.
    
        While the underlying data array can be accessed via the :attr:`data` attribute,
        it is recommended to always use the :func:`get_data` function to access the data 
        array, i.e., 

        >>> from ketos.audio.base_audio import BaseAudio
        >>> x = np.ones(6)
        >>> audio_sample = BaseAudio(data=x)
        >>> audio_sample.get_data()
        array([1., 1., 1., 1., 1., 1.])

        Args:
            data: numpy array
                Magnitude spectrogram.
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            freq_min: float
                Lower value of the frequency axis in Hz
            freq_res: float
                Frequency resolution in Hz (corresponds to the bin size used on the frequency axis)
            window_func: str
                Window function used for computing the spectrogram
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to the spectrogram. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram
            phase_angle: numpy.array
                Complex phase angle.

        Attrs:
            data: numpy array
                If the phase angle matrix is not provided, data will be a 2d numpy 
                array containing the magnitude spectrogram.
                On the other hand, if the phase angle matrix is provided, data will 
                be a 3d numpy array where data[:,:,0] contains the magnitude spectrogram 
                and data[:,:,1] contains the complex phase angle. 
            window_func: str
                Window function.
    """
    def __init__(self, data, time_res, freq_min, freq_res, window_func=None, 
        filename=None, offset=0, label=None, annot=None, transforms=None, 
        transform_log=None, waveform_transform_log=None, phase_angle=None, **kwargs):

        # create frequency axis
        freq_bins = max(1, data.shape[1])
        freq_max  = freq_min + data.shape[1] * freq_res
        ax = LinearAxis(bins=freq_bins, extent=(freq_min, freq_max), label='Frequency (Hz)')

        if phase_angle is not None:
            assert phase_angle.shape == data.shape, 'phase_angle and data array must have same shape'
            data = np.stack([data, phase_angle], axis=2)

        # create spectrogram
        kwargs.pop('type', None)
        super().__init__(data=data, time_res=time_res, type=self.__class__.__name__, freq_ax=ax,
            filename=filename, offset=offset, label=label, annot=annot, transforms=transforms, 
            transform_log=transform_log, waveform_transform_log=waveform_transform_log, **kwargs)

        self.window_func = window_func

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'freq_min':self.freq_min(), 'freq_res':self.freq_res(), 'window_func':self.window_func})
        return attrs

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = super().get_kwargs()
        kwargs.pop('freq_ax', None)
        return kwargs

    def get_data(self):
        """ Get magnitude spectrogram data """
        if np.ndim(self.data) == 3: return self.data[:,:,0]
        else: return super().get_data()

    def get_phase_angle(self):
        """ Get magnitude spectrogram complex phase angle, if available """
        if np.ndim(self.data) == 3: return self.data[:,:,1]
        else: return None

    @classmethod
    def empty(cls):
        """ Creates an empty MagSpectrogram object
        """
        return cls(data=np.empty(shape=(0,0), dtype=np.float64), time_res=0, freq_min=0, freq_res=0)

    @classmethod
    def from_waveform(cls, audio, window=None, step=None, seg_args=None, window_func='hamming', 
        freq_min=None, freq_max=None, transforms=None, compute_phase=False, decibel=True, **kwargs):
        """ Create a Magnitude Spectrogram from an :class:`audio_signal.Waveform` by 
            computing the Short Time Fourier Transform (STFT).
        
            Args:
                audio: Waveform
                    Audio signal 
                window: float
                    Window length in seconds
                step: float
                    Step size in seconds
                seg_args: dict
                    Input arguments used for evaluating :func:`audio.audio.segment_args`. 
                    Optional. If specified, the arguments `window` and `step` are ignored.
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                compute_phase: bool
                    Compute complex phase angle. Default it False
                decibel: bool
                    Convert to dB scale

            Returns:
                spec: MagSpectrogram
                    Magnitude spectrogram
        """
        if window_func is not None: window_func = window_func.lower() #make lowercase

        # compute STFT
        img, freq_nyquist, num_fft, seg_args, phase = aum.stft(x=audio.data, rate=audio.rate, window=window,
            step=step, seg_args=seg_args, window_func=window_func, compute_phase=compute_phase, decibel=decibel)

        time_res = seg_args['step_len'] / audio.rate
        freq_res = freq_nyquist / img.shape[1]

        spec = cls(data=img, time_res=time_res, freq_min=0, freq_res=freq_res, window_func=window_func, 
            filename=audio.filename, offset=audio.offset, label=audio.label, annot=audio.annot, 
            waveform_transform_log=audio.transform_log, transforms=transforms, phase_angle=phase, **kwargs)

        if freq_min is not None or freq_max is not None:
            spec = spec.crop(freq_min=freq_min, freq_max=freq_max)

        return spec

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None,
            window_func='hamming', offset=0, duration=None,
            resample_method='scipy', freq_min=None, freq_max=None,
            id=None, normalize_wav=False, transforms=None, 
            waveform_transforms=None, compute_phase=False, 
            decibel=True, smooth=0.01, **kwargs):
        """ Create magnitude spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogram in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase

                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                waveform_transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the waveform before generating 
                    the spectrogram. For example,
                    {"name":"add_gaussian_noise", "sigma":0.5}
                compute_phase: bool
                    Compute complex phase angle. Default it False
                decibel: bool
                    Convert to dB scale
                smooth: float
                    Width in seconds of the smoothing region used for stitching together audio files.

            Returns:
                : MagSpectrogram
                    Magnitude spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window=0.2, step=0.01)
                >>> # crop frequency
                >>> spec = spec.crop(freq_min=50, freq_max=800)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/spec_grunt1.png
        """
        # load audio
        audio = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,
            offset=offset, duration=duration, resample_method=resample_method, id=id, normalize_wav=normalize_wav,
            waveform_transforms=waveform_transforms, smooth=smooth)

        if audio is None:
            warnings.warn("Empty spectrogram returned", RuntimeWarning)
            return cls.empty()

        # compute spectrogram
        return cls.from_waveform(audio=audio, seg_args=audio.stft_args, window_func=window_func, 
            freq_min=freq_min, freq_max=freq_max, transforms=transforms, compute_phase=compute_phase, 
            decibel=decibel, **kwargs)

    def freq_res(self):
        """ Get frequency resolution in Hz.

            Returns:
                : float
                    Frequency resolution in Hz
        """
        return self.freq_ax.bin_width()

    def recover_waveform(self, num_iters=25, phase_angle=None, subtract=0):
        """ Estimate audio signal from magnitude spectrogram.

            Uses :func:`audio.audio.spec2wave`.

            Args:
                num_iters: 
                    Number of iterations to perform.
                phase_angle: 
                    Initial condition for phase in radians. If not specified, 
                    the phase angle computed computed at initialization will 
                    be used, if available. If not available, the phase angle 
                    will default to zero and a warning will be printed.

            Returns:
                : Waveform
                    Audio signal
        """
        mag = self.get_data()

        if phase_angle is None:
            phase_angle = self.get_phase_angle()
            if phase_angle is None:
                phase_angle = 0
                print('Warning: spectrogram phase angle not available; phase will be set to zero everywhere')

        # if the frequency axis has been cropped, pad with zeros to ensure that 
        # the spectrogram has the expected shape
        pad_low  = max(0, int(self.freq_min() / self.freq_res()))
        if pad_low > 0:
            mag = np.pad(mag, pad_width=((0,0),(pad_low,0)), mode='constant')
            if np.ndim(phase_angle) == 2:
                phase_angle = np.pad(phase_angle, pad_width=((0,0),(pad_low,0)), mode='constant')

        #use linear scale
        mag = aum.from_decibel(mag) - subtract

        target_rate = self.freq_ax.bin_width() * 2 * mag.shape[1]

        # retrieve settings used for computing STFT
        num_fft = 2 * (mag.shape[1] - 1)
        step_len = int(target_rate * self.time_res())  #self.seg_args['step_len']
        if self.window_func:
            window_func = get_window(self.window_func, num_fft)
        else:
            window_func = np.ones(num_fft)

        # iteratively estimate audio signal
        audio = aum.spec2wave(image=mag, phase_angle=phase_angle, num_fft=num_fft,\
            step_len=step_len, num_iters=num_iters, window_func=window_func)

        # sampling rate of recovered audio signal
        rate = len(audio) / (self.duration() + (num_fft - step_len) / target_rate)

        # crop at both ends to obtain correct length for waveform
        num_samples = int(self.duration() * rate)
        num_cut = int(0.5 * (num_fft - step_len))
        audio = audio[num_cut:num_cut+num_samples]
        
        return Waveform(rate=rate, data=audio)

    def plot_phase_angle(self, figsize=(5,4), cmap='viridis'):
        """ Plot the complex phase matrix.

            Returns None if the complex phase has not been computed.
            
            Set compute_phase=True when you initialize the spectrogram to ensure 
            that the phase is computed.

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                figsize: tuple
                    Figure size
                cmap: string
                    The colormap to be used. The colormaps available can be 
                    seen here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        fig, ax = super(Spectrogram, self).plot(figsize)

        x = self.get_phase_angle() # select image data  
        if x is None: 
            warnings.warn(f"The complex phase angle has not been computed and can therefore not be plotted. "\
                "Make sure to initialize the spectrogram with compute_phase=True to be able to plot the phase.", category=UserWarning)
            return None      

        extent = (0., self.duration(), self.freq_min(), self.freq_max()) # axes ranges        
        img = ax.imshow(x.T, aspect='auto', origin='lower', cmap=cmap, extent=extent)# draw image
        ax.set_ylabel(self.freq_ax.label) # axis label        
        fig.colorbar(img, ax=ax)# colobar
            
        return fig


class PowerSpectrogram(Spectrogram):
    """ Power Spectrogram.
    
        Args:
            data: 2d or 3d numpy array
                Spectrogram pixel values. 
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            freq_min: float
                Lower value of the frequency axis in Hz
            freq_res: float
                Frequency resolution in Hz (corresponds to the bin size used on the frequency axis)
            window_func: str
                Window function used for computing the spectrogram
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to the spectrogram. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram

        Attrs:
            window_func: str
                Window function.
    """
    def __init__(self, data, time_res, freq_min, freq_res, window_func=None, 
        filename=None, offset=0, label=None, annot=None, transforms=None, 
        transform_log=None, waveform_transform_log=None, **kwargs):

        # create frequency axis
        freq_bins = data.shape[1]
        freq_max  = freq_min + freq_bins * freq_res
        ax = LinearAxis(bins=freq_bins, extent=(freq_min, freq_max), label='Frequency (Hz)')

        # create spectrogram
        kwargs.pop('type', None)
        super().__init__(data=data, time_res=time_res, type=self.__class__.__name__, freq_ax=ax,
            filename=filename, offset=offset, label=label, annot=annot, transforms=transforms, 
            transform_log=transform_log, waveform_transform_log=waveform_transform_log, **kwargs)

        self.window_func = window_func

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'freq_min':self.freq_min(), 'freq_res':self.freq_res(), 'window_func':self.window_func})
        return attrs

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = super().get_kwargs()
        kwargs.pop('freq_ax', None)
        return kwargs

    @classmethod
    def empty(cls):
        """ Creates an empty PowerSpectrogram object
        """
        return cls(data=np.empty(shape=(0,0), dtype=np.float64), time_res=0, freq_min=0, freq_res=0)

    @classmethod
    def from_waveform(cls, audio, window=None, step=None, seg_args=None, window_func='hamming', 
        freq_min=None, freq_max=None, transforms=None, decibel=True, **kwargs):
        """ Create a Power Spectrogram from an :class:`audio_signal.Waveform` by 
            computing the Short Time Fourier Transform (STFT).
        
            Args:
                audio: Waveform
                    Audio signal 
                window: float
                    Window length in seconds
                step: float
                    Step size in seconds
                seg_args: dict
                    Input arguments used for evaluating :func:`audio.audio.segment_args`. 
                    Optional. If specified, the arguments `window` and `step` are ignored.
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                decibel: bool
                    Convert to dB scale

            Returns:
                : MagSpectrogram
                    Magnitude spectrogram
        """
        if window_func is not None: window_func = window_func.lower() #make lowercase

        # compute STFT
        img, freq_nyquist, num_fft, seg_args, phase = aum.stft(x=audio.data, rate=audio.rate, window=window,\
            step=step, seg_args=seg_args, window_func=window_func, decibel=False)
        img = aum.mag2pow(img, num_fft) # Magnitude->Power conversion
        if decibel:
            img = aum.to_decibel(img) # convert to dB

        time_res = seg_args['step_len'] / audio.rate
        freq_res = freq_nyquist / img.shape[1]

        spec = cls(data=img, time_res=time_res, freq_min=0, freq_res=freq_res, window_func=window_func, 
            filename=audio.filename, offset=audio.offset, label=audio.label, annot=audio.annot, 
            waveform_transform_log=audio.transform_log, transforms=transforms, **kwargs)

        if freq_min is not None or freq_max is not None:
            spec = spec.crop(freq_min=freq_min, freq_max=freq_max)

        return spec

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None,
            window_func='hamming', offset=0, duration=None,
            resample_method='scipy', freq_min=None, freq_max=None,
            id=None, normalize_wav=False, transforms=None, waveform_transforms=None, 
            decibel=True, smooth=0.01, **kwargs):            
        """ Create power spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase

                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                freq_min: float
                    Lower frequency in Hz.
                freq_max: str or float
                    Upper frequency in Hz.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                waveform_transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the waveform before generating 
                    the spectrogram. For example,
                    {"name":"add_gaussian_noise", "sigma":0.5}
                decibel: bool
                    Convert to dB scale
                smooth: float
                    Width in seconds of the smoothing region used for stitching together audio files.

            Returns:
                spec: MagSpectrogram
                    Magnitude spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio.spectrogram import MagSpectrogram
                >>> spec = MagSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window=0.2, step=0.01)
                >>> # crop frequency
                >>> spec = spec.crop(freq_min=50, freq_max=800)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/spec_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/spec_grunt1.png
        """
        # load audio
        audio = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,
            offset=offset, duration=duration, resample_method=resample_method, id=id, normalize_wav=normalize_wav,
            waveform_transforms=waveform_transforms)

        if audio is None:
            warnings.warn("Empty spectrogram returned", RuntimeWarning)
            return cls.empty()

        # compute spectrogram
        return cls.from_waveform(audio=audio, seg_args=audio.stft_args, window_func=window_func, 
            freq_min=freq_min, freq_max=freq_max, transforms=transforms, decibel=decibel, **kwargs)

    def freq_res(self):
        """ Get frequency resolution in Hz.

            Returns:
                : float
                    Frequency resolution in Hz
        """
        return self.freq_ax.bin_width()


class MelSpectrogram(Spectrogram):
    """ Mel Spectrogram.
    
        Args:
            data: 2d numpy array
                Mel spectrogram pixel values. 
            num_filters: int
                The number of filters in the filter bank.
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            freq_max: float
                Maximum frequency in Hz
            window_func: str
                Window function used for computing the spectrogram
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to the spectrogram. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram

        Attrs:
            window_func: str
                Window function.
    """
    def __init__(self, data, num_filters, time_res, freq_max, start_bin=0, bins=None, window_func=None, filename=None, offset=0, 
        label=None, annot=None, transforms=None, transform_log=None, waveform_transform_log=None, **kwargs):

        # create frequency axis
        ax = MelAxis(num_filters=num_filters, freq_max=freq_max, start_bin=start_bin, bins=bins, label='Frequency (Hz)')

        # create spectrogram
        kwargs.pop('type', None)
        super().__init__(data=data, time_res=time_res, type=self.__class__.__name__, freq_ax=ax,
            filename=filename, offset=offset, label=label, annot=annot, transforms=transforms, 
            transform_log=transform_log, waveform_transform_log=waveform_transform_log, **kwargs)

        self.window_func = window_func

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'num_filters':self.freq_ax.num_filters, 'freq_max':self.freq_ax.freq_max, 
            'start_bin':self.freq_ax.start_bin, 'bins':self.freq_ax.bins, 'window_func':self.window_func})
        return attrs

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = super().get_kwargs()
        kwargs.pop('freq_ax', None)
        return kwargs

    @classmethod
    def empty(cls):
        """ Creates an empty MelSpectrogram object
        """
        return cls(data=np.empty(shape=(0,0), dtype=np.float64), num_filters=40, time_res=1, freq_min=0, freq_max=0)

    @classmethod
    def from_waveform(cls, audio, window=None, step=None, seg_args=None, window_func='hamming',
        num_filters=40, transforms=None, **kwargs):
        """ Creates a Mel Spectrogram from an :class:`audio_signal.Waveform`.
        
            Args:
                audio: Waveform
                    Audio signal 
                window: float
                    Window length in seconds
                step: float
                    Step size in seconds
                seg_args: dict
                    Input arguments used for evaluating :func:`audio.audio.segment_args`. 
                    Optional. If specified, the arguments `window` and `step` are ignored.
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                num_filters: int
                    The number of filters in the filter bank. Default is 40.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}

            Returns:
                : MelSpectrogram
                    Mel spectrogram
        """
        if window_func is not None: window_func = window_func.lower() #make lowercase

        # compute STFT
        img, freq_nyquist, num_fft, seg_args, phase = aum.stft(x=audio.data, rate=audio.rate, window=window,
            step=step, seg_args=seg_args, window_func=window_func, decibel=False)

        # Magnitude->Mel conversion
        img = aum.mag2mel(img=img, num_fft=num_fft, rate=audio.rate, num_filters=num_filters) 
        img = np.where(img == 0, np.finfo(float).eps, img) #Numerical Stability
        img = aum.to_decibel(img) # convert to dB

        time_res = seg_args['step_len'] / audio.rate   

        return cls(data=img, num_filters=num_filters, time_res=time_res, freq_max=audio.rate/2, window_func=window_func, 
            filename=audio.filename, offset=audio.offset, label=audio.label, annot=audio.annot, 
            waveform_transform_log=audio.transform_log, transforms=transforms, **kwargs)

    @classmethod
    def from_wav(cls, path, window, step, channel=0, rate=None, window_func='hamming', num_filters=40,
            offset=0, duration=None, resample_method='scipy', id=None, normalize_wav=False, transforms=None, 
            waveform_transforms=None, smooth=0.01, **kwargs):            
        """ Create Mel spectrogram directly from wav file.

            The arguments offset and duration can be used to select a portion of the wav file.
            
            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.

            Args:
                path: str
                    Path to wav file
                window: float
                    Window size in seconds
                step: float
                    Step size in seconds 
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                num_filters: int
                    The number of filters in the filter bank. Default is 40.
                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
                        
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                waveform_transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the waveform before generating 
                    the spectrogram. For example,
                    {"name":"add_gaussian_noise", "sigma":0.5}
                smooth: float
                    Width in seconds of the smoothing region used for stitching together audio files.

            Returns:
                spec: MelSpectrogram
                    Mel spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio.spectrogram import MelSpectrogram
                >>> spec = MelSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', window=0.2, step=0.01)
                >>> # crop frequency
                >>> spec = spec.crop(freq_min=50, freq_max=800)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/mel_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/mel_grunt1.png
        """
        # load audio
        audio = load_audio_for_spec(path=path, channel=channel, rate=rate, window=window, step=step,
            offset=offset, duration=duration, resample_method=resample_method, id=id, normalize_wav=normalize_wav,
            waveform_transforms=waveform_transforms, smooth=smooth)

        if audio is None:
            warnings.warn("Empty spectrogram returned", RuntimeWarning)
            return cls.empty()

        # compute spectrogram
        spec = cls.from_waveform(audio=audio, seg_args=audio.stft_args, window_func=window_func, num_filters=num_filters, 
            transforms=transforms, **kwargs)

        return spec

    def plot(self, show_annot=False, figsize=(5,4), cmap='viridis', label_in_title=True, vmin=None, vmax=None, num_labels=5):
        """ Plot the spectrogram with proper axes ranges and labels.

            The colormaps available can be seen here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            TODO: Check implementation for filter_bank=True

            Args:
                show_annot: bool
                    Display annotations
                figsize: tuple
                    Figure size
                cmap: string
                    The colormap to be used
                label_in_title: bool
                    Include label (if available) in figure title
                num_labels: int
                    Number of labels
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        fig = super().plot(show_annot, figsize, cmap, label_in_title, vmin, vmax)
        num = min(self.get_data().shape[1] + 1, num_labels)
        ticks, labels = self.freq_ax.ticks_and_labels(num_labels=num)
        plt.yticks(ticks, labels)
        return fig


class CQTSpectrogram(Spectrogram):
    """ Magnitude Spectrogram computed from Constant Q Transform (CQT).
    
        Args:
            image: 2d or 3d numpy array
                Spectrogram pixel values. 
            time_res: float
                Time resolution in seconds (corresponds to the bin size used on the time axis)
            freq_min: float
                Lower value of the frequency axis in Hz
            bins_per_oct: int
                Number of bins per octave
            window_func: str
                Window function used for computing the spectrogram
            filename: str or list(str)
                Name of the source audio file, if available.   
            offset: float or array-like
                Position in seconds of the left edge of the spectrogram within the source 
                audio file, if available.
            label: int
                Spectrogram label. Optional
            annot: AnnotationHandler
                AnnotationHandler object. Optional
            transforms: list(dict)
                List of dictionaries, where each dictionary specifies the name of 
                a transformation to be applied to the spectrogram. For example,
                {"name":"normalize", "mean":0.5, "std":1.0}
            transform_log: list(dict)
                List of transforms that have been applied to this spectrogram
            waveform_transform_log: list(dict)
                List of transforms that have been applied to the waveform before 
                generating this spectrogram

        Attrs:
            window_func: str
                Window function.
    """
    def __init__(self, data, time_res, bins_per_oct, freq_min, 
        window_func=None, filename=None, offset=0, label=None, annot=None, transforms=None, 
        transform_log=None, waveform_transform_log=None, **kwargs):

        # create logarithmic frequency axis
        ax = Log2Axis(bins=data.shape[1], bins_per_oct=bins_per_oct,\
            min_value=freq_min, label='Frequency (Hz)')

        # create spectrogram
        kwargs.pop('type', None)
        super().__init__(data=data, time_res=time_res, type=self.__class__.__name__, freq_ax=ax,
            filename=filename, offset=offset, label=label, annot=annot, transforms=transforms, 
            transform_log=transform_log, waveform_transform_log=waveform_transform_log, **kwargs)

        self.window_func = window_func

    def get_repres_attrs(self):
        """ Get audio representation attributes """ 
        attrs = super().get_repres_attrs()
        attrs.update({'freq_min':self.freq_min(), 'bins_per_oct':self.bins_per_octave(), 'window_func':self.window_func})
        return attrs

    def get_kwargs(self):
        """ Get keyword arguments required to create a copy of this instance. 

            Does not include the data array and annotation handler.    
        """
        kwargs = super().get_kwargs()
        kwargs.pop('freq_ax', None)
        return kwargs

    @classmethod
    def empty(cls):
        """ Creates an empty CQTSpectrogram object
        """
        return cls(data=np.empty(shape=(0,0), dtype=np.float64), time_res=0, bins_per_oct=0, freq_min=0)

    @classmethod
    def from_waveform(cls, audio, step, bins_per_oct, freq_min=1, freq_max=None, 
                        window_func='hann', transforms=None, **kwargs):
        """ Magnitude Spectrogram computed from Constant Q Transform (CQT) using the librosa implementation:

            https://librosa.github.io/librosa/generated/librosa.core.cqt.html

            The frequency axis of a CQT spectrogram is essentially a logarithmic axis with base 2. It is 
            characterized by an integer number of bins per octave (an octave being a doubling of the frequency.) 

            For further details, see :func:`audio.audio.cqt`.
        
            Args:
                audio: Waveform
                    Audio signal 
                step: float
                    Step size in seconds 
                bins_per_oct: int
                    Number of bins per octave
                freq_min: float
                    Minimum frequency in Hz. Default is 1 Hz.
                freq_max: float
                    Maximum frequency in Hz
                    If None, it is set half the sampling rate.
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming
                        * hanning (default)
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}

            Returns:
                spec: CQTSpectrogram
                    CQT spectrogram
        """
        if window_func is not None: window_func = window_func.lower() #make lowercase

        # compute CQT
        img, step = aum.cqt(x=audio.data, rate=audio.rate, step=step,
            bins_per_oct=bins_per_oct, freq_min=freq_min, freq_max=freq_max,
            window_func=window_func)

        spec = cls(data=img, time_res=step, freq_min=freq_min, bins_per_oct=bins_per_oct, 
            window_func=window_func, filename=audio.filename, 
            offset=audio.offset, label=audio.label, annot=audio.annot, 
            waveform_transform_log=audio.transform_log, transforms=transforms, **kwargs)

        if freq_min is not None or freq_max is not None:
            spec = spec.crop(freq_min=freq_min, freq_max=freq_max)

        return spec

    @classmethod
    def from_wav(cls, path, step, bins_per_oct, freq_min=1, freq_max=None,
        channel=0, rate=None, window_func='hann', offset=0, duration=None,
        resample_method='scipy', id=None, normalize_wav=False, transforms=None,
        waveform_transforms=None, smooth=0.01, **kwargs):
        """ Create CQT spectrogram directly from wav file.

            The arguments offset and duration can be used to select a segment of the audio file.

            Note that values specified for the arguments window, step, offset, and duration 
            may all be subject to slight adjustments to ensure that the selected portion 
            corresponds to an integer number of window frames, and that the window and step 
            sizes correspond to an integer number of samples.
        
            Args:
                path: str
                    Complete path to wav file 
                step: float
                    Step size in seconds 
                bins_per_oct: int
                    Number of bins per octave
                freq_min: float
                    Minimum frequency in Hz. Default is 1 Hz.
                freq_max: float
                    Maximum frequency in Hz
                    If None, it is set half the sampling rate.
                channel: int
                    Channel to read from. Only relevant for stereo recordings
                rate: float
                    Desired sampling rate in Hz. If None, the original sampling rate will be used
                window_func: str
                    Window function (optional). Select between
                        * bartlett
                        * blackman
                        * hamming (default)
                        * hanning

                offset: float
                    Start time of spectrogram in seconds, relative the start of the wav file.
                duration: float
                    Length of spectrogrma in seconds.
                resample_method: str
                    Resampling method. Only relevant if `rate` is specified. Options are
                        * kaiser_best
                        * kaiser_fast
                        * scipy (default)
                        * polyphase
      
                    See https://librosa.github.io/librosa/generated/librosa.core.resample.html 
                    for details on the individual methods.
                id: str
                    Unique identifier (optional). If None, the filename will be used.
                normalize_wav: bool
                    Normalize the waveform to have a mean of zero (mean=0) and a standard 
                    deviation of unity (std=1) before computing the spectrogram. Default is False.
                transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the spectrogram. For example,
                    {"name":"normalize", "mean":0.5, "std":1.0}
                waveform_transforms: list(dict)
                    List of dictionaries, where each dictionary specifies the name of 
                    a transformation to be applied to the waveform before generating 
                    the spectrogram. For example,
                    {"name":"add_gaussian_noise", "sigma":0.5}
                smooth: float
                    Width in seconds of the smoothing region used for stitching together audio files.
                    
            Returns:
                : CQTSpectrogram
                    CQT spectrogram

            Example:
                >>> # load spectrogram from wav file
                >>> from ketos.audio.spectrogram import CQTSpectrogram
                >>> spec = CQTSpectrogram.from_wav('ketos/tests/assets/grunt1.wav', step=0.01, freq_min=10, freq_max=800, bins_per_oct=16)
                >>> # show
                >>> fig = spec.plot()
                >>> fig.savefig("ketos/tests/assets/tmp/cqt_grunt1.png")
                >>> plt.close(fig)

                .. image:: ../../../ketos/tests/assets/tmp/cqt_grunt1.png
        """
        # load audio
        audio = Waveform.from_wav(path=path, rate=rate, channel=channel,
            offset=offset, duration=duration, resample_method=resample_method, 
            id=id, normalize_wav=normalize_wav, transforms=waveform_transforms, smooth=smooth)

        if len(audio.get_data()) == 0:
            warnings.warn("Empty spectrogram returned", RuntimeWarning)
            return cls.empty()

        # create CQT spectrogram
        return cls.from_waveform(audio=audio, step=step, bins_per_oct=bins_per_oct, 
            freq_min=freq_min, freq_max=freq_max, window_func=window_func, transforms=transforms, **kwargs)

    def bins_per_octave(self):
        """ Get no. bins per octave.

            Returns:
                : int
                    No. bins per octave.
        """
        return self.freq_ax.bins_per_oct

    def plot(self, show_annot=False, figsize=(5,4), cmap='viridis', label_in_title=True, vmin=None, vmax=None):
        """ Plot the spectrogram with proper axes ranges and labels.

            Optionally, also display annotations as boxes superimposed on the spectrogram.

            The colormaps available can be seen here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

            Note: The resulting figure can be shown (fig.show())
            or saved (fig.savefig(file_name))

            Args:
                show_annot: bool
                    Display annotations
                figsize: tuple
                    Figure size
                cmap: string
                    The colormap to be used
                label_in_title: bool
                    Include label (if available) in figure title
            
            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        fig = super().plot(show_annot, figsize, cmap, label_in_title, vmin, vmax)
        ticks, labels = self.freq_ax.ticks_and_labels()
        plt.yticks(ticks, labels)
        return fig
