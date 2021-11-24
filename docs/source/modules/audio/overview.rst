Overview
=========

The audio modules provide high-level interfaces for loading and manipulating audio data 
and computing various spectral representations such as magnitude spectrograms and CQT spectrograms. 
For the implementation of these functionalities, we rely extensively on 
`LibROSA <https://librosa.github.io/librosa/>`_ and `SoundFile <https://pysoundfile.readthedocs.io/en/latest/index.html>`_ .


Waveforms
---------

The :class:`Waveform <ketos.audio.waveform.Waveform>` class provides a convenient interface for working with 
audio time series. For example, the following command will load a segment of a wav file into memory:: 

    >>> from ketos.audio.waveform import Waveform
    >>> audio = Waveform.from_wav('sound.wav', offset=3.0, duration=6.0) #load 6-s long segment starting 3 s from the beginning of the audio file

The Waveform object thus created stores the audio data as a Numpy array along with the filename, offset, and some additional attributes::

    >>> type(audio.get_data())
    <class 'numpy.ndarray'>
    >>> audio.get_filename()
    'sound.wav'
    >>> audio.get_offset()
    3.0
    >>> audio.get_attrs()
    {'rate': 1000, 'type': 'Waveform'}

To Waveform class has a number of useful methods for manipulating audio data, e.g., adding Gaussian noise to 
an audio segment (:meth:`add_gaussian_noise() <ketos.audio.waveform.Waveform.add_gaussian_noise>`), or splitting an audio segment 
into several shorter segments (:meth:`segment() <ketos.audio.waveform.Waveform.segment>`). Please consult the documentation of the 
:ref:`waveform` module for the complete list.


Spectrograms
-------------

Four different types of spectrograms have been implemented in ketos: :class:`magnitude spectrogram <ketos.audio.spectrogram.MagSpectrogram>`,
:class:`power spectrogram <ketos.audio.spectrogram.PowSpectrogram>`, :class:`mel spectrogram <ketos.audio.spectrogram.MelSpectrogram>`, and
:class:`CQT spectrogram <ketos.audio.spectrogram.CQTSpectrogram>`. These are all derived from the same 
:class:`Spectrogram <ketos.audio.spectrogram.Spectrogram>` parent class, which in turn derives from the 
:class:`BaseAudio <ketos.audio.base_audio.BaseAudio>` base class.

The spectrogram classes provide interfaces for computing and manipulating spectral frequency presentations of audio data. 
Like a waveform, a spectrogram object can also be created directly from a wav file:: 

    >>> from ketos.audio.spectrogram import MagSpectrogram
    >>> spec = MagSpectrogram.from_wav('sound.wav', window=0.2, step=0.01, offset=3.0, duration=6.0) #spectrogram of a 6-s long segment starting 3 s from the beginning of the audio file

The MagSpectrogram object thus created stores the spectral representation of the audio data as a (masked) 2D Numpy array along with the 
filename, offset, and some additional attributes::

    >>> type(spec.get_data())
    <class 'numpy.ma.core.MaskedArray'>
    >>> audio.get_filename()
    'sound.wav'
    >>> spec.get_offset()
    3.0
    >>> spec.get_attrs()
    {'time_res': 0.01, 'freq_min': 0.0, 'freq_res': 4.9504950495049505, 'window_func': 'hamming', 'type': 'MagSpectrogram'}

While the underlying data array can be accessed via the :attr:`data <ketos.audio.base_audio.BaseAudio.data>`  attribute, 
it is recommended to always use the :meth:`get_data() <ketos.audio.base_audio.BaseAudio.get_data>` function to access 
the data array, as shown in the preceding example.

The spectrogram classes have a number of useful methods for manipulating spectrograms, e.g., cropping in either the time or 
frequency dimension or both (:meth:`crop() <ketos.audio.spectrogram.Spectrogram.crop>`), or recovering 
the original waveform (:meth:`recover_waveform() <ketos.audio.spectrogram.MagSpectrogram.recover_waveform>`). 
Note that annotations can be added to both waveform and spectrogram objects using the 
:meth:`annotate() <ketos.audio.base_audio.BaseAudio.annotate>` method. For example,::

    >>> spec.annotate(start=3.5, end=4.6, label=1)
    >>> spec.get_annotations()
       label  start  end  freq_min  freq_max
    0      1    3.5  4.6       NaN       NaN

See the documentation of the :ref:`spectrogram` module for the complete list.


Loading multiple audio segments
--------------------------------

The :class:`AudioSelectionLoader <ketos.audio.audio_loader.AudioSelectionLoader>` and 
:class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` classes provide 
convenient interfaces for loading a selection or sequence of audio segments into memory, 
one at a time. For example,::

    >>> from ketos.audio.audio_loader import AudioFrameLoader
    >>> # specify the audio representation
    >>> audio_repres = {'type':'MagSpectrogram', 'window':0.2, 'step':0.01}
    >>> # create an object for loading 3-s long segments with a step size of 1.5 s (50% overlap) 
    >>> loader = AudioFrameLoader(frame=3.0, step=1.5, filename='sound.wav', repres=audio_repres)
    >>> # load the first two segments
    >>> spec1 = next(loader)
    >>> spec2 = next(loader)

See the documentation of the :ref:`audio_loader` module for more examples and details.


Configuration files
-------------------

As shown in the example above, the audio representation can be configured with a simple 
Python dictionary. Furthemore, this dictionary can be saved to a JSON file (*.json), which 
can be helpful for storing configurations for later use or for sharing with collaborators.

The audio representations currently implemented in Ketos are: 
:class:`Waveform <ketos.audio.waveform.Waveform>`, 
:class:`Magspectrogram <ketos.audio.spectrogram.MagSpectrogram>` , 
:class:`PowSpectrogram <ketos.audio.spectrogram.PowSpectrogram>`, 
:class:`MelSpectrogram <ketos.audio.spectrogram.MelSpectrogram>`, 
:class:`CQTSpectrogram <ketos.audio.spectrogram.CQTSpectrogram>`, 
:class:`GammatoneFilterBank <ketos.audio.gammatone.GammatoneFilterBank>`, and
:class:`AuralFeatures <ketos.audio.gammatone.AuralFeatures>`. 
These are also listed in the :ref:`audio_loader` module 
along with convenient, shorthand names (e.g. `Mag` for `MagSpectrogram`).

With the dictionary approach, you can specify the type of audio representation 
you wish to work with and supply parameter values for the class constructor. 
For example, your JSON file might look like this,

.. code-block:: json

    {
        "spectrogram": {
            "duration": "5.0 s",
            "rate": "10000 Hz", 
            "window": "0.051 s",
            "step": "0.01955 s",
            "freq_min": "0 Hz",
            "freq_max": "6000 Hz",
            "window_func": "hamming",
            "normalize_wav": "true",
            "type": "MagSpectrogram",
            "transforms": [
                {"name":"reduce_tonal_noise"},
                {"name":"normalize", "mean":0.0, "std":1.0}
            ] 
        }
    }

Note that parameters with physical units are specified as strings. This approach 
gives you the flexibility to use the SI unit that you find most convenient. 
You can use the :meth:`load_audio_representation() <ketos.data_handling.parsing.oad_audio_representation>` 
method to load the contents of the JSON configuration file into a Python dictionary.
This method also takes care of parsing the parameter values that have physical units.

To find out which parameters are available for a given audio representation, 
consult the docstring of the corresponding class constructor.
The `allowed_transforms` attribute will tell you which 
transformations are available. For example, for the 
:class:`Magspectrogram <ketos.audio.spectrogram.MagSpectrogram>` 
class,

    >>> from ketos.audio.spectrogram import MagSpectrogram
    >>> spec = MagSpectrogram.from_wav(path='sound.wav', window=0.2, step=0.01)
    >>> spec.allowed_transforms.keys()   
    dict_keys(['normalize', 'adjust_range', 'crop', 'blur', 'enhance_signal', 'reduce_tonal_noise', 'resize'])
