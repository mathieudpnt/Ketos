Audio Processing
-----------------

The audio modules provide high-level interfaces for loading, handling, and manipulating audio data 
and computing various spectral representations such as magnitude spectrograms and CQT spectrograms. 
For the implementation of these functionalities, we rely extensively on 
`LibROSA <https://librosa.github.io/librosa/>`_ and `SoundFile <https://pysoundfile.readthedocs.io/en/latest/index.html>`_ .


Waveforms
~~~~~~~~~
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
an audio segment (:meth:`add_gaussian_noise <ketos.audio.waveform.Waveform.add_gaussian_noise>`), or splitting an audio segment 
into several shorter segments (:meth:`segment <ketos.audio.waveform.Waveform.segment>`). Please consult the documentation of the 
:ref:`waveform` module for the complete list.


Spectrograms
~~~~~~~~~~~~
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

The spectrogram classes have a number of useful methods for manipulating spectrograms, e.g., cropping in either the time or 
frequency dimension or both (:meth:`crop <ketos.audio.spectrogram.Spectrogram.crop>`), or recovering 
the original waveform (:meth:`recover_waveform <ketos.audio.spectrogram.MagSpectrogram.recover_waveform>`). 
Note that annotations can be added to both waveform and spectrogram objects using the 
:meth:`annotate <ketos.audio.base_audio.BaseAudio.annotate>` method. For example,::

    >>> spec.annotate(start=3.5, end=4.6, label=1)
    >>> spec.get_annotations()
   label  start  end  freq_min  freq_max
0      1    3.5  4.6       NaN       NaN

Please consult the documentation of the :ref:`spectrogram` module for the complete list.


.. toctree::
   :maxdepth: 2
   :glob:

   base_audio
   waveform
   spectrogram
   audio_loader
   annotation
   Utilities <utils/index>
