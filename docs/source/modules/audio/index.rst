Audio Processing
-----------------

The audio modules provide high-level interfaces for loading, handling, and manipulating audio data 
and computing various spectral representations such as magnitude spectrograms and CQT spectrograms. 
For the implementation of these functionalities, we rely extensively on 
`LibROSA <https://librosa.github.io/librosa/>`_ and `SoundFile <https://pysoundfile.readthedocs.io/en/latest/index.html>`_ .

Waveforms
~~~~~~~~~
The Waveform class in the :ref:`waveform` module provides a convenient interface for working with 
audio time series. For example, the following command will load a segment of a wav file into memory:: 

    >>> from ketos.audio.waveform import Waveform
    >>> audio = Waveform.from_wav('sound.wav', offset=3.0, duration=6.0) #load 6-s long segment starting 3 s from the beginning of the audio file

The Waveform object thus created stores the audio data as a Numpy array along with the filename, offset, and some additional attributes::

    >>> print(type(audio.get_data()))
    <class 'numpy.ndarray'>
    >>> print(audio.get_filename())
    sound.wav
    >>> print(audio.get_offset())
    3.0
    >>> print(audio.get_attrs())
    {'rate': 44100, 'type': 'Waveform'}

To Waveform class has a number of useful methods for manipulating audio data, e.g., adding Gaussian noise to 
an audio segment, or splitting an audio segment into several shorter segments. Consult the documentation of the 
:ref:`waveform` module for the complete list.


Spectrograms
~~~~~~~~~~~~


.. toctree::
   :maxdepth: 2
   :glob:

   base_audio
   waveform
   spectrogram
   audio_loader
   annotation
   Utilities <utils/index>
