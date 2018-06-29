import pytest
import os
import numpy as np
import scipy.signal as sg
import sound_classification.pre_processing as pp


@pytest.fixture
def sine_wave():
    sampling_rate = 44100
    frequency = 20
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600*np.sin(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal


@pytest.fixture
def square_wave():
    sampling_rate = 44100
    frequency = 20
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600 * sg.square(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal

@pytest.fixture
def sawtooth_wave():
    sampling_rate = 44100
    frequency = 20
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600 * sg.square(2 * np.pi * frequency * x / sampling_rate) 

    return signal
   

@pytest.fixture
def sine_wave_file(sine_wave):
    """Create a .wav with the 'sine_wave()' fixture
    
       The file is saved as ./assets/sine_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = "./assets/sine_wave.wav"
    pp.wave.write(wav_file, rate=44100, data=sine_wave)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def square_wave_file(sine_wave):
    """Create a .wav with the 'square_wave()' fixture
    
       The file is saved as ./assets/square_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = "./assets/square_wave.wav"
    rate, sig = square_wave
    pp.wave.write(wav_file, rate=rate, data=signal)

    yield wav_file
    os.remove(wav_file)
