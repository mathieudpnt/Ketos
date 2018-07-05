""" Unit tests for the the 'pre_processing' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
import os
import numpy as np
import scipy.signal as sg
import sound_classification.pre_processing as pp

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


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

    signal = 32600 * sg.sawtooth(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal
   

@pytest.fixture
def sine_wave_file(sine_wave):
    """Create a .wav with the 'sine_wave()' fixture
    
       The file is saved as tests/assets/sine_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = os.path.join(path_to_assets,"sine_wave.wav")
    rate, sig = sine_wave
    pp.wave.write(wav_file, rate=rate, data=sig)
    
    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def square_wave_file(square_wave):
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
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def sawtooth_wave_file(sawtooth_wave):
    """Create a .wav with the 'sawtooth_wave()' fixture
    
       The file is saved as ./assets/sawtooth_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = "./assets/sawtooth_wave.wav"
    rate, sig = sawtooth_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


def test_standardize_sample_rate(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)

    duration = 3

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=22000)
    assert new_rate == 22000
    assert len(new_sig) == duration * new_rate 

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=2000)
    assert new_rate == 2000
    assert len(new_sig) == duration * new_rate 

    pp.wave.write(filename="./assets/tmp_sig.wav", rate=new_rate, data=new_sig)
    read_rate, read_sig = pp.wave.read("./assets/tmp_sig.wav")

    assert read_rate == new_rate

if __name__=="__main__":
    print(path_to_assets)