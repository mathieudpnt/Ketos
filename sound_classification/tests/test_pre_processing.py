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
def const_wave():
    sampling_rate = 44100
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = np.ones(len(x))

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
    wav_file = os.path.join(path_to_assets, "sine_wave.wav")
    rate, sig = sine_wave
    pp.wave.write(wav_file, rate=rate, data=sig)
    
    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def square_wave_file(square_wave):
    """Create a .wav with the 'square_wave()' fixture
    
       The file is saved as tests/assets/square_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "square_wave.wav")
    rate, sig = square_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def sawtooth_wave_file(sawtooth_wave):
    """Create a .wav with the 'sawtooth_wave()' fixture
    
       The file is saved as tests/assets/sawtooth_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "sawtooth_wave.wav")
    rate, sig = sawtooth_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def const_wave_file(const_wave):
    """Create a .wav with the 'const_wave()' fixture
    
       The file is saved as tests/assets/const_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "const_wave.wav")
    rate, sig = const_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.mark.test_standardize_sample_rate
def test_resampled_signal_has_correct_rate(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)

    duration = len(sig) / rate

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=22000)
    assert new_rate == 22000

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=2000)
    assert new_rate == 2000

    tmp_file = os.path.join(path_to_assets,"tmp_sig.wav")
    pp.wave.write(filename=tmp_file, rate=new_rate, data=new_sig)
    read_rate, read_sig = pp.wave.read(tmp_file)

    assert read_rate == new_rate

@pytest.mark.test_standardize_sample_rate
def test_resampled_signal_has_correct_length(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)

    duration = len(sig) / rate

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=22000)
    assert len(new_sig) == duration * new_rate 

    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=2000)
    assert len(new_sig) == duration * new_rate 

@pytest.mark.test_standardize_sample_rate
def test_resampling_preserves_signal_shape(const_wave_file):
    rate, sig = pp.wave.read(const_wave_file)
    new_rate, new_sig = pp.standardize_sample_rate(sig=sig, orig_rate=rate, new_rate=22000)

    n = min(len(sig), len(new_sig))
    for i in range(n):
        assert sig[i] == new_sig[i]

if __name__=="__main__":
    print(path_to_assets)