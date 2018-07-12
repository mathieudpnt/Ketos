import pytest
import os
import numpy as np
import scipy.signal as sg
import pandas as pd
import sound_classification.pre_processing as pp
import sound_classification.data_handling as dh

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.fixture
def sine_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600*np.sin(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal


@pytest.fixture
def square_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600 * sg.square(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal

@pytest.fixture
def sawtooth_wave():
    sampling_rate = 44100
    frequency = 2000
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


@pytest.fixture
def image_2x2():
    image = np.array([[1,2],[3,4]], np.float32)
    return image


@pytest.fixture
def datebase_with_one_image_col_and_one_label_col():
    img = image_2x2()
    d = {'image': [img], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def datebase_with_one_image_col_and_no_label_col():
    img = image_2x2()
    d = {'image': [img]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def datebase_with_two_image_cols_and_one_label_col():
    img = image_2x2()
    d = {'image1': [img], 'image2': [img], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def datebase_prepared_for_NN():
    img = image_2x2()
    d = {'image': [img,img,img,img,img,img], 'label': [1,1,1,1,1,1]}
    df = pd.DataFrame(data=d)
    divisions = {"train":(0,3),"validation":(3,4),"test":(4,6)}
    prepared = dh.prepare_database(df, "image", "label", divisions)     
    return prepared

