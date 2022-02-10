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

import pytest
import datetime
import os
import numpy as np
import scipy.signal as sg
import soundfile as sf
import pandas as pd
import ketos.audio.utils.misc as ap
from ketos.data_handling.data_handling import to1hot
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.audio.waveform import Waveform
from ketos.audio.utils.axis import LinearAxis, Log2Axis, MelAxis
import ketos.audio.base_audio as aba

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.fixture
def sine_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = np.sin(2 * np.pi * frequency * x / sampling_rate) 
    return sampling_rate, signal

@pytest.fixture
def square_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = sg.square(2 * np.pi * frequency * x / sampling_rate) 
    return sampling_rate, signal

@pytest.fixture
def sawtooth_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = sg.sawtooth(2 * np.pi * frequency * x / sampling_rate) 
    return sampling_rate, signal

@pytest.fixture
def const_wave():
    sampling_rate = 44100
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = np.ones(len(x))
    return sampling_rate, signal

@pytest.fixture
def growing_sine_wave():
    sampling_rate = 1000
    frequency = 18
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = x / np.max(x) * np.sin(2 * np.pi * frequency * x / sampling_rate) 
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
    sf.write(wav_file, sig, rate)    
    yield wav_file
    os.remove(wav_file)

@pytest.fixture
def sine_wave_file_half(sine_wave):
    """Create a .wav with the 'sine_wave()' fixture, with an amplitude
        of 0.5 instead of 1.

       The file is saved as tests/assets/sine_wave_half.wav.
       When the tests using this fixture are done, 
       the file is deleted.

       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = os.path.join(path_to_assets, "sine_wave_half.wav")
    rate, sig = sine_wave
    sf.write(wav_file, 0.5*sig, rate)    
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
    sf.write(wav_file, sig, rate)    
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
    sf.write(wav_file, sig, rate)    
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
    sf.write(wav_file, sig, rate)    
    yield wav_file
    os.remove(wav_file)

@pytest.fixture
def growing_sine_wave_file(growing_sine_wave):
    """Create a .wav with the 'growing_sine_wave()' fixture
    
       The file is saved as tests/assets/growing_sine_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.

       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = os.path.join(path_to_assets, "growing_sine_wave.wav")
    rate, sig = growing_sine_wave
    sf.write(wav_file, sig, rate)    
    yield wav_file
    os.remove(wav_file)

@pytest.fixture
def image_2x2():
    image = np.array([[1,2],[3,4]], np.float32)
    return image

@pytest.fixture
def image_3x3():
    image = np.array([[1,2,3],[4,5,6],[7,8,9]], np.float32)
    return image

@pytest.fixture
def image_ones_10x10():
    image = np.ones(shape=(10,10))
    return image

@pytest.fixture
def image_zeros_and_ones_10x10():
    image = np.ones(shape=(10,10))
    for i in range(10):
        for j in range(5):
            image[i,j] = 0
    return image

@pytest.fixture
def datebase_with_one_image_col_and_one_label_col(image_2x2):
    d = {'image': [image_2x2], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def datebase_with_one_image_col_and_no_label_col(image_2x2):
    d = {'image': [image_2x2]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def datebase_with_two_image_cols_and_one_label_col(image_2x2):
    d = {'image1': [image_2x2], 'image2': [image_2x2], 'label': [1]}
    df = pd.DataFrame(data=d)
    return df


@pytest.fixture
def database_prepared_for_NN(image_2x2):
    d = {'image': [image_2x2, image_2x2, image_2x2, image_2x2, image_2x2, image_2x2], 'label': [0,0,0,0,0,0]}
    df = pd.DataFrame(data=d)
    divisions = {"train":(0,3),"validation":(3,4),"test":(4,6)}
    prepared = prepare_database(df, "image", "label", divisions)     
    return prepared

@pytest.fixture
def database_prepared_for_NN_2_classes():
    img1 = np.zeros((20, 20))
    img2 = np.ones((20, 20))
    d = {'image': [img1, img2, img1, img2, img1, img2,
                   img1, img2, img1, img2, img1, img2,
                   img1, img2, img1, img2, img1, img2,
                   img1, img2, img1, img2, img1, img2],
         'label': [0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1]}
    database = pd.DataFrame(data=d)
    divisions= {"train":(0,12),
                "validation":(12,18),
                "test":(18,len(database))}
    prepared = prepare_database(database=database,x_column="image",y_column="label",
                                divisions=divisions)    
    return prepared


@pytest.fixture
def sine_audio(sine_wave):
    rate, data = sine_wave
    a = Waveform(rate=rate, data=data, filename='sine_wave')
    return a
    
@pytest.fixture
def data_classified_by_nn():
    x = [1, 2, 3, 4, 5, 6] # input data
    x = np.array(x)
    y = [0, 1, 0, 1, 0, 1] # labels
    y = np.array(y)
    w = [[0.8, 0.2], [0.1, 0.9], [0.96, 0.04], [0.49, 0.51], [0.45, 0.55], [0.60, 0.40]] # class weights computed by NN
    w = np.array(w)
    return x,y,w

@pytest.fixture
def data_for_TCN():
    fv0 = np.zeros(64)
    fv1 = np.ones(64)
    x_train = np.array([fv0, fv1, fv0, fv1, fv0, fv1, fv0, fv1, fv0, fv1])
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_val = np.array([fv0, fv1, fv0, fv1])
    y_val = np.array([0, 1, 0, 1])
    x_test = np.array([fv0, fv1, fv0, fv1])
    y_test = np.array([0, 1, 0, 1])
    return x_train, y_train, x_val, y_val, x_test, y_test
    

def encode_database(database, x_column, y_column):
    image_shape = database[x_column][0].shape
    depth = database[y_column].max() + 1 #number of classes
    database["one_hot_encoding"] = database[y_column].apply(to1hot,depth=depth)
    database["x_flatten"] = database[x_column].apply(lambda x: x.flatten())
    return database, image_shape

def split_database(database, divisions):
    train_data = database[divisions["train"][0]:divisions["train"][1]]
    validation_data = database[divisions["validation"][0]:divisions["validation"][1]]
    test_data = database[divisions["test"][0]:divisions["test"][1]]
    datasets = {"train": train_data,
                "validation": validation_data,
                "test": test_data}
    return datasets

def stack_dataset(dataset, input_shape):
    x = np.vstack(dataset.x_flatten).reshape(dataset.shape[0], input_shape[0], input_shape[1],1).astype(np.float32)
    y = np.vstack(dataset.one_hot_encoding)
    stacked_dataset = {'x': x,
                       'y': y}
    return stacked_dataset

def prepare_database(database, x_column, y_column, divisions):
    encoded_data, input_shape = encode_database(database=database, x_column=x_column, y_column=y_column)
    datasets = split_database(database=encoded_data, divisions=divisions)
    stacked_train = stack_dataset(dataset=datasets["train"], input_shape=input_shape)
    stacked_validation = stack_dataset(dataset=datasets["validation"], input_shape=input_shape)
    stacked_test = stack_dataset(dataset=datasets["test"], input_shape=input_shape)
    stacked_datasets = {"train_x": stacked_train["x"],
                        "train_y": stacked_train["y"],
                        "validation_x": stacked_validation["x"],
                        "validation_y": stacked_validation["y"],
                        "test_x": stacked_test["x"],
                        "test_y": stacked_test["y"]}
    return stacked_datasets


@pytest.fixture
def file_duration_table():
    """ Create a table of file durations as a pandas DataFrame.

        Yields:
            tbl: pandas DataFrame
                File duration table
    """
    N = 6
    filename = ['f{0}.wav'.format(x) for x in np.arange(N)]
    duration = [x + 30.0 for x in np.arange(N)]
    tbl = pd.DataFrame({'filename': filename, 'duration': duration})
    return tbl

@pytest.fixture
def annot_table_std():
    """ Create a standardized annotations table as a pandas DataFrame.

        Yields:
            tbl: pandas DataFrame
                Annotation table
    """
    label = [1, 2, 3, 0, 0, -1]
    N = len(label)
    filename = ['f{0}.wav'.format(x%3) for x in np.arange(N)]
    start = np.arange(N, dtype=float)
    end = start + 3.3
    tbl = pd.DataFrame({'filename': filename, 'label': label, 'start': start, 'end': end})
    return tbl

@pytest.fixture
def annot_table():
    """ Create an annotations table as a pandas DataFrame.

        Yields:
            tbl: pandas DataFrame
                Annotation table
    """
    label = [1, 2, 'k', -99, 'whale', 'zebra']
    N = len(label)
    filename = ['f{0}.wav'.format(x) for x in np.arange(N)]
    start = np.arange(N)
    stop = start + 1
    tbl = pd.DataFrame({'fname': filename, 'label': label, 'start': start, 'STOP': stop})
    return tbl

@pytest.fixture
def annot_table_mult_labels():
    """ Create an annotations table as a pandas DataFrame with 
        multiple labels per row.

        Yields:
            tbl: pandas DataFrame
                Annotation table
    """
    label = ['1,2', 3]
    N = len(label)
    filename = ['f{0}.wav'.format(x) for x in np.arange(N)]
    start = np.arange(N)
    end = start + 1
    tbl = pd.DataFrame({'filename': filename, 'label': label, 'start': start, 'end': end})
    return tbl

@pytest.fixture
def annot_table_file(annot_table):
    """ Create an annotations table csv file with the 'annot_table()' fixture
    
        The file is saved as tests/assets/tmp/annot_002.csv.
        When the tests using this fixture are done, 
        the file is deleted.

        Yields:
            csv_file : str
                A string containing the path to the .csv file.
    """
    csv_file = os.path.join(os.path.join(path_to_assets, "tmp"), "annot_002.csv")
    tbl = annot_table
    tbl.to_csv(csv_file, index=False)
    yield csv_file
    os.remove(csv_file)

@pytest.fixture
def linear_axis_200():
    """ Create a linear axis with range 0-100 and 200 bins.

        Yields:
            ax: LinearAxis
                Axis object
    """
    ax = LinearAxis(bins=200, extent=(0.,100.))
    return ax

@pytest.fixture
def log2_axis_8_16():
    """ Create a log2 axis with 8 octaves, 16 bins per octave, and 
        minimum value of 10.

        Yields:
            ax: Log2Axis
                Axis object
    """
    ax = Log2Axis(bins=8*16, bins_per_oct=16, min_value=10.)
    return ax

@pytest.fixture
def mel_axis_40_500():
    """ Create a Mel frequency axis with with 40 bins (filters)
        for a maximum frequncy of 500 Hz

        Yields:
            ax: MelAxis
                Axis object
    """
    ax = MelAxis(num_filters=40, freq_max=500)
    return ax

@pytest.fixture
def spec_image_with_attrs():
    """ Creates a spectrogram image with shape (20,10) and random pixel values, 
        with time resolution of 0.5 s, and a linear frequency axis from 0 to 
        500 Hz.

        Yields:
            img: 2d numpy array
                Pixel values
            dt: float
                Time resolution
            ax: LinearAxis
                Frequency axis            
    """
    img = np.random.rand(20,10)
    dt = 0.5
    ax = LinearAxis(bins=10, extent=(0.,500.), label='Frequency (Hz)')
    return img,dt,ax

@pytest.fixture
def base_audio_1d():
    """ Create a simple 1d BaseAudio object with value 1 everywhere, length 
        of 10 s, filename 'x', offset of 2 s, and label 13.

        Yields:
            o: BaseAudio
                BaseAudio object
    """
    N = 10000
    d = np.ones(N)
    o = aba.BaseAudio(data=d, filename='x', offset=2., label=13, duration=10.)
    return o, d

@pytest.fixture
def base_audio_time_1d():
    """ Create a simple 1d BaseAudioTime object with value 1 everywhere, length 
        of 10 s, time resolution of 0.001 s, filename 'x', offset of 2 s, and label 13.

        Yields:
            o: BaseAudio
                BaseAudio object
    """
    N = 10000
    d = np.ones(N)
    o = aba.BaseAudioTime(time_res=0.001, data=d, filename='x', offset=2., label=13)
    return o, d

@pytest.fixture
def five_time_stamped_wave_files():
    """ Each wav file is 0.5 second long. The sampling rate is 1 kHz."""
    N = 5
    path_to_tmp = os.path.join(path_to_assets,'tmp')
    folder = path_to_tmp + '/five_time_stamped_wave_files/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    files = list()
    for i in range(N):
        fname = 'empty_HMS_12_ 5_ {0}__DMY_23_ 2_84.wav'.format(i)
        full_path = os.path.join(folder, fname)
        a = Waveform(rate=1000, data=np.random.uniform(size=500))
        a.to_wav(full_path)
        files.append(full_path)

    yield folder

    for f in files:
        os.remove(f)

@pytest.fixture
def spectr_settings():
    j = '{"spectrogram": {"type":"MagSpectrogram", "rate": "20 kHz",\
        "window": "0.1 s", "step": "0.025 s",\
        "window_func": "hamming", "freq_min": "30Hz", "freq_max": "3000Hz",\
        "duration": "1.0s", "resample_method": "scipy", "normalize_wav": "False", "decibel": "true",\
        "transforms": [{"name":"enhance_signal", "enhancement":1.0}, {"name":"adjust_range", "range":"(0,1)"}],\
        "waveform_transforms": [{"name":"add_gaussian_noise", "sigma":0.2}] }}'
    return j

@pytest.fixture
def sample_data():
    data = np.vstack([np.zeros((10,64,64,1)), np.ones((10,64,64,1))])
    labels = np.concatenate([np.array([[1,0] for i in range(10)]), np.array([[0,1] for i in range(10)])])
    return (data, labels)

@pytest.fixture
def sample_data_1d():
    data = np.vstack([np.zeros((10,20000,1)), np.ones((10,20000,1))])
    labels = np.concatenate([np.array([[1,0] for i in range(10)]), np.array([[0,1] for i in range(10)])])
    return (data, labels)