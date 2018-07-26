import pytest
import os
import numpy as np
import scipy.signal as sg
import pandas as pd
import sound_classification.pre_processing as pp
import sound_classification.data_handling as dh
import sound_classification.neural_networks as nn
from tensorflow import reset_default_graph

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
def database_prepared_for_NN():
    img = image_2x2()
    d = {'image': [img,img,img,img,img,img], 'label': [0,0,0,0,0,0]}
    df = pd.DataFrame(data=d)
    divisions = {"train":(0,3),"validation":(3,4),"test":(4,6)}
    prepared = dh.prepare_database(df, "image", "label", divisions)     
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

    prepared = dh.prepare_database(database=database,x_column="image",y_column="label",
                                divisions=divisions)    
    return prepared


@pytest.fixture
def trained_CNNWhale(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    path_to_saved_model = os.path.join(path_to_assets, "saved_models")
    path_to_meta = os.path.join(path_to_saved_model, "trained_CNNWhale")     
    
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = nn.CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, batch_size=1, num_channels=2, num_labels=2)
    tf_nodes = network.create_net_structure()
    network.set_tf_nodes(tf_nodes)
    network.train()
    network.save_model(path_to_meta)

    test_acc = network.accuracy_on_test()

    meta = path_to_meta + ".meta"

    reset_default_graph()
    return meta, path_to_saved_model, test_acc
