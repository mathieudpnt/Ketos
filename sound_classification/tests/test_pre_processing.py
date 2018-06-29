import pytest
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

    return signal
