""" Unit tests for the the 'audio_signal' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
import sound_classification.audio_signal as aud
import datetime
import numpy as np


today = datetime.datetime.today()


@pytest.fixture
def audio(sine_wave):
    rate, data = sine_wave
    a = aud.TimeStampedAudioSignal(rate=rate, data=data, time_stamp=today, tag="audio")
    return a

@pytest.fixture
def audio_without_time_stamp(sine_wave):
    rate, data = sine_wave
    a = aud.AudioSignal(rate=rate, data=data)
    return a

def test_time_stamped_audio_signal_has_correct_begin_and_end_times(audio):
    seconds = len(audio.data) / audio.rate
    duration = datetime.timedelta(seconds=seconds) 
    assert audio.begin() == today
    assert audio.end() == today + duration

def test_crop_audio_signal(audio):
    seconds = len(audio.data) / audio.rate
    crop_begin = audio.begin() + datetime.timedelta(seconds=seconds/10.)
    crop_end = audio.end() - datetime.timedelta(seconds=seconds/10.)
    audio_cropped = audio
    audio_cropped.crop(begin=crop_begin, end=crop_end)
    seconds_cropped = len(audio_cropped.data) / audio_cropped.rate
    assert seconds_cropped/seconds == pytest.approx(8./10., rel=1./audio.rate)
    assert audio_cropped.begin() == crop_begin

def test_append_audio_signal(audio):
    len_sum = 2 * len(audio.data)
    audio_copy = audio.copy()
    audio_copy.time_stamp = audio.end()
    audio.append(audio_copy)
    assert len(audio.data) == len_sum
    
def test_append_audio_signal_without_time_stamp(audio, audio_without_time_stamp): 
    len_sum = len(audio.data) + len(audio_without_time_stamp.data)
    audio.append(audio_without_time_stamp)
    assert len(audio.data) == len_sum

def test_append_audio_signal_with_smoothing(audio): 
    t = audio.seconds()
    audio.append(signal=audio, delay=-0.2)
    assert audio.seconds() == pytest.approx(2.*t - 0.2, rel=1./audio.rate)
    
def test_add_identical_audio_signals(audio): 
    t = audio.seconds()
    v = np.copy(audio.data)
    audio.add(signal=audio)
    assert audio.seconds() == t
    assert np.all(audio.data == 2*v)
    
def test_add_identical_audio_signals_with_delay(audio): 
    t = audio.seconds()
    v = np.copy(audio.data)
    delay = 1
    audio.add(signal=audio, delay=delay)
    assert audio.seconds() == t
    i = int(audio.rate * delay)
    assert audio.data[5] == v[5]
    assert audio.data[i+5] == v[i+5] + v[5]    
    
def test_add_identical_audio_signals_with_scaling_factor(audio): 
    v = np.copy(audio.data)
    audio.add(signal=audio, scale=0.5)
    assert np.all(audio.data == 1.5*v)

def test_morlet_with_default_params():
    mor = aud.AudioSignal.morlet(rate=4000, frequency=20, width=1)
    assert len(mor.data) == int(6*1*4000) # check number of samples
    assert max(mor.data) == pytest.approx(1, abs=0.01) # check max signal is 1
    assert np.argmax(mor.data) == pytest.approx(0.5*len(mor.data), abs=1) # check peak is centered
    assert mor.data[0] == pytest.approx(0, abs=0.02) # check signal is approx zero at start

def test_gaussian_noise():
    noise = aud.AudioSignal.gaussian_noise(rate=2000, sigma=2, samples=40000)
    assert noise.std() == pytest.approx(2, rel=0.01) # check standard deviation
    assert noise.average() == pytest.approx(0, abs=3*2/np.sqrt(40000)) # check mean
    assert noise.seconds() == 20 # check length

def test_resampled_signal_has_correct_rate(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)
    signal.resample(new_rate=22000)
    assert signal.rate == 22000
    signal.resample(new_rate=2000)
    assert signal.rate == 2000

def test_resampled_signal_has_correct_length(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)
    duration = len(signal.data) / signal.rate
    signal.resample(new_rate=22000)
    assert len(signal.data) == duration * signal.rate 
    signal.resample(new_rate=2000)
    assert len(signal.data) == duration * signal.rate 

def test_resampling_preserves_signal_shape(const_wave_file):
    signal = aud.AudioSignal.from_wav(const_wave_file)
    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)
    n = min(len(signal.data), len(new_signal.data))
    for i in range(n):
        assert signal.data[i] == new_signal.data[i]

def test_resampling_preserves_signal_frequency(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)
    y = abs(np.fft.rfft(signal.data))
    freq = np.argmax(y)
    freqHz = freq * signal.rate / len(signal.data)
    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)
    new_y = abs(np.fft.rfft(new_signal.data))
    new_freq = np.argmax(new_y)
    new_freqHz = new_freq * new_signal.rate / len(new_signal.data)
    assert freqHz == new_freqHz
