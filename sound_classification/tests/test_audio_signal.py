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




def test_time_stamped_audio_signal_has_correct_begin_and_end_times(sine_audio):
    audio = sine_audio
    today = datetime.datetime.today()
    audio.time_stamp = today 
    seconds = sine_audio.seconds()
    duration = datetime.timedelta(seconds=seconds)
    assert audio.begin() == today
    assert audio.end() == today + duration

def test_crop_audio_signal(sine_audio):
    audio = sine_audio
    seconds = len(audio.data) / audio.rate
    crop_begin = audio.begin() + datetime.timedelta(seconds=seconds/10.)
    crop_end = audio.end() - datetime.timedelta(seconds=seconds/10.)
    audio_cropped = audio
    audio_cropped.crop(begin=crop_begin, end=crop_end)
    seconds_cropped = len(audio_cropped.data) / audio_cropped.rate
    assert seconds_cropped/seconds == pytest.approx(8./10., rel=1./audio.rate)
    assert audio_cropped.begin() == crop_begin


def test_clip_with_positive_sample_id(sine_audio):
    audio = sine_audio 
    t0 = audio.time_stamp
    first = audio.data[0]
    last = audio.data[-1]
    n = len(audio.data)
    m = int(0.1*n)
    dt = m / audio.rate
    s = audio.clip(m)
    assert len(s.data) == m
    assert len(audio.data) == n-m
    assert first == s.data[0]
    assert last == audio.data[-1]
    assert audio.time_stamp == t0 + datetime.timedelta(microseconds=1e6*dt)
    assert s.time_stamp == t0

def test_clip_with_sample_larger_than_length(sine_audio):
    audio = sine_audio
    n = len(audio.data)
    m = int(1.5*n)
    s = audio.clip(m)
    assert len(s.data) == n
    assert audio.empty() == True

def test_clip_with_negative_sample_id(sine_audio):
    audio = sine_audio
    t0 = audio.begin()
    t1 = audio.end()
    first = audio.data[0]
    last = audio.data[-1]
    n = len(audio.data)
    m = -int(0.2*n)
    dt = -m / audio.rate
    s = audio.clip(m)
    assert len(s.data) == -m
    assert len(audio.data) == n+m
    assert first == audio.data[0]
    assert last == s.data[-1]
    assert s.time_stamp == t1 - datetime.timedelta(microseconds=1e6*dt)
    assert audio.time_stamp == t0

def test_append_audio_signal_to_itself(sine_audio):
    audio = sine_audio
    len_sum = 2 * len(audio.data)
    audio_copy = audio.copy()
    audio_copy.time_stamp = audio.end()
    audio.append(audio_copy)
    assert len(audio.data) == len_sum
    


def test_append_audio_signal_without_time_stamp_to_itself(sine_audio_without_time_stamp): 
    len_sum = 2 * len(sine_audio_without_time_stamp.data)
    sine_audio_without_time_stamp.append(sine_audio_without_time_stamp)
    assert len(sine_audio_without_time_stamp.data) == len_sum

def test_append_with_smoothing(sine_audio):
    audio = sine_audio
    t = audio.seconds()
    at = audio.append(signal=audio, n_smooth=100)
    assert audio.seconds() == pytest.approx(2.*t - 100/audio.rate, rel=1./audio.rate)
    assert at == audio.begin() + datetime.timedelta(microseconds=1e6*(t - 100/audio.rate))

def test_append_with_delay(sine_audio):
    audio = sine_audio
    t = audio.seconds()
    delay = 3.0
    audio.append(signal=audio, delay=3)
    assert audio.seconds() == 2.*t + 3.0

def test_append_with_max_length(sine_audio):
    audio = sine_audio
    audio2 = audio.copy()
    t = audio.seconds()
    n = len(audio.data)
    nmax = int(1.5 * n)
    audio.append(signal=audio2, max_length=nmax)
    assert len(audio.data) == nmax
    assert len(audio2.data) == 2*n - nmax

def test_append_with_max_length_and_smooth(sine_audio):
    audio = sine_audio
    audio2 = audio.copy()
    t = audio.seconds()
    n = len(audio.data)
    nmax = int(1.5 * n)
    n_smooth = 200
    audio.append(signal=audio2, n_smooth=n_smooth, max_length=nmax)
    assert len(audio.data) == nmax
    assert len(audio2.data) == 2*n - n_smooth - nmax

def test_append_with_max_length_and_delay(sine_audio):
    audio = sine_audio
    audio2 = audio.copy()
    t = audio.seconds()
    n = len(audio.data)
    nmax = int(1.5 * n)
    delay = t
    audio.append(signal=audio2, delay=t, max_length=nmax)
    assert len(audio.data) == nmax
    assert len(audio2.data) == n
    assert np.ma.is_masked(audio.data[-1]) 

def test_append_delay_determined_from_time_stamps(sine_audio):
    audio = sine_audio
    audio2 = audio.copy()
    dt = 5.
    audio2.time_stamp = audio.end() + datetime.timedelta(microseconds=1e6*dt)
    t = audio.seconds()
    n = len(audio.data)
    audio.append(signal=audio2)
    assert audio.seconds() == 2*t + dt
    assert np.ma.is_masked(audio.data[n]) 
    
def test_add_identical_audio_signals(sine_audio):
    audio = sine_audio
    t = audio.seconds()
    v = np.copy(audio.data)
    audio.add(signal=audio)
    assert audio.seconds() == t
    assert np.all(audio.data == 2*v)
    
def test_add_identical_audio_signals_with_delay(sine_audio):
    audio = sine_audio
    t = audio.seconds()
    v = np.copy(audio.data)
    delay = 1
    audio.add(signal=audio, delay=delay)
    assert audio.seconds() == t
    i = int(audio.rate * delay)
    assert audio.data[5] == v[5]
    assert audio.data[i+5] == v[i+5] + v[5]    
    
def test_add_identical_audio_signals_with_scaling_factor(sine_audio):
    audio = sine_audio
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
