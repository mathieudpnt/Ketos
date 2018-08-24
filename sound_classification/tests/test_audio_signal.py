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
    audio.append(audio)
    assert len(audio.data) == len_sum
    
def test_append_audio_signal_without_time_stamp(audio, audio_without_time_stamp): 
    len_sum = len(audio.data) + len(audio_without_time_stamp.data)
    audio.append(audio_without_time_stamp)
    assert len(audio.data) == len_sum

def test_append_audio_signal_with_smoothing(audio): 
    t = audio.seconds()
    audio.append(signal=audio, overlap_sec=0.2)
    assert audio.seconds() == pytest.approx(2.*t - 0.2, rel=1./audio.rate)
