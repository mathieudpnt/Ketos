""" Unit tests for the the 'spectrogram' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""

import pytest
import numpy as np
from sound_classification.spectrogram import MagSpectrogram, PowerSpectrogram, MelSpectrogram, Spectrogram
from sound_classification.json_parsing import Interval
from sound_classification.audio_signal import AudioSignal
import datetime


def test_init_mag_spectrogram_from_sine_wave(sine_audio):
    
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    NFFT = 256
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(2000, abs=spec.fres)
    
    assert spec.NFFT == NFFT
    assert spec.tres == winstep
    assert spec.fmin == 0    

def test_init_power_spectrogram_from_sine_wave(sine_audio):
    
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    NFFT = 256
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(2000, abs=spec.fres)
    
    assert spec.NFFT == NFFT
    assert spec.tres == winstep
    assert spec.fmin == 0
    
def test_init_mel_spectrogram_from_sine_wave(sine_audio):
    
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    NFFT = 256
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(1360, abs=spec.fres)
    
    assert spec.NFFT == NFFT
    assert spec.tres == winstep
    assert spec.fmin == 0

def test_init_mel_spectrogram_with_kwargs(sine_audio):
    
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    NFFT = 256
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT, n_filters=80, n_ceps=40)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(4444, abs=spec.fres)
    
    assert spec.NFFT == NFFT
    assert spec.tres == winstep
    assert spec.fmin == 0
    assert spec.image.shape[1] == 40

def test_find_bins():    
    img = np.ones(shape=(20,30))
    spec = Spectrogram(image=img, fmin=60.5, fres=0.5)
    b = spec._find_tbin(t=[0.5, 4.5, 99])
    assert b[0] == 0
    assert b[1] == 4
    assert b[2] == img.shape[0]-1
    b = spec._find_fbin(f=[30, 61.2])
    assert b[0] == 0
    assert b[1] == 1
    b = spec._find_fbin(f=[200])
    assert b[0] == img.shape[1]-1

def test_clip_one_box():    
    img = np.ones(shape=(20,30))
    img[6,0] = 1.2
    img[13,0] = 1.66
    spec = Spectrogram(image=img, fmin=60.5, fres=0.5)
    box = [5.1, 10.5, 30., 64.3]
    y = spec.clip(boxes=box)
    assert len(y) == 1
    assert y[0].image.shape[0] == 5
    assert y[0].image.shape[1] == 7
    assert y[0].fmin == 60.5
    assert spec.image.shape[0] == img.shape[0] - y[0].image.shape[0]
    assert y[0].image[1,0] == 1.2
    assert spec.image[8,0] == 1.66

def test_clip_2d_box():    
    img = np.ones(shape=(20,30))
    img[6,0] = 1.2
    img[13,0] = 1.66
    spec = Spectrogram(image=img, fmin=60.5, fres=0.5)
    box = [5.1, 10.5]
    y = spec.clip(boxes=box)
    assert len(y) == 1
    assert y[0].image.shape[0] == 5
    assert y[0].image.shape[1] == img.shape[1]

def test_clip_two_boxes():    
    img = np.ones(shape=(20,30))
    spec = Spectrogram(image=img, fmin=60.5, fres=0.5)
    box1 = [5.1, 10.5, 30., 64.3]
    box2 = [6.1, 11.5, 64.1, 65.1]
    y = spec.clip(boxes=[box1,box2])
    assert len(y) == 2
    assert y[0].image.shape[0] == 5
    assert y[0].image.shape[1] == 7
    assert y[0].fmin == 60.5
    assert y[1].image.shape[0] == 5
    assert y[1].image.shape[1] == 2
    assert y[1].fmin == 64.0
    assert spec.image.shape[0] == img.shape[0] - 6

def test_append_spectrogram(sine_audio):
    spec1 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    spec2 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    size = spec1.image.shape
    spec1.append(spec2)
    assert spec1.image.shape[0] == 2*size[0]
    assert spec1.image.shape[1] == size[1]

def test_sum_spectrogram_has_same_shape_as_original(sine_audio):
    spec1 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    orig_shape = spec1.image.shape
    spec2 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    spec2.crop(tlow=1.0, thigh=2.5, flow=1000, fhigh=4000)
    spec1.add(spec2, delay=0, scale=1)
    assert spec1.image.shape == orig_shape

def test_add_spectrograms_with_different_shapes(sine_audio):
    spec1 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    v_00 = spec1.image[0,0]
    t = spec1._find_tbin(1.5)
    f = spec1._find_fbin(2000)
    v_tf = spec1.image[t,f]
    spec2 = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    spec2.crop(tlow=1.0, thigh=2.5, flow=1000, fhigh=4000)
    spec1.add(spec2, delay=1.0, scale=1.3)
    assert spec1.image[0,0] == v_00 # values outside addition region are unchanged
    assert spec1.image[t,f] == pytest.approx(2.3 * v_tf, rel=0.001) # values outside addition region have changed

def test_cropped_mag_spectrogram_has_correct_size(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    spec.crop(fhigh=4000)
    assert spec.image.shape == (57, 23)
    spec.crop(flow=1000)
    assert spec.image.shape == (57, 18)
    spec.crop(tlow=1.0)
    assert spec.image.shape == (37, 18)
    spec.crop(thigh=2.5)
    assert spec.image.shape == (30, 18)

def test_cropped_power_spectrogram_has_correct_size(sine_audio):
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    spec.crop(fhigh=4000)
    assert spec.image.shape == (57, 23)
    spec.crop(flow=1000)
    assert spec.image.shape == (57, 18)
    spec.crop(tlow=1.0)
    assert spec.image.shape == (37, 18)
    spec.crop(thigh=2.5)
    assert spec.image.shape == (30, 18)

# TODO: Fix cropping method so it also works for Mel spectrograms
#def test_cropped_mel_spectrogram_has_correct_size(sine_audio):
#    spec = MelSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
#    spec.crop(fhigh=4000)
#    assert spec.image.shape == (57, 20)
#    spec.crop(flow=1000)
#    assert spec.image.shape == (57, 15)
#    spec.crop(tlow=1.0)
#    assert spec.image.shape == (37, 15)
#    spec.crop(thigh=2.5)
#    assert spec.image.shape == (30, 15)


def test_mag_compute_average_and_median_without_cropping(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average()
    med = spec.median()
    assert avg == pytest.approx(8620, abs=2.0)
    assert med == pytest.approx(970, abs=2.0)

def test_mag_compute_average_and_median_without_cropping(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average()
    med = spec.median()
    assert avg == pytest.approx(8620, abs=2.0)
    assert med == pytest.approx(970, abs=2.0)
    
def test_power_compute_average_and_median_without_cropping(sine_audio):
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average()
    med = spec.median()
    assert avg == pytest.approx(3567190, abs=2.0)
    assert med == pytest.approx(3677, abs=2.0)

def test_mel_compute_average_and_median_without_cropping(sine_audio):
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average()
    med = spec.median()
    assert avg == pytest.approx(-260, abs=2.0)
    assert med == pytest.approx(-172, abs=2.0)

def test_mag_compute_average_and_median_with_cropping(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=0, thigh=0.4)
    med = spec.median(flow=1000, fhigh=2000)
    assert avg == pytest.approx(8618, abs=0.5)
    assert med == pytest.approx(30931, abs=0.5)   

def test_power_compute_average_and_median_with_cropping(sine_audio):
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=0, thigh=0.4)
    med = spec.median(flow=1000, fhigh=2000)
    assert avg == pytest.approx(3567190, abs=1.0)
    assert med == pytest.approx(3772284, abs=0.5)   

def test_mel_compute_average_and_median_with_cropping(sine_audio):
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=0, thigh=0.4)
    med = spec.median(flow=1000, fhigh=2000)
    assert avg == pytest.approx(-259, abs=1.0)
    assert med == pytest.approx(270, abs=1.0) 


def test_mag_compute_average_with_axis(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=1, thigh=1.2, axis=1)
    assert avg.shape == (3,)
    expected =  np.array([8618.055108, 8618.055108, 8618.055108])
    np.testing.assert_array_almost_equal(avg, expected)

def test_power_compute_average_with_axis(sine_audio):
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=1, thigh=1.2, axis=1)
    assert avg.shape == (3,)
    expected =  np.array([3567190.528536, 3567190.528536, 3567190.528536])
    np.testing.assert_array_almost_equal(avg, expected)

def test_mel_compute_average_with_axis(sine_audio):
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    avg = spec.average(tlow=1, thigh=1.2, axis=1)
    assert avg.shape == (3,)
    expected =  np.array([-259.345679, -259.345679, -259.345679])
    np.testing.assert_array_almost_equal(avg, expected)
        
def test_mag_spectrogram_has_correct_time_axis(sine_audio):
    now = datetime.datetime.today()
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=1, winstep=1, NFFT=256, timestamp=now)
    assert len(spec.taxis()) == 3
    assert spec.taxis()[0] == now
    assert spec.taxis()[1] == now + datetime.timedelta(seconds=1)
    assert spec.taxis()[2] == now + datetime.timedelta(seconds=2)   
    
def test_power_spectrogram_has_correct_time_axis(sine_audio):
    now = datetime.datetime.today()
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=1, winstep=1, NFFT=256, timestamp=now)
    assert len(spec.taxis()) == 3
    assert spec.taxis()[0] == now
    assert spec.taxis()[1] == now + datetime.timedelta(seconds=1)
    assert spec.taxis()[2] == now + datetime.timedelta(seconds=2)   

def test_mel_spectrogram_has_correct_time_axis(sine_audio):
    now = datetime.datetime.today()
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=1, winstep=1, NFFT=256, timestamp=now)
    assert len(spec.taxis()) == 3
    assert spec.taxis()[0] == now
    assert spec.taxis()[1] == now + datetime.timedelta(seconds=1)
    assert spec.taxis()[2] == now + datetime.timedelta(seconds=2)

def test_mag_spectrogram_has_correct_NFFT(sine_audio):
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep)
    
    assert spec.NFFT == int(round(winlen * sine_audio.rate))

def test_power_spectrogram_has_correct_NFFT(sine_audio):
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    spec = PowerSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep)
    
    assert spec.NFFT == int(round(winlen * sine_audio.rate))

def test_mel_spectrogram_has_correct_NFFT(sine_audio):
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    spec = MelSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep)
    
    assert spec.NFFT == int(round(winlen * sine_audio.rate))

def test_blur_time_axis():
    spec = Spectrogram()
    img = np.zeros((21,21))
    img[10,10] = 1
    spec.image = img
    sig = 2.0
    spec.blur_gaussian(tsigma=sig, fsigma=0.01)
    xy = spec.image / np.max(spec.image)
    x = xy[:,10]
    assert x[10] == pytest.approx(1, rel=0.001)
    assert x[9] == pytest.approx(np.exp(-pow(1,2)/(2.*pow(sig,2))), rel=0.001)
    assert x[8] == pytest.approx(np.exp(-pow(2,2)/(2.*pow(sig,2))), rel=0.001)    
    assert xy[10,9] == pytest.approx(0, rel=0.001) 
    
def test_blur_freq_axis():
    spec = Spectrogram()
    img = np.zeros((21,21))
    img[10,10] = 1
    spec.image = img
    sig = 4.2
    spec.blur_gaussian(tsigma=0.01, fsigma=sig)
    xy = spec.image / np.max(spec.image)
    y = xy[10,:]
    assert y[10] == pytest.approx(1, rel=0.001)
    assert y[9] == pytest.approx(np.exp(-pow(1,2)/(2.*pow(sig,2))), rel=0.001)
    assert y[8] == pytest.approx(np.exp(-pow(2,2)/(2.*pow(sig,2))), rel=0.001)    
    assert xy[9,10] == pytest.approx(0, rel=0.001) 

def test_create_audio_from_spectrogram(sine_audio):
    duration = sine_audio.seconds()
    winlen = duration/4
    winstep = duration/10
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep)
    audio = spec.audio_signal()
    assert audio.rate == sine_audio.rate

