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
from sound_classification.spectrogram import MagSpectrogram, PowerSpectrogram, MelSpectrogram
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
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT)
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
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=winlen, winstep=winstep, NFFT=NFFT)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(2000, abs=spec.fres)
    
    assert spec.NFFT == NFFT
    assert spec.tres == winstep
    assert spec.fmin == 0
    

def test_cropped_mag_spectrogram_has_correct_size(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    print(spec.image.shape)
    spec.crop(fhigh=4000)
    print(spec.image.shape)
    assert spec.image.shape == (57, 23)
    spec.crop(flow=1000)
    assert spec.image.shape == (57, 18)
    spec.crop(tlow=1.0)
    assert spec.image.shape == (37, 18)
    spec.crop(thigh=2.5)
    assert spec.image.shape == (30, 18)

def test_cropped_spectrogram_has_correct_position(image_zeros_and_ones_10x10):
    spec = Spectrogram(image=image_zeros_and_ones_10x10, NFFT=256, tres=0.1, fres=2)
    spec_crop = Spectrogram.cropped(spec, flow=8.0, fhigh=12.0)
    assert spec_crop.shape() == (10, 2)
    for i in range(10):
        assert spec_crop.image[i,0] == 0
        assert spec_crop.image[i,1] == 1

def test_compute_average_and_median_without_cropping(image_2x2):
    img = image_2x2
    spec = Spectrogram(image=img, NFFT=256, tres=0.5, fres=2)
    avg = spec.average()
    med = spec.median()
    assert avg == np.average(img)
    assert med == np.median(img)
    
def test_compute_average_and_median_with_cropping(image_3x3):
    img = image_3x3
    spec = Spectrogram(image=img, NFFT=256, tres=1./3., fres=2)
    avg = spec.average(tlow=0, thigh=0.4)
    med = spec.median(flow=5.0, fhigh=6.1)
    assert avg == 2
    assert med == 6    
    
def test_compute_average_with_axis(image_3x3):
    img = image_3x3
    spec = Spectrogram(image=img, NFFT=256, tres=1./3., fres=2)
    avg = spec.average(axis=0)
    assert avg.shape == (3,)
    for i in range(3):
        assert avg[i] == (img[0,i]+img[1,i]+img[2,i])/3.
        
def test_spectrogram_has_correct_time_axis(image_3x3):
    img = image_3x3
    now = datetime.datetime.today()
    spec = Spectrogram(image=img, NFFT=256, tres=1, fres=2, timestamp=now)
    assert len(spec.taxis()) == 3
    assert spec.taxis()[0] == now
    assert spec.taxis()[1] == now + datetime.timedelta(seconds=1)
    assert spec.taxis()[2] == now + datetime.timedelta(seconds=2)   
    
@pytest.mark.test_from_signal
def test_init_spectrogram_from_a_sine_wave(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = AudioSignal(rate, sig)
    spec = Spectrogram.from_signal(signal=signal, winlen=winlen, winstep=winstep)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(2000, abs=spec.fres)

@pytest.mark.test_from_signal
def test_init_spectrogram_with_user_specified_NFFT(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = AudioSignal(rate, sig)
    spec = Spectrogram.from_signal(signal=signal, winlen=winlen, winstep=winstep, NFFT=512)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq   = np.argmax(mag[i])
        freqHz = freq * spec.fres
        assert freqHz == pytest.approx(2000, abs=spec.fres)

@pytest.mark.test_from_signal
def test_spectrogram_has_correct_NFFT(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = AudioSignal(rate, sig)
    spec = Spectrogram.from_signal(signal=signal, winlen=winlen, winstep=winstep)
    assert spec.NFFT == int(round(winlen * rate))

 
