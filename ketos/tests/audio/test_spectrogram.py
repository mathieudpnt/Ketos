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

""" Unit tests for the 'audio.spectrogram' module within the ketos library
"""
import pytest
import warnings
import numpy as np
import copy
import os
from ketos.audio.spectrogram import MagSpectrogram,\
    PowerSpectrogram, MelSpectrogram, Spectrogram, CQTSpectrogram
from ketos.audio.waveform import Waveform
from ketos.audio.utils.axis import LinearAxis
from ketos.audio.utils.misc import from_decibel

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_init_spec(spec_image_with_attrs):
    """Test that we can initialize an instance of the Spectrogram class"""
    img, dt, ax = spec_image_with_attrs
    spec = Spectrogram(data=img, time_res=dt, type='MagSpectrogram', freq_ax=ax)
    assert np.all(spec.get_data() == img)
    assert spec.type == 'MagSpectrogram'

def test_init_mag_spec():
    """Test that we can initialize an instance of the MagSpectrogram class
       from keyword arguments"""
    img = np.ones((20,10))
    spec = MagSpectrogram(data=img, time_res=1.0, freq_min=100, freq_res=4)
    assert np.all(spec.get_data() == img)
    assert spec.type == 'MagSpectrogram'

def test_copy_spec(spec_image_with_attrs):
    """Test that we can make a copy of spectrogram"""
    img, dt, ax = spec_image_with_attrs
    spec = Spectrogram(data=img, time_res=dt, type='MagSpectrogram', freq_ax=ax)
    spec2 = spec.deepcopy()
    assert np.all(spec.get_data() == spec2.get_data())
    spec2.data += + 1.5 #modify copied image
    spec2.time_ax.x_min += 30. #modify copied time axis
    assert np.all(spec.get_data() + 1.5 == spec2.get_data()) #check that original image was not affected
    assert spec.time_ax.min() + 30. == spec2.time_ax.min() #check that original time axis was not affected

def test_mag_spec_of_sine_wave(sine_audio):
    """Test that we can compute the magnitude spectrogram of a sine wave"""
    duration = sine_audio.duration()
    win = duration / 4
    step = duration / 10
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    assert spec.time_res() == step
    assert spec.freq_min() == 0    
    freq = np.argmax(spec.get_data(), axis=1)
    freqHz = freq * spec.freq_res()
    assert np.all(np.abs(freqHz - 2000) < spec.freq_res())

def test_power_spec_of_sine_wave(sine_audio):
    """Test that we can compute the power spectrogram of a sine wave"""
    duration = sine_audio.duration()
    win = duration / 4
    step = duration / 10
    spec = PowerSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    assert spec.time_res() == step
    assert spec.freq_min() == 0
    assert spec.type == 'PowerSpectrogram'
    freq = np.argmax(spec.get_data(), axis=1)
    freqHz = freq * spec.freq_res()
    assert np.all(np.abs(freqHz - 2000) < spec.freq_res())

def test_mel_spec_of_sine_wave(sine_audio):
    """Test that we can compute the Mel spectrogram of a sine wave"""    
    duration = sine_audio.duration()
    win = duration / 4
    step = duration / 10
    spec = MelSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    assert spec.time_res() == step
    assert spec.freq_min() == 0    

def test_cqt_spec_of_sine_wave(sine_audio):
    """Test that we can compute the CQT spectrogram of a sine wave"""    
    duration = sine_audio.duration()
    step = duration / 10
    spec = CQTSpectrogram.from_waveform(audio=sine_audio, step=step, bins_per_oct=64, freq_min=1, freq_max=4000)
    assert spec.freq_min() == 1
    freq = np.argmax(spec.get_data(), axis=1)
    freqHz = spec.freq_ax.low_edge(freq)
    assert np.all(np.abs(freqHz - 2000) < 2 * spec.freq_ax.bin_width(freq))
    
def test_add_preserves_shape(sine_audio):
    """Test that when we add a spectrogram the shape of the present instance is preserved"""
    spec1 = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    orig_shape = spec1.get_data().shape
    spec2 = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    spec2.crop(start=1.0, end=2.5, freq_min=1000, freq_max=4000)
    spec1.add(spec2)
    assert spec1.get_data().shape == orig_shape

def test_add(sine_audio):
    """Test that when we add two spectrograms, we get the expected result"""
    spec1 = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    spec2 = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    spec12 = spec1.add(spec2, offset=1.0, scale=1.3, make_copy=True)
    bx = spec12.time_ax.bin(1.0)
    assert np.all(np.abs(spec12.get_data()[:bx] - spec2.get_data()[:bx]) < 0.001) # values before t=1.0 s are unchanged
    sum_spec = spec1.get_data()[bx:] + 1.3 * spec2.get_data()[:spec1.get_data().shape[0]-bx]
    assert np.all(np.abs(spec12.get_data()[bx:] - sum_spec) < 0.001) # values outside addition region have changed

def test_cropped_mag_spec_has_correct_frequency_axis_range(sine_audio):
    """Test that when we crop a spectrogram along the frequency axis, we get the correct range"""
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    spec.crop(freq_max=4000)
    assert np.abs(spec.freq_max() - 4000) < 2*spec.freq_res()
    spec.crop(freq_min=1000)
    assert np.abs(spec.freq_min() - 1000) < 2*spec.freq_res()

def test_blur_time_axis():
    """Test that blurring along time axis gives expected results"""
    img = np.zeros((21,21))
    img[10,10] = 1
    ax = ax = LinearAxis(bins=img.shape[1], extent=(0., 21.), label='Frequency (Hz)')
    spec = Spectrogram(data=img, time_res=1, type='MagSpectrogram', freq_ax=ax)
    sig = 2.0
    spec.blur(sigma_time=sig, sigma_freq=0.01)
    xy = spec.get_data() / np.max(spec.get_data())
    x = xy[:,10]
    assert x[10] == pytest.approx(1, rel=0.001)
    assert x[9] == pytest.approx(np.exp(-pow(1,2)/(2.*pow(sig,2))), rel=0.001)
    assert x[8] == pytest.approx(np.exp(-pow(2,2)/(2.*pow(sig,2))), rel=0.001)    
    assert xy[10,9] == pytest.approx(0, abs=0.001) 

def test_blur_freq_axis():
    """Test that blurring along frequency axis gives expected results"""
    img = np.zeros((21,21))
    img[10,10] = 1
    ax = ax = LinearAxis(bins=img.shape[1], extent=(0., 21.), label='Frequency (Hz)')
    spec = Spectrogram(data=img, time_res=1, type='MagSpectrogram', freq_ax=ax)
    sig = 4.2
    spec.blur(sigma_time=0.01, sigma_freq=sig)
    xy = spec.get_data() / np.max(spec.get_data())
    y = xy[10,:]
    assert y[10] == pytest.approx(1, rel=0.001)
    assert y[9] == pytest.approx(np.exp(-pow(1,2)/(2.*pow(sig,2))), rel=0.001)
    assert y[8] == pytest.approx(np.exp(-pow(2,2)/(2.*pow(sig,2))), rel=0.001)    
    assert xy[9,10] == pytest.approx(0, abs=0.001) 

def test_recover_waveform(sine_audio):
    """Test that the recovered waveform has the correct sampling rate"""
    sine_audio.resample(new_rate=16000)
    duration = sine_audio.duration()
    win = duration / 4
    step = duration / 10
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    audio = spec.recover_waveform(num_iters=10, phase_angle=0)
    assert audio.rate == sine_audio.rate

def test_recover_waveform_after_time_crop(sine_audio):
    """Test that the recovered waveform from a time-cropped spectrogram has the correct sampling rate"""
    sine_audio.resample(new_rate=16000)
    win = 0.2
    step = 0.02
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    spec.crop(start=0.4, end=2.7)
    audio = spec.recover_waveform(num_iters=10, phase_angle=0)
    assert audio.rate == pytest.approx(sine_audio.rate, abs=0.1)

def test_recover_waveform_after_freq_crop(sine_audio):
    """Test that the recovered waveform from a frequency-cropped spectrogram has the correct sampling rate"""
    sine_audio.resample(new_rate=16000)
    win = 0.2
    step = 0.02
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=win, step=step)
    spec.crop(freq_min=200, freq_max=2300)
    audio = spec.recover_waveform(num_iters=10, phase_angle=0)
    assert audio.rate == pytest.approx(2*2300, abs=0.5)

def test_recover_waveform_with_phase():
    """ Test that the recovered waveform matches the original waveform
        if the appropriate complex phase angle is used"""
    wf = Waveform.morlet(rate=20000, frequency=100., width=0.1, displacement=0.3/100., samples=20000)
    duration = wf.duration()
    win = duration / 20
    step = duration / 100
    spec = MagSpectrogram.from_waveform(audio=wf, window=win, step=step, compute_phase=True)
    wf_r = spec.recover_waveform(num_iters=25)
    assert wf_r.rate == wf.rate
    assert wf_r.duration() == wf.duration()
    y = wf.get_data()
    yr = wf_r.get_data()
    assert y.shape == y.shape
    assert np.all(np.abs(yr - y) < 1e-2 * np.max(np.abs(y))) #agree within 1% of max value
    
def test_mag_from_wav(sine_wave_file):
    # duration is even integer multiply of step size
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02)
    assert spec.time_res() == 0.02
    assert spec.duration() == 3.0
    # duration is not integer even multiply of step size, but adjust duration automatically
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.01)
    assert spec.time_res() == pytest.approx(0.01, abs=0.001)
    assert spec.duration() == pytest.approx(3.0, abs=0.01)
    # segment is empty returns empty spectrogram
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.01, offset=4.0)
        # Verify some things about the warning
        assert len(w) == 2
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Empty spectrogram returned" in str(w[-1].message)
        # Verify some things about the spectrogram
        spec.get_data().shape == (0,0)

    # duration can be less than full length
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02, duration=2.14)
    assert spec.time_res() == 0.02
    assert spec.duration() == 2.14
    # specify both offset and duration
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02, offset=0.13, duration=2.14)
    assert spec.time_res() == 0.02
    assert spec.duration() == 2.14
    assert spec.offset == 0.13
    # check file name
    assert spec.filename == 'sine_wave.wav'
    # normalize waveform
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02, normalize_wav=False, offset=0.2, duration=0.6)
    spec_norm = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02, normalize_wav=True, offset=0.2, duration=0.6)
    d1 = from_decibel(spec.get_data())
    d2 = from_decibel(spec_norm.get_data()) / np.sqrt(2)
    assert np.all(np.isclose(np.mean(d1), np.mean(d2), rtol=2e-2))

def test_mag_from_wav_time_shift():
    """ Check that spectrogram transforms properly under time translation """
    fname = os.path.join(path_to_assets, 'grunt1.wav')
    spec0 = MagSpectrogram.from_wav(fname, window=0.1, step=0.02, offset=0.00, duration=0.4, freq_max=800)
    spec1 = MagSpectrogram.from_wav(fname, window=0.1, step=0.02, offset=0.12, duration=0.4, freq_max=800)
    n_shift = int(0.12 / 0.02)
    assert np.all(np.isclose(spec0.get_data()[n_shift:], spec1.get_data()[:-n_shift], rtol=1e-6))

def test_mag_from_wav_id(sine_wave_file):
    """ Test that mag spectrogram created with from_wav method 
        has expected filename attribute"""
    # duration is even integer multiply of step size
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02)
    assert spec.filename == 'sine_wave.wav'
    spec = MagSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02, id='test/audio.wav')
    assert spec.filename == 'test/audio.wav'

def test_cqt_from_wav(sine_wave_file):
    # zero offset
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32)
    assert spec.duration() == pytest.approx(3.0, abs=0.01)
    # non-zero offset
    offset = 1.0
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32, offset=1.0)
    assert spec.offset == offset
    assert spec.duration() == pytest.approx(3.0 - offset, abs=0.01)
    # duration is less than segment length
    duration = 1.1
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32, duration=duration)
    assert spec.duration() == pytest.approx(duration, abs=0.01)
    # step size is not divisor of duration
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.017, freq_min=1, freq_max=300, bins_per_oct=32)
    assert spec.duration() == pytest.approx(3.0, abs=0.02)
    # normalize waveform
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32, normalize_wav=False)
    spec_norm = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32, normalize_wav=True)
    d1 = from_decibel(spec.get_data())
    d2 = from_decibel(spec_norm.get_data()) / np.sqrt(2)
    assert np.all(np.isclose(np.mean(d1), np.mean(d2), rtol=2e-2))

def test_cqt_from_wav_id(sine_wave_file):
    """ Test that cqt spectrogram created with from_wav method 
        has expected filename attribute"""
    # duration is even integer multiply of step size
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32)
    assert spec.filename == 'sine_wave.wav'
    spec = CQTSpectrogram.from_wav(sine_wave_file, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32, id='test/audio.wav')
    assert spec.filename == 'test/audio.wav'

def test_mel_from_wav(sine_wave_file):
    spec = MelSpectrogram.from_wav(sine_wave_file, window=0.2, step=0.02)
    assert spec.time_res() == 0.02

def test_resize_mag_spec_with_shape(sine_audio):
    """Test that when we resize a magnitude spectrogram using the shape argument, we get 
       spectrogram with the expected shape and resolution"""
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    new_spec = spec.deepcopy()
    new_spec.resize(shape=(20,300))
    assert new_spec.data.shape == (20,300)
    assert new_spec.time_res() == spec.time_res() * spec.data.shape[0] / new_spec.data.shape[0]
    assert new_spec.freq_res() == spec.freq_res() * spec.data.shape[1] / new_spec.data.shape[1]

def test_resize_mag_spec_with_time_res(sine_audio):
    """Test that when we resize a magnitude spectrogram using the time_res argument, we get 
       spectrogram with the expected shape and resolution"""
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05)
    new_spec = spec.deepcopy()
    new_spec.resize(time_res=0.2)
    assert np.abs(new_spec.time_res() - 0.2) < 0.001
    assert new_spec.freq_res() == spec.freq_res()
    n_bins = int(spec.data.shape[0] * spec.time_res() / new_spec.time_res())
    assert new_spec.data.shape == (n_bins, spec.data.shape[1])

def test_resize_mag_spec_complex_phase(sine_audio):
    """Test that when we resize a magnitude spectrogram with a complex phase angle"""
    spec = MagSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.05, compute_phase=True)
    new_spec = spec.deepcopy()
    new_spec.resize(shape=(20,300))
    assert new_spec.data.shape == (20,300,2)
    assert new_spec.time_res() == spec.time_res() * spec.data.shape[0] / new_spec.data.shape[0]
    assert new_spec.freq_res() == spec.freq_res() * spec.data.shape[1] / new_spec.data.shape[1]

def test_resize_cqt_spec(sine_audio):
    """Test that when we resize a cqt spectrogram"""
    spec = CQTSpectrogram.from_waveform(audio=sine_audio, step=0.01, freq_min=1, freq_max=300, bins_per_oct=32)
    new_spec = spec.deepcopy()
    new_spec.resize(shape=(40,100))
    assert new_spec.data.shape == (40,100)

def test_resize_mel_spec(sine_audio):
    """Test that when we resize a mel spectrogram"""
    spec = MelSpectrogram.from_waveform(audio=sine_audio, window=0.2, step=0.02)
    new_spec = spec.deepcopy()
    new_spec.resize(shape=(6,10))
    assert new_spec.data.shape == (6,10)

def test_infer_shape_mag_spec(sine_wave_file):
    """Test that we can infer the shape of a magnitude spectrogram"""
    kwargs = {'window':0.2, 'step':0.05}
    # without duration and rate the shape cannot be inferred
    assert MagSpectrogram.infer_shape(**kwargs) == None
    # when we include these parameters, the shape can be inferred
    kwargs['duration'] = 0.8
    kwargs['rate'] = 12000
    spec = MagSpectrogram.from_wav(path=sine_wave_file, **kwargs)
    assert MagSpectrogram.infer_shape(**kwargs) == spec.get_data().shape
    # if we cut on frequency, the inferred shape is still okay
    kwargs['freq_min'] = 300
    kwargs['freq_max'] = 4000
    spec = MagSpectrogram.from_wav(path=sine_wave_file, **kwargs)
    assert MagSpectrogram.infer_shape(**kwargs) == spec.get_data().shape

def test_infer_shape_cqt_spec(sine_wave_file):
    """Test that we can infer the shape of a CQT spectrogram"""
    kwargs = {'step':0.05, 'bins_per_oct':8, 'duration':0.72, 'rate':8000}
    spec = CQTSpectrogram.from_wav(path=sine_wave_file, **kwargs)
    assert CQTSpectrogram.infer_shape(**kwargs) == spec.get_data().shape

def test_infer_shape_mel_spec(sine_wave_file):
    """Test that we can infer the shape of a Mel spectrogram"""
    kwargs = {'window':0.2, 'step':0.05, 'duration':7.2, 'rate':8000}
    spec = MelSpectrogram.from_wav(path=sine_wave_file, **kwargs)
    assert MelSpectrogram.infer_shape(**kwargs) == spec.get_data().shape
