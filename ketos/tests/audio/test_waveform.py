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

""" Unit tests for the 'audio' module within the ketos library
"""
import pytest
from ketos.audio.waveform import Waveform, merge, get_duration
import numpy as np
import warnings


def test_init_audio_signal():
    """Test if the audio signal has expected attribute values"""
    N = 10000
    d = np.ones(N)
    a = Waveform(rate=1000, data=d, filename='x', offset=2., label=13)
    assert np.all(a.get_data() == d)
    assert a.rate == 1000
    assert a.filename == 'x'
    assert a.offset == 2.
    assert a.label == 13

def test_from_wav(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file"""
    a = Waveform.from_wav(sine_wave_file)
    sig = sine_wave[1]
    assert a.duration() == 3.
    assert a.rate == 44100
    assert a.filename == "sine_wave.wav"
    assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_from_multiple_files(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from multiple audio files"""
    #same offset and duration
    a = Waveform.from_wav([sine_wave_file, sine_wave_file])  
    assert a.duration() == 6.0
    assert a.rate == 44100
    assert a.filename == "sine_wave.wav"
    assert np.all(np.isclose(a.data[:100], sine_wave[1][:100], atol=0.001))
    #different offsets and duration
    a = Waveform.from_wav([sine_wave_file, sine_wave_file], offset=[0.2, 0.4], duration=[1.2, 0.8])  
    assert a.duration() == 2.0
    i0 = int(a.rate * 0.2)
    assert np.all(np.isclose(a.data[:100], sine_wave[1][i0:i0+100], atol=0.001))

def test_from_wav_zero_pad(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file
        if offset + duration exceed the file length"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a = Waveform.from_wav(sine_wave_file, offset=2, duration=4, pad_mode="zero")
        # Verify some things about the warning
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Waveform padded with zeros to achieve desired length" in str(w[-1].message)
        # Verify some things about the waveform
        sig = sine_wave[1][2*44100:] #the last 1 second of the sine wave
        sig = np.concatenate([sig,np.zeros(3*44100)]) #append 3 seconds of zeros
        assert a.duration() == 4.
        assert a.rate == 44100
        assert a.filename == "sine_wave.wav"
        assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_from_wav_negative_offset(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file
        with negative offset and zero padding"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a = Waveform.from_wav(sine_wave_file, offset=-2, duration=4, pad_mode="zero")
        # Verify some things about the warning
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Waveform padded with zeros to achieve desired length" in str(w[-1].message)
        # Verify some things about the waveform
        sig = sine_wave[1][:2*44100] #first 2 seconds of the sine wave
        sig = np.concatenate([np.zeros(2*44100),sig]) #append 2 seconds of zeros
        assert a.duration() == 4.
        assert a.rate == 44100
        assert a.filename == "sine_wave.wav"
        assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_from_wav_negative_offset_pad_with_reflection(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file
        with negative offset and reflective padding"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a = Waveform.from_wav(sine_wave_file, offset=-1, duration=3, pad_mode='reflect')
        # Verify some things about the warning
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Waveform padded with its own reflection to achieve desired length" in str(w[-1].message)
        # Verify some things about the waveform
        sig = sine_wave[1][:2*44100] #first 2 seconds of the sine wave
        sig = np.concatenate([sig[44100:0:-1],sig[:]]) #pre-pend reflection
        assert a.duration() == 3.
        assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_from_wav_offset_exceeds_file_duration(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file
        if offset exceeds file length"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a = Waveform.from_wav(sine_wave_file, rate=8000, offset=5)
        # Verify some things about the warning
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Offset exceeds file duration. Empty waveform returned" in str(w[-1].message)
        # Verify some things about the waveform
        assert a.duration() == 0.
        assert a.rate == 8000
        assert a.filename == "sine_wave.wav"
        assert len(a.data) == 0

def test_from_wav_id(sine_wave_file, sine_wave):
    """ Test if an audio signal can be created from a wav file,
        with user specified ID"""
    id = 'folder/audio.wav'
    a = Waveform.from_wav(sine_wave_file, id=id)
    sig = sine_wave[1]
    assert a.duration() == 3.
    assert a.rate == 44100
    assert a.filename == id
    assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_from_wav_norm(sine_wave_file_half, sine_wave):
    """ Test if an audio signal can be created from a wav file,
        so that it has zero mean and unity standard deviation"""
    a = Waveform.from_wav(sine_wave_file_half, normalize_wav=True)
    assert np.isclose(np.mean(a.data), 0, atol=1e-12)
    assert np.isclose(np.std(a.data), 1, atol=1e-12)
    assert a.transform_log == [{'name':'normalize','mean':0,'std':1}]

def test_from_wav_path_is_None():
    """ Test if a zero waveform can be created by specifying path=None"""
    a = Waveform.from_wav(None, rate=1000, duration=3.0)
    assert np.all(np.isclose(a.data, 0, atol=1e-12))
    assert a.duration() == 3.0
    assert len(a.data) == 3000

def test_append_audio_signal(sine_audio):
    """Test if we can append an audio signal to itself"""
    audio_orig = sine_audio.deepcopy()
    sine_audio.append(sine_audio)
    assert sine_audio.duration() == 2 * audio_orig.duration()
    assert np.all(sine_audio.data == np.concatenate([audio_orig.data,audio_orig.data],axis=0))

def test_append_audio_signal_with_overlap(sine_audio):
    """Test if we can append an audio signal to itself"""
    audio_orig = sine_audio.deepcopy()
    sine_audio.append(sine_audio, n_smooth=100)
    assert sine_audio.duration() == 2 * audio_orig.duration()

def test_merge_audio_signals(sine_audio):
    """Test that we can merge multiple audio signals"""
    wf = merge([sine_audio, sine_audio, sine_audio], smooth=0.01)
    assert wf.duration() == 3 * sine_audio.duration()

def test_get_duration(sine_wave_file):
    """Test that we can predict the duration of an audio signal 
       obtained by merging several files"""
    path = [sine_wave_file, None, sine_wave_file]
    offset = [1.0, 0, -0.5]
    duration = [1.2, 0.7, None]
    d = get_duration(path=path, offset=offset, duration=duration)
    assert d == [1.2, 0.7, 3.5]

def test_add_audio_signals(sine_audio):
    """Test if we can add an audio signal to itself"""
    t = sine_audio.duration()
    v = np.copy(sine_audio.data)
    sine_audio.add(signal=sine_audio)
    assert np.abs(sine_audio.duration() - t) < 0.00001
    assert np.all(np.abs(sine_audio.data - 2*v) < 0.00001)
    
def test_add_audio_signals_with_offset(sine_audio):
    """Test if we can add an audio signal to itself with a time offset"""
    t = sine_audio.duration()
    v = np.copy(sine_audio.data)
    offset = 1.1
    sine_audio.add(signal=sine_audio, offset=offset)
    assert sine_audio.duration() == t
    b = sine_audio.time_ax.bin(offset) 
    assert np.all(np.abs(sine_audio.data[:b] - v[:b]) < 0.00001)
    assert np.all(np.abs(sine_audio.data[b:] - 2 * v[b:]) < 0.00001)    

def test_add_audio_signals_with_scaling(sine_audio):
    """Test if we can add an audio signal to itself with a scaling factor"""
    t = sine_audio.duration()
    v = np.copy(sine_audio.data)
    scale = 1.3
    sine_audio.add(signal=sine_audio, scale=1.3)
    assert np.all(np.abs(sine_audio.data - (1. + scale) * v) < 0.00001)

def test_add_morlet_on_cosine():
    cos = Waveform.cosine(rate=100, frequency=1., duration=4)
    mor = Waveform.morlet(rate=100, frequency=7., width=0.5)
    cos.add(signal=mor, offset=3.0, scale=0.5)

def test_morlet_with_default_params():
    """Test can create Morlet wavelet"""
    mor = Waveform.morlet(rate=4000, frequency=20, width=1)
    assert len(mor.data) == int(6*1*4000) # check number of samples
    assert max(mor.data) == pytest.approx(1, abs=0.01) # check max signal is 1
    assert np.argmax(mor.data) == pytest.approx(0.5*len(mor.data), abs=1) # check peak is centered
    assert mor.data[0] == pytest.approx(0, abs=0.02) # check signal is approx zero at start

def test_gaussian_noise():
    """Test can add Gaussian noise"""
    noise = Waveform.gaussian_noise(rate=2000, sigma=2, samples=40000)
    assert noise.std() == pytest.approx(2, rel=0.05) # check standard deviation
    assert noise.average() == pytest.approx(0, abs=6*2/np.sqrt(40000)) # check mean
    assert noise.duration() == 20 # check length

def test_resampled_signal_has_correct_rate(sine_wave_file):
    """Test the resampling method produces audio signal with correct rate"""
    signal = Waveform.from_wav(sine_wave_file)
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=22000)
    assert new_signal.rate == 22000
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=2000)
    assert new_signal.rate == 2000

def test_resampled_signal_has_correct_duration(sine_wave_file):
    """Test the resampling method produces audio signal with correct duration"""
    signal = Waveform.from_wav(sine_wave_file)
    duration = signal.duration()
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=22000)
    assert len(new_signal.data) == duration * new_signal.rate 
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=2000)
    assert len(new_signal.data) == duration * new_signal.rate 

def test_resampling_preserves_signal_shape(const_wave_file):
    """Test that resampling of a constant signal produces a constant signal"""
    signal = Waveform.from_wav(const_wave_file)
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=22000)
    assert np.all(np.abs(new_signal.data - np.average(signal.data)) < 0.0001)

def test_resampling_preserves_frequency_of_sine_wave(sine_wave_file):
    """Test that resampling of a sine wave produces a sine wave with the same frequency"""
    signal = Waveform.from_wav(sine_wave_file)
    rate = signal.rate
    sig = signal.data
    y = abs(np.fft.rfft(sig))
    freq = np.argmax(y)
    freqHz = freq * rate / len(sig)
    signal = Waveform(rate=rate, data=sig)
    new_signal = signal.deepcopy()
    new_signal.resample(new_rate=22000)
    new_y = abs(np.fft.rfft(new_signal.data))
    new_freq = np.argmax(new_y)
    new_freqHz = new_freq * new_signal.rate / len(new_signal.data)
    assert freqHz == new_freqHz

def test_segment():
    mor = Waveform.morlet(rate=100, frequency=5, width=0.5)
    segs = mor.segment(window=2., step=1.)
    assert segs[0].get_filename() == 'morlet'

def test_infer_shape(sine_wave_file):
    """Test that we can infer the shape of a Waveform"""
    kwargs = {'duration':17.2, 'rate':8000}
    wf = Waveform.from_wav(path=sine_wave_file, **kwargs)
    assert Waveform.infer_shape(**kwargs) == wf.get_data().shape
