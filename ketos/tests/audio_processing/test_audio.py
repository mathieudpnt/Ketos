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
import ketos.audio_processing.audio as aud
import numpy as np


def test_init_audio_signal():
    """Test if the audio signal has expected attribute values"""
    N = 10000
    d = np.ones(N)
    a = aud.AudioSignal(rate=1000, data=d, filename='x', offset=2., label=13)
    assert np.all(a.get_data() == d)
    assert a.rate == 1000
    assert a.filename == 'x'
    assert a.offset == 2.
    assert a.label == 13

def test_init_stacked_audio_signal():
    """Test if a stacked audio signal has expected attribut values"""
    N = 10000
    d = np.ones((N,3))
    a = aud.AudioSignal(rate=1000, data=d, filename='x', offset=2., label=13)
    assert np.all(a.get_data(1) == d[:,1])
    assert a.rate == 1000
    assert np.all(a.filename == 'x')
    assert np.all(a.offset == 2.)
    assert np.all(a.label == 13)

def test_from_wav(sine_wave_file, sine_wave):
    """Test if an audio signal can be created from a wav file"""
    a = aud.AudioSignal.from_wav(sine_wave_file)
    sig = sine_wave[1]
    assert a.length() == 3.
    assert a.rate == 44100
    assert a.filename == "sine_wave.wav"
    assert np.all(np.isclose(a.data, sig, atol=0.001))

def test_append_audio_signal(sine_audio):
    """Test if we can append an audio signal to itself"""
    audio_orig = sine_audio.deepcopy()
    sine_audio.append(sine_audio)
    assert sine_audio.length() == 2 * audio_orig.length()
    assert np.all(sine_audio.data == np.concatenate([audio_orig.data,audio_orig.data],axis=0))

def test_append_audio_signal_with_overlap(sine_audio):
    """Test if we can append an audio signal to itself"""
    audio_orig = sine_audio.deepcopy()
    sine_audio.append(sine_audio, n_smooth=100)
    assert sine_audio.length() == 2 * audio_orig.length() - 100/sine_audio.rate

def test_add_audio_signals(sine_audio):
    """Test if we can add an audio signal to itself"""
    t = sine_audio.length()
    v = np.copy(sine_audio.data)
    sine_audio.add(signal=sine_audio)
    assert pytest.approx(sine_audio.length(), t, abs=0.00001)
    assert np.all(np.abs(sine_audio.data - 2*v) < 0.00001)
    
def test_add_audio_signals_with_offset(sine_audio):
    """Test if we can add an audio signal to itself with a time offset"""
    t = sine_audio.length()
    v = np.copy(sine_audio.data)
    offset = 1.1
    sine_audio.add(signal=sine_audio, offset=offset)
    assert sine_audio.length() == t
    b = sine_audio.time_ax.bin(offset) 
    assert np.all(np.abs(sine_audio.data[:b] - v[:b]) < 0.00001)
    assert np.all(np.abs(sine_audio.data[b:] - 2 * v[b:]) < 0.00001)    

def test_add_audio_signals_with_scaling(sine_audio):
    """Test if we can add an audio signal to itself with a scaling factor"""
    t = sine_audio.length()
    v = np.copy(sine_audio.data)
    scale = 1.3
    sine_audio.add(signal=sine_audio, scale=1.3)
    assert np.all(np.abs(sine_audio.data - (1. + scale) * v) < 0.00001)



# old tests below

def test_morlet_with_default_params():
    mor = aud.AudioSignal.morlet(rate=4000, frequency=20, width=1)
    assert len(mor.data) == int(6*1*4000) # check number of samples
    assert max(mor.data) == pytest.approx(1, abs=0.01) # check max signal is 1
    assert np.argmax(mor.data) == pytest.approx(0.5*len(mor.data), abs=1) # check peak is centered
    assert mor.data[0] == pytest.approx(0, abs=0.02) # check signal is approx zero at start

def test_gaussian_noise():
    noise = aud.AudioSignal.gaussian_noise(rate=2000, sigma=2, samples=40000)
    assert noise.std() == pytest.approx(2, rel=0.05) # check standard deviation
    assert noise.average() == pytest.approx(0, abs=6*2/np.sqrt(40000)) # check mean
    assert noise.duration() == 20 # check length

def test_clip(sine_audio):
    audio = sine_audio
    segs = audio.clip(boxes=[[0.1, 0.4],[0.3, 0.7]])
    assert len(segs) == 2
    assert segs[0].duration() == pytest.approx(0.3, abs=2./audio.rate)
    assert segs[1].duration() == pytest.approx(0.4, abs=2./audio.rate)
    assert audio.duration() == pytest.approx(3.0-0.6, abs=2./audio.rate)

def test_resampled_signal_has_correct_rate(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)

    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)
    assert new_signal.rate == 22000

    new_signal = signal.copy()
    new_signal.resample(new_rate=2000)
    assert new_signal.rate == 2000

def test_resampled_signal_has_correct_length(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)

    duration = signal.duration()

    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)
    assert len(new_signal.data) == duration * new_signal.rate 

    new_signal = signal.copy()
    new_signal.resample(new_rate=2000)
    assert len(new_signal.data) == duration * new_signal.rate 

def test_resampling_preserves_signal_shape(const_wave_file):
    signal = aud.AudioSignal.from_wav(const_wave_file)
    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)

    n = min(len(signal.data), len(new_signal.data))
    for i in range(n):
        assert signal.data[i] == new_signal.data[i]

def test_resampling_preserves_signal_frequency(sine_wave_file):
    signal = aud.AudioSignal.from_wav(sine_wave_file)
    rate = signal.rate
    sig = signal.data
    y = abs(np.fft.rfft(sig))
    freq = np.argmax(y)
    freqHz = freq * rate / len(sig)
    signal = aud.AudioSignal(rate, sig)
    new_signal = signal.copy()
    new_signal.resample(new_rate=22000)
    new_y = abs(np.fft.rfft(new_signal.data))
    new_freq = np.argmax(new_y)
    new_freqHz = new_freq * new_signal.rate / len(new_signal.data)

    assert freqHz == new_freqHz

def test_signal_is_padded(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = 2*duration
    winstep = 2*duration
    signal = aud.AudioSignal(rate, sig)
    frames = signal.make_frames(winlen=winlen, winstep=winstep, zero_padding=True)
    assert frames.shape[0] == 1
    assert frames.shape[1] == 2*len(sig)
    assert frames[0, len(sig)] == 0
    assert frames[0, 2*len(sig)-1] == 0

def test_can_make_overlapping_frames(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/2
    winstep = duration/4
    signal = aud.AudioSignal(rate, sig)
    frames = signal.make_frames(winlen=winlen, winstep=winstep)
    assert frames.shape[0] == 3
    assert frames.shape[1] == len(sig)/2

def test_can_make_non_overlapping_frames(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/2
    signal = aud.AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(sig)/4

def test_first_frame_matches_original_signal(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = aud.AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 8
    for i in range(int(winlen*rate)):
        assert sig[i] == pytest.approx(frames[0,i], rel=1E-6)

def test_window_length_can_exceed_duration(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = 2 * duration
    winstep = duration
    signal = aud.AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 1

