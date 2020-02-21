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

""" Unit tests for the 'audio.audio_provider' module within the ketos library
"""
import pytest
import numpy as np
from ketos.audio.audio_provider import AudioProvider

def test_init_audio_provider_with_folder(five_time_stamped_wave_files):
    """ Test that we can initialize an instance of the AudioProvider class from a folder"""
    provider = AudioProvider(path=five_time_stamped_wave_files, seg_dur=0.5)
    assert len(provider.files) == 5

def test_init_audio_provider_with_wav_file(sine_wave_file):
    """ Test that we can initialize an instance of the AudioProvider class 
        from a single wav file"""
    provider = AudioProvider(path=sine_wave_file, seg_dur=0.5)
    assert len(provider.files) == 1

def test_audio_provider_mag(five_time_stamped_wave_files):
    """ Test that we can use the AudioProvider class to compute MagSpectrograms""" 
    kwargs = {'rep':'MagSpectrogram','window':0.1,'step':0.02}
    provider = AudioProvider(path=five_time_stamped_wave_files, seg_dur=0.5, **kwargs)
    assert len(provider.files) == 5
    s = next(provider)
    assert s.duration() == 0.5
    s = next(provider)
    assert s.duration() == 0.5
    assert provider.file_id == 2

def test_audio_provider_dur(five_time_stamped_wave_files):
    """ Test that we can use the AudioProvider class to compute MagSpectrograms
        with durations shorter than file durations""" 
    kwargs = {'rep':'MagSpectrogram','window':0.1,'step':0.02}
    provider = AudioProvider(path=five_time_stamped_wave_files, seg_dur=0.2, **kwargs)
    assert len(provider.files) == 5
    s = next(provider)
    assert s.duration() == 0.2
    s = next(provider)
    assert s.duration() == 0.2
    s = next(provider)
    assert s.duration() == 0.2
    assert provider.file_id == 1

def test_audio_provider_overlap(five_time_stamped_wave_files):
    """ Test that we can use the AudioProvider class to compute overlapping 
        MagSpectrograms""" 
    kwargs = {'rep':'MagSpectrogram','window':0.1,'step':0.02}
    provider = AudioProvider(path=five_time_stamped_wave_files, seg_dur=0.2, seg_step=0.06, **kwargs)
    assert len(provider.files) == 5
    s = next(provider)
    assert s.duration() == 0.2
    s = next(provider)
    assert s.duration() == 0.2
    s = next(provider)
    assert s.duration() == 0.2
    assert provider.time == pytest.approx(3*0.06, abs=1e-6)
    assert provider.file_id == 0

def test_audio_provider_uniform_length(five_time_stamped_wave_files):
    """ Check that the AudioProvider always returns segments of the same length""" 
    kwargs = {'rep':'MagSpectrogram','window':0.1,'step':0.02}
    provider = AudioProvider(path=five_time_stamped_wave_files, seg_dur=0.2, **kwargs)
    assert len(provider.files) == 5
    for _ in range(10):
        s = next(provider)
        assert s.duration() == 0.2

def test_audio_provider_number_of_segments(sine_wave_file):
    """ Check that the AudioProvider computes expected number of segments""" 
    kwargs = {'rep':'MagSpectrogram','window':0.1,'step':0.01,'rate':2341}
    import librosa
    dur = librosa.core.get_duration(filename=sine_wave_file)
    # duration is an integer number of lengths
    l = 0.2
    provider = AudioProvider(path=sine_wave_file, seg_dur=l, **kwargs)
    assert len(provider.files) == 1
    N = int(dur / l)
    assert N == provider.num_segs
    # duration is *not* an integer number of lengths
    l = 0.21
    provider = AudioProvider(path=sine_wave_file, seg_dur=l, **kwargs)
    N = int(np.ceil(dur / l))
    assert N == provider.num_segs
    # loop over all segments
    for _ in range(N):
        _ = next(provider)
    # non-zero overlap
    l = 0.21
    o = 0.8*l
    provider = AudioProvider(path=sine_wave_file, seg_dur=l, seg_step=l-o, **kwargs)
    step = l - o
    N = int(np.ceil((dur-l) / step) + 1)
    assert N == provider.num_segs
    # loop over all segments
    for _ in range(N):
        _ = next(provider)