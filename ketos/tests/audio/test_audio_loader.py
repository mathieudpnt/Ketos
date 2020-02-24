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

""" Unit tests for the 'audio.audio_loader' module within the ketos library
"""
import pytest
import numpy as np
import pandas as pd
from ketos.audio.audio_loader import AudioSequenceLoader, AudioSelectionLoader
from ketos.data_handling.selection_table import use_multi_indexing
from ketos.data_handling.data_handling import find_wave_files

def test_init_audio_seq_loader_with_folder(five_time_stamped_wave_files):
    """ Test that we can initialize an instance of the AudioSequenceLoader class from a folder"""
    loader = AudioSequenceLoader(path=five_time_stamped_wave_files, window=0.5)
    assert len(loader.files) == 5

def test_init_audio_seq_loader_with_wav_file(sine_wave_file):
    """ Test that we can initialize an instance of the AudioSequenceLoader class 
        from a single wav file"""
    loader = AudioSequenceLoader(path=sine_wave_file, window=0.5)
    assert len(loader.files) == 1

def test_audio_seq_loader_mag(five_time_stamped_wave_files):
    """ Test that we can use the AudioSequenceLoader class to compute MagSpectrograms""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.02}
    loader = AudioSequenceLoader(path=five_time_stamped_wave_files, window=0.5, repres=rep)
    assert len(loader.files) == 5
    s = next(loader)
    assert s.duration() == 0.5
    s = next(loader)
    assert s.duration() == 0.5
    assert loader.file_id == 2

def test_audio_seq_loader_dur(five_time_stamped_wave_files):
    """ Test that we can use the AudioSequenceLoader class to compute MagSpectrograms
        with durations shorter than file durations""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.02}
    loader = AudioSequenceLoader(path=five_time_stamped_wave_files, window=0.2, repres=rep)
    assert len(loader.files) == 5
    s = next(loader)
    assert s.duration() == 0.2
    s = next(loader)
    assert s.duration() == 0.2
    s = next(loader)
    assert s.duration() == 0.2
    assert loader.file_id == 1

def test_audio_seq_loader_overlap(five_time_stamped_wave_files):
    """ Test that we can use the AudioSequenceLoader class to compute overlapping 
        MagSpectrograms""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.02}
    loader = AudioSequenceLoader(path=five_time_stamped_wave_files, window=0.2, step=0.06, repres=rep)
    assert len(loader.files) == 5
    s = next(loader)
    assert s.duration() == 0.2
    s = next(loader)
    assert s.duration() == 0.2
    s = next(loader)
    assert s.duration() == 0.2
    assert loader.time == pytest.approx(3*0.06, abs=1e-6)
    assert loader.file_id == 0

def test_audio_seq_loader_uniform_length(five_time_stamped_wave_files):
    """ Check that the AudioSequenceLoader always returns segments of the same length""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.02}
    loader = AudioSequenceLoader(path=five_time_stamped_wave_files, window=0.2, repres=rep)
    assert len(loader.files) == 5
    for _ in range(10):
        s = next(loader)
        assert s.duration() == 0.2

def test_audio_seq_loader_number_of_segments(sine_wave_file):
    """ Check that the AudioSequenceLoader computes expected number of segments""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.01,'rate':2341}
    import librosa
    dur = librosa.core.get_duration(filename=sine_wave_file)
    # duration is an integer number of lengths
    l = 0.2
    loader = AudioSequenceLoader(path=sine_wave_file, window=l, repres=rep)
    assert len(loader.files) == 1
    N = int(dur / l)
    assert N == loader.num_segs
    # duration is *not* an integer number of lengths
    l = 0.21
    loader = AudioSequenceLoader(path=sine_wave_file, window=l, repres=rep)
    N = int(np.ceil(dur / l))
    assert N == loader.num_segs
    # loop over all segments
    for _ in range(N):
        _ = next(loader)
    # non-zero overlap
    l = 0.21
    o = 0.8*l
    loader = AudioSequenceLoader(path=sine_wave_file, window=l, step=l-o, repres=rep)
    step = l - o
    N = int(np.ceil((dur-l) / step) + 1)
    assert N == loader.num_segs
    # loop over all segments
    for _ in range(N):
        _ = next(loader)

def test_audio_select_loader_mag(five_time_stamped_wave_files):
    """ Test that we can use the AudioSelectionLoader class to compute MagSpectrograms""" 
    rep = {'type':'MagSpectrogram','window':0.1,'step':0.02}
    # create a selection table
    files = find_wave_files(path=five_time_stamped_wave_files, fullpath=False, subdirs=True)
    sel = pd.DataFrame({'filename':[files[0],files[1]],'start':[0.10,0.12],'end':[0.46,0.42]})
    sel = use_multi_indexing(sel, 'sel_id')
    # init loader
    loader = AudioSelectionLoader(path=five_time_stamped_wave_files, selections=sel, repres=rep)
    assert len(loader.files) == 5
    s = next(loader)
    assert s.duration() == pytest.approx(0.36, abs=1e-6)
    s = next(loader)
    assert s.duration() == pytest.approx(0.30, abs=1e-6)
