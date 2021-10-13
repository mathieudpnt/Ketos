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

""" Unit tests for the 'parsing' module within the ketos library
"""
import pytest
import json
import ketos.data_handling.parsing as jp

def test_parse_audio_representation(spectr_settings):
    data = json.loads(spectr_settings)
    d = jp.parse_audio_representation(data['spectrogram'])
    assert d['rate'] == 20000
    assert d['window'] == 0.1
    assert d['step'] == 0.025
    assert d['window_func'] == 'hamming'
    assert d['freq_min'] == 30
    assert d['freq_max'] == 3000
    assert d['duration'] == 1.0
    assert d['resample_method'] == 'scipy'
    assert d['type'] == 'MagSpectrogram'
    assert not d['normalize_wav']
    assert d['transforms'] == [{"name":"enhance_signal", "enhancement":1.0}, {"name":"adjust_range", "range":(0,1)}]
    assert d['waveform_transforms'] == [{"name":"add_gaussian_noise", "sigma":0.2}]
    assert d['decibel']
