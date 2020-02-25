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

@pytest.fixture
def spectr_settings():
    j = '{"spectrogram": {"type":"MagSpectrogram", "rate": "20 kHz",\
        "window": "0.1 s", "step": "0.025 s", "bins_per_oct": "32",\
        "window_func": "hamming", "freq_min": "30Hz", "freq_max": "3000Hz",\
        "duration": "1.0s", "resample_method": "scipy"}}'
    return j

def test_parse_spectr_settings(spectr_settings):
    data = json.loads(spectr_settings)
    d = jp.parse_audio_representation(data['spectrogram'])
    assert d['rate'] == 20000
    assert d['window'] == 0.1
    assert d['step'] == 0.025
    assert d['bins_per_oct'] == 32
    assert d['window_func'] == 'hamming'
    assert d['freq_min'] == 30
    assert d['freq_max'] == 3000
    assert d['duration'] == 1.0
    assert d['resample_method'] == 'scipy'
    assert d['type'] == 'MagSpectrogram'
