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

""" Unit tests for the 'gammatone' module within the ketos library
"""
import pytest
from ketos.audio.gammatone import GammatoneFilterBank, AuralFeatures
import matplotlib.pyplot as plt
import numpy as np
import warnings


def test_init_from_wav():
    """Test if we can initialize a Gammatone Filter Bank from a wav file"""
    # load gammatone filter bank from wav file
    gfb = GammatoneFilterBank.from_wav('ketos/tests/assets/grunt1.wav', num_chan=20, freq_min=10, rate=1000)

def test_shape():
    """Check that the Gammatone Filter Bank has the expected dimensions"""
    # load gammatone filter bank from wav file
    gfb = GammatoneFilterBank.from_wav('ketos/tests/assets/grunt1.wav', num_chan=20, freq_min=10, rate=1000)
    # check that the filter bank frequencies are as expected
    assert len(gfb.freqs) == 20
    assert np.abs(np.min(gfb.freqs) - 10) < 1e-6
    # check that the data object has correct y-axis dimension
    assert gfb.get_data().shape[1] == 20

def test_annotate_plot():
    """Test that we can annotate and plot"""
    # load gammatone filter bank from wav file
    gfb = GammatoneFilterBank.from_wav('ketos/tests/assets/grunt1.wav', num_chan=20, freq_min=10, rate=1000)
    # check that we can annotate
    gfb.annotate(start=1.2, end=1.6, freq_min=70, freq_max=600, label=1)
    # check that we can plot
    fig = gfb.plot(filter_id=13, show_annot=True, show_envelope=True)
    plt.close(fig)

def test_init_aural_features_from_wav():
    """Test if we can initialize an Aural Features object from a wav file"""
    try:
        af = AuralFeatures.from_wav('ketos/tests/assets/grunt1.wav')
        assert af.get_data().shape == (46,)
        assert np.all(~np.isnan(af.get_data()))
    except:
        with pytest.raises(ImportError):
            af = AuralFeatures.from_wav('ketos/tests/assets/grunt1.wav')      
