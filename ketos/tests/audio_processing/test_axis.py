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

""" Unit tests for the 'axis' module within the ketos library
"""
import os
import pytest
import numpy as np
from ketos.audio_processing.axis import LinearAxis, CQTAxis

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

def test_linear_axis():
    ax = LinearAxis(bins=200, extent=(0.,100.))
    #Get a single bin
    b = ax.bin(0.6)
    assert b == 1
    #Get several bin numbes in one call 
    b = ax.bin([0.6,11.1])
    assert np.all(b == [1,22])
    #Get bin number for values at bin edges
    b = ax.bin([0.0,0.5,1.0,100.])
    assert np.all(b == [0,1,2,199])
    #Note that when the value sits between two bins, 
    #the higher bin number is returned, expect if the 
    #value sits at the upper edge of the last bin, in 
    #which case the lower bin number (i.e. the last bin)
    #is returned.  

    #Get bin numbers outside the axis range
    b = ax.bin([-2.1, 100.1])
    assert np.all(b == [-5,200])
    b = ax.bin([-2.1, 100.1], truncate=True)
    print(b)
    assert np.all(b == [0,199])
