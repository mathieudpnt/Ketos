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

""" Unit tests for the 'audio.utils.filter' module within the ketos library
"""
import pytest
import unittest
import os
import numpy as np
import scipy.signal as sg
import scipy.ndimage as ndimage
import ketos.audio.utils.filter as fir

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_enhance_signal():
    #create a toy image
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    x,y = np.meshgrid(x,y)
    img = np.exp(-(x**2+y**2)/(2*0.5**2)) #symmetrical Gaussian 
    img += 0.2 * np.random.rand(100,100)  #add some noise
    img_enh = fir.enhance_signal(img, enhancement=3.0)
    idx = np.nonzero(img > np.median(img)+np.std(img)) #pixels with intensity > median + std dev
    assert np.all(img_enh[idx] > 0.5 * img[idx])
    idx = np.nonzero(img < np.median(img)+np.std(img)) #pixels with intensity < median + std dev
    assert np.all(img_enh[idx] < 0.5 * img[idx])

def test_uniform_image_is_unchanged_by_blurring():
    img = np.ones(shape=(10,10), dtype=np.float32)
    img_median = fir.blur_image(img,5,gaussian=False)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            assert img_median[i,j] == img[i,j]
    img_gaussian = fir.blur_image(img,9,gaussian=True)
    np.testing.assert_array_equal(img, img_gaussian)
            
def test_median_filter_can_work_with_kernel_size_greater_than_five():
    img = np.ones(shape=(10,10), dtype=np.float32)
    fir.blur_image(img,13,gaussian=False)

def test_median_filter_works_as_expected():
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = fir.apply_median_filter(img,row_factor=1,col_factor=1)
    img_res = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float32)
    np.testing.assert_array_equal(img_fil,img_res)
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = fir.apply_median_filter(img,row_factor=15,col_factor=1)
    assert img_fil[2,2] == 0
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = fir.apply_median_filter(img,row_factor=1,col_factor=15)
    assert img_fil[2,2] == 0
    
def test_preemphasis_has_no_effect_if_coefficient_is_zero():
    sig = np.array([1,2,3,4,5], np.float32)
    sig_new = fir.apply_preemphasis(sig,coeff=0)
    for i in range(len(sig)):
        assert sig[i] == sig_new[i]

def test_filter_isolated_spots_removes_single_pixels():
    img = np.array([[0,0,1,1,0,0],
                    [0,0,0,1,0,0],
                    [0,1,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,1,0,0]])
    
    expected = np.array([[0,0,1,1,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
    
    #Struct defines the relationship between a pixel and its neighbors.
    #If a pixel complies with this relationship, it is not removed
    #in this case, if the pixel has any neighbors, it will not be removed.
    struct=np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])

    filtered_img = fir.filter_isolated_spots(img,struct)

    assert np.array_equal(filtered_img, expected)

