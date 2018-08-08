""" Unit tests for the the 'spectrogram' class in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
import numpy as np
from sound_classification.spectrogram import Spectrogram
from sound_classification.json_parsing import Interval



def test_init_spectrogram_with_2x2_image(image_2x2):
    img = image_2x2
    NFFT = 256
    seconds = 1
    freq_res = 0.2
    spec = Spectrogram(image=img, NFFT=NFFT, seconds=seconds, freq_res=freq_res)
    assert np.array_equal(spec.image, img)
    assert spec.NFFT == NFFT
    assert spec.seconds == seconds
    assert spec.freq_res == freq_res
    assert spec.freq_min == 0
    assert spec.freq_max() == freq_res * img.shape[1]

def test_cropped_spectrogram_has_correct_size(image_ones_10x10):
    img = image_ones_10x10
    NFFT = 256
    seconds = 1
    freq_res = 2
    spec = Spectrogram(image=img, NFFT=NFFT, seconds=seconds, freq_res=freq_res)
    w = Interval(1.0, 5.0) 
    spec_crop = spec.crop_freq(w)
    img_crop = spec_crop.image
    assert img_crop.shape == (10, 2)
    w = Interval(1.0, 6.0) 
    spec_crop = spec.crop_freq(w)
    img_crop = spec_crop.image
    assert img_crop.shape == (10, 3)
    w = Interval(6.0, 8.0) 
    spec_crop = spec.crop_freq(w)
    img_crop = spec_crop.image
    assert img_crop.shape == (10, 1)

def test_cropped_spectrogram_has_correct_position(image_zeros_and_ones_10x10):
    img = image_zeros_and_ones_10x10
    NFFT = 256
    seconds = 1
    freq_res = 2
    spec = Spectrogram(image=img, NFFT=NFFT, seconds=seconds, freq_res=freq_res)
    w = Interval(8.0, 12.0) 
    spec_crop = spec.crop_freq(w)
    img_crop = spec_crop.image
    assert img_crop.shape == (10, 2)
    for i in range(10):
        assert img_crop[i,0] == 0
        assert img_crop[i,1] == 1

def test_compute_average_and_median(image_2x2):
    img = image_2x2
    NFFT = 256
    seconds = 1
    freq_res = 2
    spec = Spectrogram(image=img, NFFT=NFFT, seconds=seconds, freq_res=freq_res)
    avg = spec.average()
    assert avg == np.average(img)
    assert avg == np.median(img)
