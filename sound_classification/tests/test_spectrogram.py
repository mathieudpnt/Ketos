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
    duration = 1
    fres = 0.2
    spec = Spectrogram(image=img, NFFT=NFFT, duration=duration, fres=fres)
    assert np.array_equal(spec.image, img)
    assert spec.NFFT == NFFT
    assert spec.duration() == duration
    assert spec.fres == fres
    assert spec.fmin == 0
    assert spec.fmax() == fres * img.shape[1]
    assert spec.tres == 0.5

def test_cropped_spectrogram_has_correct_size(image_ones_10x10):

    spec = Spectrogram(image=image_ones_10x10, NFFT=256, duration=1, fres=2)

    spec_crop = Spectrogram.cropped(spec, flow=1.0, fhigh=5.0)
    assert spec_crop.shape() == (10, 2)

    spec_crop = Spectrogram.cropped(spec, flow=1.0, fhigh=6.0)
    assert spec_crop.shape() == (10, 3)
    
    spec_crop = Spectrogram.cropped(spec, flow=6.0, fhigh=8.0)
    assert spec_crop.shape() == (10, 1)

    spec_crop = Spectrogram.cropped(spec, tlow=0.0, thigh=0.5, flow=1.0, fhigh=6.0)
    assert spec_crop.shape() == (5, 3)


def test_cropped_spectrogram_has_correct_position(image_zeros_and_ones_10x10):

    spec = Spectrogram(image=image_zeros_and_ones_10x10, NFFT=256, duration=1, fres=2)

    spec_crop = Spectrogram.cropped(spec, flow=8.0, fhigh=12.0)

    assert spec_crop.shape() == (10, 2)

    for i in range(10):
        assert spec_crop.image[i,0] == 0
        assert spec_crop.image[i,1] == 1


def test_compute_average_and_median(image_2x2):
    img = image_2x2
    NFFT = 256
    duration = 1
    fres = 2
    spec = Spectrogram(image=img, NFFT=NFFT, duration=duration, fres=fres)
    avg = spec.average()
    assert avg == np.average(img)
    assert avg == np.median(img)
