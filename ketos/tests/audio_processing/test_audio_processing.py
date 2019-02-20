""" Unit tests for the the 'pre_processing' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
import os
import numpy as np
import scipy.signal as sg
import ketos.audio_processing.audio_processing as ap
from ketos.audio_processing.audio import AudioSignal
from ketos.audio_processing.spectrogram import Spectrogram, MagSpectrogram
import cv2

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.mark.test_to_decibel
def test_to_decibel_returns_decibels():
    x = 7
    y = ap.to_decibel(x)
    assert y == 20 * np.log10(x) 

@pytest.mark.test_to_decibel
def test_to_decibel_can_handle_arrays():
    x = np.array([7,8])
    y = ap.to_decibel(x)
    assert np.all(y == 20 * np.log10(x))

@pytest.mark.test_to_decibel
def test_to_decibel_returns_inf_if_input_is_negative():
    x = -7
    y = ap.to_decibel(x)
    assert np.ma.getmask(y) == True

@pytest.mark.test_make_frames
def test_signal_is_padded(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = 2*duration
    winstep = 2*duration
    signal = AudioSignal(rate, sig)
    frames = signal.make_frames(winlen=winlen, winstep=winstep, zero_padding=True)
    assert frames.shape[0] == 1
    assert frames.shape[1] == 2*len(sig)
    assert frames[0, len(sig)] == 0
    assert frames[0, 2*len(sig)-1] == 0

@pytest.mark.test_make_frames
def test_can_make_overlapping_frames(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/2
    winstep = duration/4
    signal = AudioSignal(rate, sig)
    frames = signal.make_frames(winlen=winlen, winstep=winstep)
    assert frames.shape[0] == 3
    assert frames.shape[1] == len(sig)/2

@pytest.mark.test_make_frames
def test_can_make_non_overlapping_frames(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/2
    signal = AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(sig)/4

@pytest.mark.test_make_frames
def test_first_frame_matches_original_signal(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 8
    for i in range(int(winlen*rate)):
        assert sig[i] == pytest.approx(frames[0,i], rel=1E-6)

@pytest.mark.test_make_frames
def test_window_length_can_exceed_duration(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = 2 * duration
    winstep = duration
    signal = AudioSignal(rate, sig)
    frames = signal.make_frames(winlen, winstep)
    assert frames.shape[0] == 1

@pytest.mark.test_normalize_spec
def test_normalized_spectrum_has_values_between_0_and_1(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    mag_norm = ap.normalize_spec(spec.image)
    for i in range(mag_norm.shape[0]):
        val = mag_norm[0,i]
        assert 0 <= val <= 1

@pytest.mark.test_crop_high_freq
def test_cropped_spectrogram_has_correct_size_and_content(sine_audio):
    spec = MagSpectrogram(audio_signal=sine_audio, winlen=0.2, winstep=0.05, NFFT=256)
    mag = spec.image
    cut = int(0.7 * mag.shape[1])
    mag_cropped = ap.crop_high_freq(mag, cut)
    assert mag_cropped.shape[1] == cut
    assert mag_cropped[0,0] == mag[0,0]

@pytest.mark.test_blur_img
def test_uniform_image_is_unchanged_by_blurring():
    img = np.ones(shape=(10,10), dtype=np.float32)
    img_median = ap.blur_image(img,5,Gaussian=False)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            assert img_median[i,j] == img[i,j]
    img_gaussian = ap.blur_image(img,9,Gaussian=True)
    np.testing.assert_array_equal(img, img_gaussian)
            
@pytest.mark.test_blur_img
def test_median_filter_can_work_with_kernel_size_greater_than_five():
    img = np.ones(shape=(10,10), dtype=np.float32)
    ap.blur_image(img,13,Gaussian=False)

@pytest.mark.test_apply_broadband_filter
def test_broadband_filter_works_as_expected_for_uniform_columns():
    img = np.array([[1,1],[2,2],[3,3]], dtype=np.float32)
    img_fil = ap.apply_broadband_filter(img)
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            assert img_fil[i,j] == 0

@pytest.mark.test_apply_broadband_filter
def test_broadband_filter_works_as_expected_for_non_uniform_columns():
    img = np.array([[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = ap.apply_broadband_filter(img)
    assert img_fil[0,0] == 0
    assert img_fil[0,1] == 0
    assert img_fil[0,2] == 0
    assert img_fil[1,0] == 0
    assert img_fil[1,1] == 0
    assert img_fil[1,2] == 9

@pytest.mark.test_apply_narrowband_filter
def test_narrowband_filter_works_as_expected_for_uniform_rows():
    img = np.array([[1,3],[1,3],[1,3],[1,3]], dtype=np.float32)
    img_fil = ap.apply_narrowband_filter(img,time_res=1,time_const=1)
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            assert img_fil[i,j] == 0

@pytest.mark.test_apply_median_filter
def test_median_filter_works_as_expected():
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = ap.apply_median_filter(img,row_factor=1,col_factor=1)
    img_res = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float32)
    np.testing.assert_array_equal(img_fil,img_res)
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = ap.apply_median_filter(img,row_factor=15,col_factor=1)
    assert img_fil[2,2] == 0
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = ap.apply_median_filter(img,row_factor=1,col_factor=15)
    assert img_fil[2,2] == 0
    
@pytest.mark.test_apply_preemphasis
def test_preemphasis_has_no_effect_if_coefficient_is_zero():
    sig = np.array([1,2,3,4,5], np.float32)
    sig_new = ap.apply_preemphasis(sig,coeff=0)
    for i in range(len(sig)):
        assert sig[i] == sig_new[i]

@pytest.mark.test_prepare_for_binary_cnn
def test_prepare_for_binary_cnn():
    n = 1
    l = 2
    specs = list()
    for i in range(n):
        t1 = (i+1) * 2
        t2 = t1 + l + 1
        img = np.ones(shape=(20,30))
        img[t1:t2,:] = 2.5
        s = Spectrogram(image=img)       
        s.annotate(labels=7,boxes=[t1, t2])
        specs.append(s)

    img_wid = 4
    framer = ap.BinaryClassFramer(specs=specs, label=7, image_width=img_wid, step_size=1, signal_width=2)
    x, y, _ = framer.get_frames()
    m = 1 + 20 - 4
    q = 4
    assert y.shape == (m*n,)
    assert x.shape == (m*n, img_wid, specs[0].image.shape[1])
    assert np.all(y[0:q] == 1)
    assert y[4] == 0
    assert np.all(x[0,2,:] == 2.5)
    assert np.all(x[1,1,:] == 2.5)

    framer = ap.BinaryClassFramer(specs=specs, label=7, image_width=img_wid, step_size=1, signal_width=2, equal_rep=True)
    x, y, _ = framer.get_frames()
    assert y.shape == (2*q,)
    assert np.sum(y) == q

@pytest.mark.test_filter_isolated_cells
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

    filtered_img = ap.filter_isolated_spots(img,struct)

    assert np.array_equal(filtered_img, expected)

