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
import sound_classification.pre_processing as pp
from sound_classification.audio_signal import AudioSignal
import cv2

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.mark.test_to_decibel
def test_to_decibel_returns_decibels():
    x = 7
    y = pp.to_decibel(x)
    assert y == 20 * np.log10(x) 

@pytest.mark.test_to_decibel
def test_to_decibel_can_handle_arrays():
    x = np.array([7,8])
    y = pp.to_decibel(x)
    assert np.all(y == 20 * np.log10(x))

@pytest.mark.test_to_decibel
def test_to_decibel_throws_assertion_error_if_input_is_negative():
    x = -7
    with pytest.raises(AssertionError):
        pp.to_decibel(x)

@pytest.mark.test_resample
def test_resampled_signal_has_correct_rate(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)
    signal = AudioSignal(rate, sig)

    new_signal = pp.resample(signal=signal, new_rate=22000)
    assert new_signal.rate == 22000

    new_signal = pp.resample(signal=signal, new_rate=2000)
    assert new_signal.rate == 2000

    tmp_file = os.path.join(path_to_assets,"tmp_sig.wav")
    r = int(new_signal.rate)
    d = new_signal.data.astype(dtype=np.int16)
    pp.wave.write(filename=tmp_file, rate=r, data=d)
    read_rate, _ = pp.wave.read(tmp_file)

    assert read_rate == new_signal.rate

@pytest.mark.test_resample
def test_resampled_signal_has_correct_length(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)
    signal = pp.AudioSignal(rate, sig)

    duration = len(sig) / rate

    new_signal = pp.resample(signal=signal, new_rate=22000)
    assert len(new_signal.data) == duration * new_signal.rate 

    new_signal = pp.resample(signal=signal, new_rate=2000)
    assert len(new_signal.data) == duration * new_signal.rate 

@pytest.mark.test_resample
def test_resampling_preserves_signal_shape(const_wave_file):
    rate, sig = pp.wave.read(const_wave_file)
    signal = pp.AudioSignal(rate, sig)
    new_signal = pp.resample(signal=signal, new_rate=22000)

    n = min(len(signal.data), len(new_signal.data))
    for i in range(n):
        assert signal.data[i] == new_signal.data[i]

@pytest.mark.test_resample
def test_resampling_preserves_signal_frequency(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)
    y = abs(np.fft.rfft(sig))
    freq = np.argmax(y)
    freqHz = freq * rate / len(sig)

    signal = pp.AudioSignal(rate, sig)
    new_signal = pp.resample(signal=signal, new_rate=22000)
    new_y = abs(np.fft.rfft(new_signal.data))
    new_freq = np.argmax(new_y)
    new_freqHz = new_freq * new_signal.rate / len(new_signal.data)

    assert freqHz == new_freqHz

@pytest.mark.test_make_frames
def test_signal_is_padded(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = 2*duration
    winstep = 2*duration
    signal = pp.AudioSignal(rate, sig)
    frames = pp.make_frames(signal=signal, winlen=winlen, winstep=winstep, zero_padding=True)
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
    signal = pp.AudioSignal(rate, sig)
    frames = pp.make_frames(signal, winlen, winstep)
    assert frames.shape[0] == 3
    assert frames.shape[1] == len(sig)/2

@pytest.mark.test_make_frames
def test_can_make_non_overlapping_frames(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/2
    signal = pp.AudioSignal(rate, sig)
    frames = pp.make_frames(signal, winlen, winstep)
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(sig)/4

@pytest.mark.test_make_frames
def test_first_frame_matches_original_signal(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    frames = pp.make_frames(signal, winlen, winstep)
    assert frames.shape[0] == 8
    for i in range(int(winlen*rate)):
        assert sig[i] == pytest.approx(frames[0,i], rel=1E-6)

@pytest.mark.test_make_magnitude_spec
def test_make_magnitude_spec_of_sine_wave_is_delta_function(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal, winlen, winstep)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq = np.argmax(mag[i])
        freqHz = freq * spec.freq_res
        assert freqHz == pytest.approx(2000, abs=spec.freq_res)

@pytest.mark.test_make_magnitude_spec
def test_user_can_set_number_of_points_for_FFT(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal=signal, winlen=winlen, winstep=winstep, hamming=True, NFFT=512)
    mag = spec.image
    for i in range(mag.shape[0]):
        freq   = np.argmax(mag[i])
        freqHz = freq * spec.freq_res
        assert freqHz == pytest.approx(2000, abs=spec.freq_res)

@pytest.mark.test_make_magnitude_spec
def test_make_magnitude_spec_returns_correct_NFFT_value(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal, winlen, winstep)
    assert spec.NFFT == int(round(winlen * rate))

@pytest.mark.test_normalize_spec
def test_normalized_spectrum_has_values_between_0_and_1(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal, winlen, winstep)
    mag_norm = pp.normalize_spec(spec.image)
    for i in range(mag_norm.shape[0]):
        val = mag_norm[0,i]
        assert 0 <= val <= 1

@pytest.mark.test_crop_high_freq
def test_cropped_spectrogram_has_correct_size_and_content(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal, winlen, winstep)
    mag = spec.image
    cut = int(0.7 * mag.shape[1])
    mag_cropped = pp.crop_high_freq(mag, cut)
    assert mag_cropped.shape[1] == cut
    assert mag_cropped[0,0] == mag[0,0]

@pytest.mark.test_blur_img
def test_uniform_image_is_unchanged_by_blurring():
    img = np.ones(shape=(10,10), dtype=np.float32)
    img_median = pp.blur_image(img,5,Gaussian=False)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            assert img_median[i,j] == img[i,j]
    img_gaussian = pp.blur_image(img,9,Gaussian=True)
    np.testing.assert_array_equal(img, img_gaussian)
            
@pytest.mark.test_blur_img
def test_median_filter_can_work_with_kernel_size_greater_than_five():
    img = np.ones(shape=(10,10), dtype=np.float32)
    pp.blur_image(img,13,Gaussian=False)

@pytest.mark.test_apply_broadband_filter
def test_broadband_filter_works_as_expected_for_uniform_columns():
    img = np.array([[1,1],[2,2],[3,3]], dtype=np.float32)
    img_fil = pp.apply_broadband_filter(img)
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            assert img_fil[i,j] == 0

@pytest.mark.test_apply_broadband_filter
def test_broadband_filter_works_as_expected_for_non_uniform_columns():
    img = np.array([[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = pp.apply_broadband_filter(img)
    assert img_fil[0,0] == 0
    assert img_fil[0,1] == 0
    assert img_fil[0,2] == 0
    assert img_fil[1,0] == 0
    assert img_fil[1,1] == 0
    assert img_fil[1,2] == 9

@pytest.mark.test_apply_narrowband_filter
def test_narrowband_filter_works_as_expected_for_uniform_rows():
    img = np.array([[1,3],[1,3],[1,3],[1,3]], dtype=np.float32)
    img_fil = pp.apply_narrowband_filter(img,time_res=1,time_const=1)
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            assert img_fil[i,j] == 0

@pytest.mark.test_apply_median_filter
def test_median_filter_works_as_expected():
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = pp.apply_median_filter(img,row_factor=1,col_factor=1)
    img_res = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float32)
    np.testing.assert_array_equal(img_fil,img_res)
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = pp.apply_median_filter(img,row_factor=15,col_factor=1)
    assert img_fil[2,2] == 0
    img = np.array([[1,1,1],[1,1,1],[1,1,10]], dtype=np.float32)
    img_fil = pp.apply_median_filter(img,row_factor=1,col_factor=15)
    assert img_fil[2,2] == 0
    
@pytest.mark.test_apply_preemphasis
def test_preemphasis_has_no_effect_if_coefficient_is_zero():
    sig = np.array([1,2,3,4,5], np.float32)
    sig_new = pp.apply_preemphasis(sig,coeff=0)
    for i in range(len(sig)):
        assert sig[i] == sig_new[i]

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

    filtered_img = pp.filter_isolated_spots(img,struct)

    assert np.array_equal(filtered_img, expected)

@pytest.mark.test_extract_mfcc_features
def test_extract_mfcc_features_from_sine_wave(sine_wave):
    rate, sig = sine_wave
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    signal = pp.AudioSignal(rate, sig)
    spec = pp.make_magnitude_spec(signal, winlen, winstep, True, 512)
    mag = spec.image
    filter_banks, mfcc = pp.extract_mfcc_features(mag, spec.NFFT, rate)
    with pytest.raises(AssertionError):
        filter_banks, mfcc = pp.extract_mfcc_features(mag, 1, rate)
    
