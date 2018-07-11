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

import cv2

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")


@pytest.fixture
def sine_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600*np.sin(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal


@pytest.fixture
def square_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600 * sg.square(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal

@pytest.fixture
def sawtooth_wave():
    sampling_rate = 44100
    frequency = 2000
    duration = 3
    x = np.arange(duration * sampling_rate)

    signal = 32600 * sg.sawtooth(2 * np.pi * frequency * x / sampling_rate) 

    return sampling_rate, signal

@pytest.fixture
def const_wave():
    sampling_rate = 44100
    duration = 3
    x = np.arange(duration * sampling_rate)
    signal = np.ones(len(x))

    return sampling_rate, signal


@pytest.fixture
def sine_wave_file(sine_wave):
    """Create a .wav with the 'sine_wave()' fixture
    
       The file is saved as tests/assets/sine_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file = os.path.join(path_to_assets, "sine_wave.wav")
    rate, sig = sine_wave
    pp.wave.write(wav_file, rate=rate, data=sig)
    
    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def square_wave_file(square_wave):
    """Create a .wav with the 'square_wave()' fixture
    
       The file is saved as tests/assets/square_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "square_wave.wav")
    rate, sig = square_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def sawtooth_wave_file(sawtooth_wave):
    """Create a .wav with the 'sawtooth_wave()' fixture
    
       The file is saved as tests/assets/sawtooth_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "sawtooth_wave.wav")
    rate, sig = sawtooth_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.fixture
def const_wave_file(const_wave):
    """Create a .wav with the 'const_wave()' fixture
    
       The file is saved as tests/assets/const_wave.wav.
       When the tests using this fixture are done, 
       the file is deleted.


       Yields:
            wav_file : str
                A string containing the path to the .wav file.
    """
    wav_file =  os.path.join(path_to_assets, "const_wave.wav")
    rate, sig = const_wave
    pp.wave.write(wav_file, rate=rate, data=sig)

    yield wav_file
    os.remove(wav_file)


@pytest.mark.test_resample
def test_resampled_signal_has_correct_rate(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)

    duration = len(sig) / rate

    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=22000)
    assert new_rate == 22000

    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=2000)
    assert new_rate == 2000

    tmp_file = os.path.join(path_to_assets,"tmp_sig.wav")
    pp.wave.write(filename=tmp_file, rate=new_rate, data=new_sig)
    read_rate, read_sig = pp.wave.read(tmp_file)

    assert read_rate == new_rate

@pytest.mark.test_resample
def test_resampled_signal_has_correct_length(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)

    duration = len(sig) / rate

    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=22000)
    assert len(new_sig) == duration * new_rate 

    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=2000)
    assert len(new_sig) == duration * new_rate 

@pytest.mark.test_resample
def test_resampling_preserves_signal_shape(const_wave_file):
    rate, sig = pp.wave.read(const_wave_file)
    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=22000)

    n = min(len(sig), len(new_sig))
    for i in range(n):
        assert sig[i] == new_sig[i]

@pytest.mark.test_resample
def test_resampling_preserves_signal_frequency(sine_wave_file):
    rate, sig = pp.wave.read(sine_wave_file)
    y = abs(np.fft.rfft(sig))
    freq = np.argmax(y)
    freqHz = freq * rate / len(sig)

    new_rate, new_sig = pp.resample(sig=sig, orig_rate=rate, new_rate=22000)
    new_y = abs(np.fft.rfft(new_sig))
    new_freq = np.argmax(new_y)
    new_freqHz = new_freq * new_rate / len(new_sig)

    assert freqHz == new_freqHz

@pytest.mark.test_make_frames
def test_signal_is_padded():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = 2*duration
    winstep = 2*duration
    frames = pp.make_frames(sig, rate, winlen, winstep)
    assert frames.shape[0] == 1
    assert frames.shape[1] == 2*len(sig)
    assert frames[0, len(sig)] == 0
    assert frames[0, 2*len(sig)-1] == 0

@pytest.mark.test_make_frames
def test_can_make_overlapping_frames():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/2
    winstep = duration/4
    frames = pp.make_frames(sig, rate, winlen, winstep)
    assert frames.shape[0] == 4
    assert frames.shape[1] == len(sig)/2

@pytest.mark.test_make_frames
def test_can_make_non_overlapping_frames():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/2
    frames = pp.make_frames(sig, rate, winlen, winstep)
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(sig)/4

@pytest.mark.test_make_frames
def test_first_frame_matches_original_signal():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    frames = pp.make_frames(sig, rate, winlen, winstep)
    assert frames.shape[0] == 10
    for i in range(int(winlen*rate)):
        assert sig[i] == frames[0,i]

@pytest.mark.test_make_magnitude_spec
def test_make_magnitude_spec_of_sine_wave_is_delta_function():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    mag, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep)
    for i in range(mag.shape[0]):
        freq   = np.argmax(mag[i])
        freqHz = freq * Hz
        assert freqHz == pytest.approx(2000, Hz)

@pytest.mark.test_make_magnitude_spec
def test_make_magnitude_spec_returns_decibels():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    mag, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep)
    mag_dB, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep, True)
    assert np.max(mag_dB[0]) == 20 * np.log10(np.max(mag[0])) 

@pytest.mark.test_make_magnitude_spec
def test_user_can_set_number_of_points_for_FFT():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    mag, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep, False, True, 512)
    for i in range(mag.shape[0]):
        freq   = np.argmax(mag[i])
        freqHz = freq * Hz
        assert freqHz == pytest.approx(2000, Hz)

@pytest.mark.test_normalize_spec
def test_normalized_spectrum_has_values_between_0_and_1():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    mag, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep)
    mag_norm = pp.normalize_spec(mag)
    for i in range(mag_norm.shape[0]):
        val = mag_norm[0,i]
        assert 0 <= val <= 1

@pytest.mark.test_crop_high_freq
def test_cropped_spectrogram_has_correct_size_and_content():
    rate, sig = sine_wave()
    duration = len(sig) / rate
    winlen = duration/4
    winstep = duration/10
    mag, Hz = pp.make_magnitude_spec(sig, rate, winlen, winstep)
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
    img_median = pp.blur_image(img,13,Gaussian=False)

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


    
