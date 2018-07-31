""" Unit tests for the the 'json_parsing' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
import json
import sound_classification.json_parsing as jp


@pytest.fixture
def spectr_config_json_complete():
    j = '{"spectrogram": {"rate": "20 kHz", "window_size": "0.1 s", "step_size": "0.025 s", "window_function": "HAMMING"}}'
    return j

@pytest.fixture
def spectr_config_json_partial():
    j = '{"spectrogram": {"rate": "20 kHz", "window_size": "0.1 s"}}'
    return j

@pytest.fixture
def one_frequency_band_json():
    j = '{"frequency_bands": [{"name": "15.6Hz", "range": ["11.0Hz", "22.1Hz"]}]}'
    return j

@pytest.fixture
def two_frequency_bands_json():
    j = '{"frequency_bands": [{"name": "15.6Hz", "range": ["11.0Hz", "22.1Hz"]},{"name": "test", "range": ["9kHz", "10kHz"]}]}'
    return j


@pytest.mark.test_parse_spectrogram_config
def test_parse_complete_spectrogram_config(spectr_config_json_complete):
    data = json.loads(spectr_config_json_complete)
    cfg = jp.parse_spectrogram_config(data['spectrogram'])
    assert cfg.rate == 20000
    assert cfg.window_size == 0.1
    assert cfg.step_size == 0.025
    assert cfg.window_function == jp.WinFun.HAMMING

@pytest.mark.test_parse_spectrogram_config
def test_parse_partial_spectrogram_config(spectr_config_json_partial):
    data = json.loads(spectr_config_json_partial)
    cfg = jp.parse_spectrogram_config(data['spectrogram'])
    assert cfg.rate == 20000
    assert cfg.window_size == 0.1
    assert cfg.step_size == None
    assert cfg.window_function == None    

@pytest.mark.test_parse_frequency_bands
def test_parse_one_frequency_band(one_frequency_band_json):
    data = json.loads(one_frequency_band_json)
    band_name, freq_interval = jp.parse_frequency_bands(data['frequency_bands'])
    assert len(band_name) == 1
    assert band_name[0] == '15.6Hz'
    assert freq_interval[0].low == 11.0
    assert freq_interval[0].high == 22.1

@pytest.mark.test_parse_frequency_bands
def test_parse_two_frequency_bands(two_frequency_bands_json):
    data = json.loads(two_frequency_bands_json)
    band_name, freq_interval = jp.parse_frequency_bands(data['frequency_bands'])
    assert len(band_name) == 2
    assert band_name[1] == 'test'
    assert freq_interval[1].low == 9000.0
    assert freq_interval[1].high == 10000.0