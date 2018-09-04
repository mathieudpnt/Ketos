""" Unit tests for the the 'batch_reader' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""


import pytest
from sound_classification.batch_reader import BatchReader
from sound_classification.audio_signal import AudioSignal
import datetime
import numpy as np


today = datetime.datetime.today()


def test_init_batch_reader_with_single_file(sine_wave_file):
    reader = BatchReader(source=sine_wave_file)
    assert len(reader.files) == 1
    assert reader.files[0][0] == sine_wave_file

def test_init_batch_reader_with_two_files(sine_wave_file, sawtooth_wave_file):
    reader = BatchReader(source=[sine_wave_file, sawtooth_wave_file])
    assert len(reader.files) == 2
    assert reader.files[0][0] == sine_wave_file
    assert reader.files[1][0] == sawtooth_wave_file

def test_next_batch_with_single_file(sine_wave_file):
    s = AudioSignal.from_wav(sine_wave_file)
    reader = BatchReader(source=sine_wave_file)
    assert reader.finished() == False
    b = reader.next()
    assert reader.finished() == True
    assert b.seconds() == s.seconds()

def test_next_batch_with_multiple_files(sine_wave_file, sawtooth_wave_file, const_wave_file):
    reader = BatchReader(source=[sine_wave_file, sawtooth_wave_file, const_wave_file])
    b = reader.next()
    s1 = AudioSignal.from_wav(sine_wave_file)
    s2 = AudioSignal.from_wav(sawtooth_wave_file)
    s3 = AudioSignal.from_wav(const_wave_file)
    assert len(b.data) == len(s1.data) + len(s2.data) + len(s3.data) - 2 * reader.n_smooth
    assert reader.finished() == True

def test_next_batch_with_two_files_and_limited_batch_size(sine_wave_file, sawtooth_wave_file):
    s1 = AudioSignal.from_wav(sine_wave_file)
    s2 = AudioSignal.from_wav(sawtooth_wave_file)
    n1 = len(s1.data)
    n2 = len(s1.data)
    size = int((n1+n2) / 1.5)
    reader = BatchReader(source=[sine_wave_file, sawtooth_wave_file])
    b = reader.next(size)
    assert reader.finished() == False
    assert len(b.data) == size
    b = reader.next(size)
    assert reader.finished() == True
    assert len(b.data) == n1 + n2 - reader.n_smooth - size

