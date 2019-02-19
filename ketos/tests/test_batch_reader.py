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
from sound_classification.audio_signal import AudioSignal
import sound_classification.pre_processing as pp
import datetime
import numpy as np
import os

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")

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

@pytest.fixture
def five_time_stamped_wave_files():

    files = list()
    N = 5

    folder = path_to_assets+'/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(N):
        fname = 'empty_HMS_12_ 5_ {0}__DMY_23_ 2_84.wav'.format(i)
        full_path = os.path.join(folder, fname)
        a = AudioSignal(rate=1000, data=np.zeros(500))
        a.to_wav(full_path)
        files.append(full_path)

    yield folder

    for f in files:
        os.remove(f)

def test_init_batch_reader_with_directory(five_time_stamped_wave_files):
    folder = five_time_stamped_wave_files
    reader = BatchReader(source=folder)
    assert len(reader.files) == 5

def test_batch_reader_can_parse_date_time(five_time_stamped_wave_files):
    folder = five_time_stamped_wave_files
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    reader = BatchReader(source=folder, datetime_fmt=fmt)
    b = reader.next(700)
    assert b.begin() == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=0, microsecond=0)
    b = reader.next(600)
    assert b.begin() == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=1, microsecond=0)
    b = reader.next(300)
    assert b.begin() == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=2, microsecond=0)
    b = reader.next()
    assert b.begin() == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=2, microsecond=int(3E5))

def test_batch_reader_log_has_correct_data(five_time_stamped_wave_files):
    folder = five_time_stamped_wave_files
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    reader = BatchReader(source=folder, datetime_fmt=fmt)
    reader.next()
    log = reader.log()
    for i in range(5):
        assert log['time'][i] == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=i, microsecond=0)
        fname = 'empty_HMS_12_ 5_ {0}__DMY_23_ 2_84.wav'.format(i)
        full_path = os.path.join(folder, fname)
        assert log['file'][i] == full_path
    reader.reset()
    b = reader.next(700)
    b = reader.next(600)
    b = reader.next(300)
    b = reader.next()
    for i in range(5):
        assert log['time'][i] == datetime.datetime(year=2084, month=2, day=23, hour=12, minute=5, second=i, microsecond=0)
        fname = 'empty_HMS_12_ 5_ {0}__DMY_23_ 2_84.wav'.format(i)
        full_path = os.path.join(folder, fname)
        assert log['file'][i] == full_path


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
    n2 = len(s2.data)
    size = int((n1+n2) / 1.5)
    reader = BatchReader(source=[sine_wave_file, sawtooth_wave_file])
    b = reader.next(size)
    assert reader.finished() == False
    assert len(b.data) == size
    b = reader.next(size)
    assert reader.finished() == True
    assert len(b.data) == n1 + n2 - reader.n_smooth - size

def test_next_batch_with_two_very_short_files():
    short_file_1 = os.path.join(path_to_assets, "super_short_1.wav")
    pp.wave.write(short_file_1, rate=4000, data=np.array([1.,2.,3.]))
    short_file_2 = os.path.join(path_to_assets, "super_short_2.wav")
    pp.wave.write(short_file_2, rate=4000, data=np.array([1.,2.,3.]))
    s1 = AudioSignal.from_wav(short_file_1)
    s2 = AudioSignal.from_wav(short_file_2)
    n1 = len(s1.data)
    n2 = len(s2.data)
    reader = BatchReader(source=[short_file_1, short_file_2])
    b = reader.next()
    assert reader.finished() == True
    assert len(b.data) == n1+n2-2

def test_next_batch_with_empty_file_and_resampling():
    empty = os.path.join(path_to_assets, "empty.wav")
    pp.wave.write(empty, rate=4000, data=np.empty(0))
    s = AudioSignal.from_wav(empty)
    n = len(s.data)
    reader = BatchReader(source=[empty], rate=2000)
    b = reader.next()
    assert reader.finished() == True
    assert b is None

def test_next_batch_with_multiple_files(sine_wave_file, sawtooth_wave_file, const_wave_file):
    reader = BatchReader(source=[sine_wave_file, sawtooth_wave_file, const_wave_file])
    b = reader.next()
    s1 = AudioSignal.from_wav(sine_wave_file)
    s2 = AudioSignal.from_wav(sawtooth_wave_file)
    s3 = AudioSignal.from_wav(const_wave_file)
    assert len(b.data) == len(s1.data) + len(s2.data) + len(s3.data) - 2 * reader.n_smooth
    assert reader.finished() == True
