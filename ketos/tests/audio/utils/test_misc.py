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

""" Unit tests for the 'audio.utils.misc' module within the ketos library
"""
import pytest
import unittest
import os
import numpy as np
import scipy.signal as sg
import ketos.audio.utils.misc as aum
from ketos.utils import morlet_func

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_pad_reflect_1d():
    x = np.random.rand(9)
    # default does nothing
    res = aum.pad_reflect(x, invert=True)
    assert np.all(res == x)
    # padding on left only
    res = aum.pad_reflect(x, pad_left=5, invert=True)
    assert np.all(res[5:] == x)
    assert np.all(np.flip(res[:5], axis=0) == 2*x[0] - x[1:6])
    # padding on both sides
    res = aum.pad_reflect(x, pad_left=5, pad_right=2, invert=True)
    assert np.all(res[5:-2] == x)
    assert np.all(np.flip(res[:5], axis=0) == 2*x[0] - x[1:6])
    assert np.all(np.flip(res[-2:], axis=0) == 2*x[-1] - x[-3:-1])
    # pad more than array size
    res = aum.pad_reflect(x, pad_left=12, invert=True)
    assert np.all(res[12:] == x)
    assert np.all(np.flip(res[4:12], axis=0) == 2*x[0] - x[1:9])
    assert np.all(res[:4] == 0)

def test_pad_reflect_2d():
    x = np.random.rand(9,14)
    # default does nothing
    res = aum.pad_reflect(x)
    assert np.all(res == x)
    # padding on left only
    res = aum.pad_reflect(x, pad_left=5, invert=True)
    assert np.all(res[5:] == x)
    assert np.all(np.flip(res[:5], axis=0) == 2*x[0] - x[1:6])
    # padding on both sides
    res = aum.pad_reflect(x, pad_left=5, pad_right=2, invert=True)
    assert np.all(res[5:-2] == x)
    assert np.all(np.flip(res[:5], axis=0) == 2*x[0] - x[1:6])
    assert np.all(np.flip(res[-2:], axis=0) == 2*x[-1] - x[-3:-1])

def test_num_samples():
    assert aum.num_samples(time=1.0, rate=10.) == 10
    assert aum.num_samples(time=1.1, rate=10.) == 11
    assert aum.num_samples(time=1.11, rate=10., even=True) == 12

def test_segment_args():
    # simple cases produces expected output
    args = aum.segment_args(rate=10., duration=8., offset=0., window=4., step=2.)
    assert args == {'win_len':40, 'step_len':20, 'num_segs':4, 'offset_len':-10}
    args = aum.segment_args(rate=1000., duration=1., offset=0., window=0.2, step=0.04)
    assert args == {'win_len':200, 'step_len':40, 'num_segs':25, 'offset_len':-80}
    # change in offset produces expected change in offset_len
    args = aum.segment_args(rate=10., duration=8., offset=3., window=4., step=2.)
    assert args == {'win_len':40, 'step_len':20, 'num_segs':4, 'offset_len':20}
    # window_len is always even
    args = aum.segment_args(rate=10., duration=8., offset=0., window=4.11, step=2.)
    assert args == {'win_len':42, 'step_len':20, 'num_segs':4, 'offset_len':-11}
    # step_len is always even
    args = aum.segment_args(rate=10., duration=8., offset=0., window=4., step=2.11)
    assert args == {'win_len':40, 'step_len':22, 'num_segs':4, 'offset_len':-9}
    # if the duration is not an integer multiple of the step size, we still get the expected output
    args = aum.segment_args(rate=10., duration=9., offset=0., window=4., step=2.)
    assert args == {'win_len':40, 'step_len':20, 'num_segs':5, 'offset_len':-10}

def test_segment_1d():
    x = np.arange(10)
    # check that segment with zero offset yields expected result
    res = aum.segment(x, win_len=4, step_len=2, num_segs=3, offset_len=0, )    
    ans = np.array([[0, 1, 2, 3],\
                    [2, 3, 4, 5],\
                    [4, 5, 6, 7]])
    assert np.all(ans == res)
    # check that segment with non-zero offset yields expected result
    res = aum.segment(x, win_len=4, step_len=2, num_segs=3, offset_len=-3)    
    ans = np.array([[ 3,  2,  1,  0],\
                    [ 1,  0,  1,  2],\
                    [ 1,  2,  3,  4]])
    assert np.all(ans == res)
    # check that segment with num_segs not specified yields expected result
    res = aum.segment(x, win_len=4, step_len=2)    
    ans = np.array([[0, 1, 2, 3],\
                    [2, 3, 4, 5],\
                    [4, 5, 6, 7],\
                    [6, 7, 8, 9]])
    assert np.all(ans == res)
    x = np.arange(11)
    res = aum.segment(x, win_len=4, step_len=2)    
    ans = np.array([[0, 1, 2, 3],\
                    [2, 3, 4, 5],\
                    [4, 5, 6, 7],\
                    [6, 7, 8, 9],\
                    [8, 9, 10, 9]])
    assert np.all(ans == res)

def test_segment_2d():
    x = [[n for _ in range(4)] for n in range(8)]
    x = np.array(x)
    # check that segment yields expected result
    res = aum.segment(x, num_segs=3, offset_len=0, win_len=3, step_len=2)    
    ans = np.array([[[0, 0, 0, 0],\
                     [1, 1, 1, 1],\
                     [2, 2, 2, 2]],\
                    [[2, 2, 2, 2],\
                     [3, 3, 3, 3],\
                     [4, 4, 4, 4]],\
                    [[4, 4, 4, 4],\
                     [5, 5, 5, 5],\
                     [6, 6, 6, 6]]])
    assert np.all(ans == res)

def test_segment_pass_args_as_dict():
    x = np.arange(10)
    d = {'win_len':4, 'step_len':2}
    res = aum.segment(x, **d)    
    ans = np.array([[0, 1, 2, 3],\
                    [2, 3, 4, 5],\
                    [4, 5, 6, 7],\
                    [6, 7, 8, 9]])
    assert np.all(ans == res)

def test_stft():
    # 1-s sinus signal with frequency of 100 Hz and 1 kHz sampling rate
    x = np.arange(1000)
    sig = np.sin(2 * np.pi * 100 * x / 1000) 
    # compute STFT with window length of 200, step length of 40, and offset of -(200-40)/2=-80
    seg_args = aum.segment_args(rate=1000., duration=1., offset=0., window=0.2, step=0.04)
    mag, freq_max, num_fft, seg_args, phase = aum.stft(x=sig, rate=1000, seg_args=seg_args)
    assert freq_max == 500.
    assert num_fft == 200
    assert seg_args == {'win_len':200, 'step_len':40, 'num_segs':25, 'offset_len':-80}
    assert np.all(np.argmax(mag[1:-1], axis=1) == int(100. / 500. * mag.shape[1])) #peak at f=100Hz (except first and last time bin)
    # compute STFT with window size of 0.2 s and step size of 0.04 s, and check that results are identical to above
    x = np.arange(1000)
    sig = np.sin(2 * np.pi * 100 * x / 1000) 
    mag_s, freq_max_s, num_fft_s, seg_args_s, phase = aum.stft(x=sig, rate=1000, window=0.2, step=0.04)
    assert freq_max_s == freq_max
    assert num_fft_s == num_fft
    assert seg_args_s == {'win_len':200, 'step_len':40, 'num_segs':mag_s.shape[0], 'offset_len':-80}
    assert np.all(mag_s == mag)

def test_to_decibel_returns_decibels():
    x = 7
    y = aum.to_decibel(x)
    assert y == 20 * np.log10(x) 

def test_to_decibel_can_handle_arrays():
    x = np.array([7,8])
    y = aum.to_decibel(x)
    assert np.all(y == 20 * np.log10(x))

def test_to_decibel_returns_inf_if_input_is_negative():
    x = -7
    y = aum.to_decibel(x)
    assert np.ma.getmask(y) == True

def test_spec2wave_sine_without_phase():
    ''' Check that we can reconstruct a sine waveform from a magnitude 
        spectrogram without considering the complex phase'''
    # 1-s sinus signal with frequency of 10 Hz and 1 kHz sampling rate
    x = np.arange(1000)
    sig = np.sin(2 * np.pi * 10 * x / 1000) 
    # compute STFT
    win_fun = 'hamming'
    seg_args = dict(win_len=100, step_len=20, offset_len=0)
    mag, freq_max, num_fft, seg_args, phase = aum.stft(x=sig, rate=1000, seg_args=seg_args, window_func=win_fun)
    img = aum.from_decibel(mag)
    # invert
    aud = aum.spec2wave(image=img, phase_angle=0, num_fft=num_fft, step_len=seg_args['step_len'], num_iters=50, window_func=win_fun)
    # check that recovered signal looks like a harmonic with frequency of 10 Hz
    assert sig.shape == aud.shape
    fft = np.fft.rfft(aud)
    assert np.argmax(fft) == int(10. / 500. * len(fft))

def test_spec2wave_morlet_with_phase():
    ''' Check that we can reconstruct a Morlet waveform from a magnitude 
        spectrogram considering also the complex phase'''
    # superposition of two Morlet wavelets
    sr = 20000
    x = np.linspace(start=0, stop=1, num=int(sr))
    y = morlet_func(time=x, frequency=100., width=0.1, displacement=0.5+0.3/100.) \
        + morlet_func(time=x, frequency=60., width=0.1, displacement=0.5+0.7/60.)
    # compute STFT, including complex phase
    win_fun = 'hamming'
    seg_args = dict(win_len=1000, step_len=200, offset_len=0)
    mag, freq_max, num_fft, seg_args, phase = aum.stft(x=y, rate=sr, seg_args=seg_args, window_func=win_fun, compute_phase=True)
    img = aum.from_decibel(mag)
    # recover waveform
    yr = aum.spec2wave(image=img, phase_angle=phase, num_fft=num_fft, step_len=seg_args['step_len'], num_iters=50, window_func=win_fun)
    # check that recovered signal matches the original within 1% of max amplitude
    assert yr.shape == y.shape
    assert np.all(np.abs(yr - y) < 1e-2 * np.max(np.abs(y)))
    # now, recover waveform without using phase 
    yr2 = aum.spec2wave(image=img, phase_angle=0, num_fft=num_fft, step_len=seg_args['step_len'], num_iters=50, window_func=win_fun)
    # check that recovered signal does not match the original
    assert np.any(np.abs(yr2 - y) > 0.1 * np.max(np.abs(y))) #deviation of more than 10%
