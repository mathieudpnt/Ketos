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

""" Parsing module within the ketos library

    This module provides utilities to parse various string 
    structures.
"""
import os
import json
from pint import UnitRegistry

ureg = UnitRegistry()
    

def load_audio_representation(path, name='config'):
    """ Load audio representation settings from JSON file.

        Args:
            path: str
                Path to json file
            name: str
                Heading of the relevant section of the json file

        Returns:
            d: dict
                Dictionary with the settings

        Example:
            >>> import json
            >>> from ketos.data_handling.parsing import load_audio_representation
            >>> # create json file with spectrogram settings
            >>> json_str = '{"spectrogram": {"type": "MagSpectrogram", "rate": "20 kHz", "window": "0.1 s", "step": "0.025 s", "window_func": "hamming", "freq_min": "30Hz", "freq_max": "3000Hz"}}'
            >>> path = 'ketos/tests/assets/tmp/config.py'
            >>> file = open(path, 'w')
            >>> _ = file.write(json_str)
            >>> file.close()
            >>> # load settings back from json file
            >>> settings = load_audio_representation(path=path, name='spectrogram')
            >>> print(settings)
            {'type': 'MagSpectrogram', 'rate': 20000.0, 'window': 0.1, 'step': 0.025, 'bins_per_oct': None, 'freq_min': 30, 'freq_max': 3000, 'window_func': 'hamming', 'resample_method': None}
            >>> # clean up
            >>> os.remove(path)
    """
    f = open(path, 'r')
    data = json.load(f)
    d = parse_audio_representation(data[name])
    f.close()
    return d

def parse_audio_representation(s):
    """ Parse audio representation settings for generating waveforms or spectrograms.
    
        Args:
            s: str
                Json-format string with the settings 

        Returns:
            d: dict
                Dictionary with the settings
    """
    rep_type        = parse_value(s, 'type', typ='str')
    rate            = parse_value(s, 'rate', unit='Hz')
    window          = parse_value(s, 'window', unit='s')
    step            = parse_value(s, 'step', unit='s')
    bins_per_oct    = parse_value(s, 'bins_per_oct', typ='int')
    freq_min        = parse_value(s, 'freq_min', unit='Hz')
    freq_max        = parse_value(s, 'freq_max', unit='Hz')
    window_func     = parse_value(s, 'window_func', typ='str')
    duration        = parse_value(s, 'duration', unit='s')
    resample_method = parse_value(s, 'resample_method', typ='str')

    return {'type':rep_type, 'rate':rate, 'window': window, 'step':step, 
        'bins_per_oct':bins_per_oct, 'freq_min':freq_min, 'freq_max':freq_max, 
        'window_func':window_func, 'resample_method':resample_method, 'duration':duration}

def parse_value(x, name, unit=None, typ='float'):
    Q = ureg.Quantity
    v = None
    if x.get(name) is not None:
        if unit is None:
            v = x[name]
        else:
            v = Q(x[name]).m_as(unit)

        if typ is 'int':
            v = int(v)
        elif unit is 'float':
            v = float(v)
        elif typ is 'str':
            v = str(v)

    return v

def str2bool(v):
    """ Convert most common answers to yes/no questions to boolean

    Args:
        v : str
            Answer 
    
    Returns:
        res : bool
            Answer converted to boolean 
    """
    res = v.lower() in ("yes", "YES", "Yes", "true", "True", "TRUE", "on", "ON", "t", "T", "1")
    return res

