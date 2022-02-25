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


""" Standard audio-representation parameters recognized by Ketos.
"""
audio_std_params = {'type':                     {'type':str,   'unit':None},
                    'rate':                     {'type':float, 'unit':'Hz'},
                    'window':                   {'type':float, 'unit':'s'},
                    'step':                     {'type':float, 'unit':'s'},
                    'bins_per_oct':             {'type':int,   'unit':None},
                    'freq_min':                 {'type':float, 'unit':'Hz'},
                    'freq_max':                 {'type':float, 'unit':'Hz'},
                    'window_func':              {'type':str,   'unit':None},
                    'resample_method':          {'type':str,   'unit':None},
                    'duration':                 {'type':float, 'unit':'s'},
                    'normalize_wav':            {'type':bool,  'unit':None},
                    'transforms':               {'type':list,  'unit':None},
                    'waveform_transforms':      {'type':list,  'unit':None},
                    'num_chan':                 {'type':int,   'unit':None},
                    'filter_pad_samples':       {'type':int,   'unit':None},
                    'global_km_window_seconds': {'type':float, 'unit':'s'},
                    'local_km_window_seconds':  {'type':float, 'unit':'s'},
                    'filter_n':                 {'type':int,   'unit':None},
                    'filter_min_hz':            {'type':float, 'unit':'Hz'},
                    'decibel':                  {'type':bool,  'unit':None}
                    }


def is_encoded(s):
    """ Check that the audio presentation has been encoded.

        More specifically, the method checks that items specified as having a 
        physical unit in `audio_std_params`, have string values.
    
        Args:
            s: dict
                Audio representation  

        Returns:
            : bool
                True, if the audio representation is encoded. False, otherwise.
    """
    s_dict = {'s': s} if 'type' in s.keys() else s
        
    for _,s in s_dict.items():     
        for key, value in s.items():
            if key in audio_std_params.keys() and \
                    audio_std_params[key]['unit'] != None and not isinstance(value, str):
                return False

    return True


def load_audio_representation(path, name=None, return_unparsed=False):
    """ Load audio representation from JSON file.

        By default the function attempts to parse the individual parameter 
        values, e.g., the value "20 kHz" will be returned as 20000 and the 
        value "11 ms" will be returned as 0.011. Use the `return_unparsed` 
        argument to change this behaviour.

        Args:
            path: str
                Path to json file
            name: str
                Heading of the relevant section of the json file. If None, 
                the function returns the entire content of the JSON file.
            return_unparsed: bool
                Do not parse the parameter values. Default is False.

        Returns:
            d: dict
                Audio representation 

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
            {'type': 'MagSpectrogram', 'rate': 20000.0, 'window': 0.1, 'step': 0.025, 'window_func': 'hamming', 'freq_min': 30, 'freq_max': 3000}
            >>> # clean up
            >>> os.remove(path)
    """
    with open(path, 'r') as fil:
        data = json.load(fil)
        if name != None: 
            data = data[name]
        if not return_unparsed:
            data = parse_audio_representation(data)

    return data

def parse_audio_representation(s):
    """ Parse audio representation parameters.
    
        Args:
            s: dict
                Unparsed audio representation  

        Returns:
            s: dict
                Parsed audio representation
    """
    for key,value in s.items():
        s[key] = parse_parameter(name=key, value=value)

    return s

def parse_parameter(name, value):
    """ Parse the parameter value according to the type and unit specified 
        in the `audio_std_params` dictionary. For example, if name='window' 
        and value='22.1 ms', the function returns the float 0.0221.

        If the parameter is not found in the `audio_std_params` dictionary, 
        the function returns the input value unmodified.

        Args:
            name: str
                Name of the parameter to be parsed
            value: str
                Value of the parameter to be parsed
            
        Returns:
            parsed_value: str, int, float, bool, or list 
                Parsed value

        Example:
            >>> from ketos.data_handling.parsing import parse_parameter
            >>> print(parse_parameter(name='step', value='23 ms'))
            0.023
    """
    Q = ureg.Quantity
    parsed_value = value

    if name in audio_std_params.keys():
        param = audio_std_params[name]
        typ  = param['type'] 
        unit = param['unit']

        if unit is not None: 
            parsed_value = Q(value).m_as(unit)

        if typ in ['int', int]:
            parsed_value = int(parsed_value)

        elif unit in ['float', float]:
            parsed_value = float(parsed_value)

        elif typ in ['str', str]:
            parsed_value = str(parsed_value)

        elif typ in ['bool', bool]:
            parsed_value = (parsed_value.lower() == "true")

        elif typ in [list]:
            # convert specific transform arguments from str to tuple
            if name == 'transforms':
                for tr in parsed_value:
                    if tr['name'] == 'adjust_range':
                        s = tr['range'][1:-1]
                        tr['range'] = tuple(map(int, s.split(',')))

                    elif tr['name'] == 'resize' and 'shape' in tr.keys():
                        s = tr['shape'][1:-1]
                        tr['shape'] = tuple(map(int, s.split(',')))

    return parsed_value

def encode_audio_representation(s):
    """ Encode audio representation.

        Every parameter listed in the `audio_std_params` dictionary 
        with a unit is encoded as a str.
    
        Args:
            s: dict
                Input audio representation  

        Returns:
            s: dict
                Encoded audio representation
    """
    s_dict = {'s': s} if 'type' in s.keys() else s     
    for i,s in s_dict.items():         
        for key,value in s.items():
            s_dict[i][key] = encode_parameter(name=key, value=value)

    s = s_dict['s'] if 's' in s_dict.keys() else s_dict
    return s

def encode_parameter(name, value):
    """ Encode paramater as a string with an SI unit, according to the 
        unit specified in the `audio_std_params` dictionary. For example, 
        if name='window' and value=4.22, the function returns the str '4.22 s'.

        If the parameter is not found in the `audio_std_params` dictionary, 
        the function returns the input value unmodified.
    
        Args:
            name: str
                Name of the parameter to be encoded
            value: str
                Value of the parameter to be encoded
            
        Returns:
            encoded_value: str or type of input value 
                Encoded value

        Example:
            >>> from ketos.data_handling.parsing import encode_parameter
            >>> print(encode_parameter(name='step', value=0.037))
            0.037 s
    """
    encoded_value = value
    if name in audio_std_params.keys():
        param = audio_std_params[name]
        unit = param['unit']
        if unit is not None:
            encoded_value = f'{value} {unit}'

    return encoded_value

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

