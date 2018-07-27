import json
from collections import namedtuple
from pint import UnitRegistry # SI units
from enum import Enum

class Stat(Enum):
    AVERAGE = 1
    MEDIAN = 2

class WinFun(Enum):
    HAMMING = 1

ureg = UnitRegistry()

Interval = namedtuple('Interval', 'low high')
Interval.__doc__ = '''\
Numerical intervals

low - Lower limit (float)
high - Upper limit (float)''' 

SpectrConfig = namedtuple('SpectrConfig', 'rate window_size step_size window_function')
Interval.__doc__ = '''\
Configuration parameters for generation of spectrograms

rate - Sampling rate in Hz (int)
window_size - Window size used for framing in seconds (float)
step_size - Step size used for framing in seconds (float)
window_function - Window function used for framing (e.g. Hamming window)''' 


def parse_spectrogram_config(data):

    # for parsing values with units
    Q = ureg.Quantity

    # default values
    rate, wsiz, step, wfun = None, None, None, None

    # check that entry exists, before attempting to read
    if data.get('rate') is not None:
        rate = Q(data['rate'])
        rate = rate.m_as("Hz")
    if data.get('window_size') is not None:
        wsiz = Q(data['window_size'])
        wsiz = wsiz.m_as("s")
    if data.get('step_size') is not None:
        step = Q(data['step_size'])
        step = step.m_as("s")
    if data.get('window_func') is not None:
        for name, member in WinFun.__members__.items():
            if data['window_func'] == name:
                wfun = member
        if wfun is None:
            s = ", ".join(name for name, _ in WinFun.__members__.items())
            raise ValueError("Unknown window function. Select between: "+s)

    # return
    c = SpectrConfig(rate, wsiz, step, wfun)    
    return c


def parse_frequency_bands(data):

    # for parsing values with units
    Q = ureg.Quantity
    
    name, freq_intv = list(), list()
    for band in data:
        name.append(band['name'])
        low = Q(band['range'][0])
        low = low.m_as('Hz')
        high = Q(band['range'][1])
        high = high.m_as('Hz')
        intv = Interval(low, high)
        freq_intv.append(intv)

    return name, freq_intv

    
def parse_stat_time_series_config(data):   

    stat_quantity = Stat(1)
    window_size = None

    if data.get('stat_quantity') is not None:
        stat_quantity = None
        for name, member in Stat.__members__.items():
            if data['stat_quantity'] == name:
                stat_quantity = member

        if stat_quantity is None:
            s = ", ".join(name for name, _ in Stat.__members__.items())
            raise ValueError("Unknown statistical quantity. Select between: "+s)

    # for parsing values with units
    Q = ureg.Quantity

    if data.get('window_size') is not None:
        window_size = Q(data['window_size'])
        window_size = window_size.m_as("s")

    return stat_quantity, window_size