import json
from collections import namedtuple
from pint import UnitRegistry # SI units
from enum import Enum


class WinFun(Enum):
    HAMMING = 1


ureg = UnitRegistry()


Interval = namedtuple('Interval', 'low high')
Interval.__doc__ = '''\
Numerical intervals

low - Lower limit (float)
high - Upper limit (float)''' 


SpectrogramConfiguration = namedtuple('SpectrogramConfiguration', 'rate window_size step_size window_function low_frequency_cut high_frequency_cut')
SpectrogramConfiguration.__doc__ = '''\
Configuration parameters for generation of spectrograms

rate - Sampling rate in Hz (int)
window_size - Window size used for framing in seconds (float)
step_size - Step size used for framing in seconds (float)
window_function - Window function used for framing (e.g. Hamming window)
low_frequency_cut - Low-frequency cut-off in Hz (float)
high_frequency_cut - High-frequency cut-off in Hz (float)''' 


def parse_spectrogram_configuration(data):
    """ Parse configuration settings for generating spectrograms.

        Any setting not specified in the json string will be set 
        to None.

    Args:
        data : str
            Json-format string with the configuration settings 
    
    Returns:
        c : SpectrogramConfiguration
            Spectrogram configuration settings

    Example:
    
        >>> import json
        >>> import ketos.data_handling.parsing.parse_spectrogram_configuration as parse
        >>> 
        >>> input = '{"spectrogram": {"rate": "20 kHz", "window_size": "0.1 s", "step_size": "0.025 s", "window_function": "HAMMING", "low_frequency_cut": "30Hz", "high_frequency_cut": "3000Hz"}}'
        >>> data = json.loads(input)
        >>> settings = parse(data['spectrogram'])
        >>> print(settings.rate)  # print sampling rate in Hz
    """
    Q = ureg.Quantity

    # default values
    rate, wsiz, step, wfun, flow, fhigh = None, None, None, None, None, None    

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

    if data.get('window_function') is not None:

        for name, member in WinFun.__members__.items():
            if data['window_function'] == name:
                wfun = member

        if wfun is None:
            s = ", ".join(name for name, _ in WinFun.__members__.items())
            raise ValueError("Unknown window function. Select between: "+s)

    if data.get('low_frequency_cut') is not None:
        flow = Q(data['low_frequency_cut'])
        flow = flow.m_as("Hz")

    if data.get('high_frequency_cut') is not None:
        fhigh = Q(data['high_frequency_cut'])
        fhigh = fhigh.m_as("Hz")

    c = SpectrogramConfiguration(rate, wsiz, step, wfun, flow, fhigh)    

    return c


def parse_frequency_bands(data):
    """ Parse list of frequency bands

    Args:
        data : str
            Json-format string frequency bands 
    
    Returns:
        name : list(str)
            Band names
        freq_intv: list(Interval)
            Band frequency ranges in Hz 
    """
    Q = ureg.Quantity
    
    name, freq_intv = list(), list()

    for band in data:

        assert band.get('name') is not None, 'name field is required for frequency band'
        assert band.get('range') is not None, 'range field is required for frequency band'

        name.append(band['name'])

        low = Q(band['range'][0])
        low = low.m_as('Hz')

        high = Q(band['range'][1])
        high = high.m_as('Hz')

        intv = Interval(low, high)
        freq_intv.append(intv)

    return name, freq_intv