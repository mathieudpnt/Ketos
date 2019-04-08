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

""" Utilities module within the ketos library

    This module provides a number of auxiliary methods.
"""

import os
import numpy as np

def ensure_dir(file_path):
    """ Ensure that destination directory exists.

        If the directory does not exist, it is created.
        If it already exists, nothing happens.
        
        Args:
            file_path: str
                Full path to destination
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def random_floats(size=1, low=0, high=1, seed=1):
    """ Returns a random number or numpy array of randum numbers uniformly distributed in the half-open interval [low, high)
        
        Args:
            size: int
                Number of random numbers to be generated
            low: float
                Lower value
            high: float
                Upper value (not included)
            seed: int
                Seed for the random number generator

        Returns:
            res: float or numpy array
                Generated random number(s)

        Example:
            >>> from ketos.utils import random_floats
            >>> 
            >>> x = random_floats(3, 0.4, 7.2)
            >>> print(x)
            [3.23574963 5.29820656 0.40077775]
    """

    assert high >= low, "Upper limit must be greater than or equal to lower limit"
    assert size >= 1, "Size must be an int greater than or equal to 1"

    np.random.seed(seed)

    if high == low:
        if size == 1:
            res = high
        else:
            res = np.ones(size, dtype=float) * high
    
    else:
        rndm = np.random.random_sample(size)
        res = low + (high - low) * rndm
        if len(res) == 1:
            res = np.float(res)

    return res

def ndim(a):
    """ Returns the number of dimensions of a list/tuple/array.
        
        Args:
            a: list, tuple or numpy array
                Object that we wish to obtain the dimension of 

        Returns:
            n: int
                Number of dimensions

        Example:
            >>> from ketos.utils import ndim
            >>> 
            >>> x = [[0, 1, 2, 3],(4, 5)]
            >>> print(ndim(x))
            2
    """
    if not (type(a) == list or type(a) == tuple or type(a) == np.ndarray):
        return 0

    if len(a) == 0:
        return 1

    n = 1 + ndim(a[0])
    return n

def tostring(box, decimals=None):
    """ Convert an array, tuple or list into a string.

        Args:
            box: array, tuple or list
                Array, tuple or list that will be converted into a string.
            decimals: int
                Number of decimals that will be kept in the conversion to string.

        Returns:
            s: str
                String representation of array/tuple/list.

        Example:
            >>> from ketos.utils import tostring
            >>> 
            >>> y = [[0, 1, 2, 3],(4, 5)]
            >>> print(tostring(y))
            [[0,1,2,3],[4,5]]
    """
    if box is None:
        return ''

    box = np.array(box)

    if decimals is not None:
        box = np.around(box, decimals=int(decimals))

    box = box.tolist()

    s = str(box)
    s = s.replace(' ', '')
    s = s.replace('(', '[')
    s = s.replace(')', ']')

    return s

def octave_bands(band_min=-1, band_max=9):
    """ Compute the min, central, and max frequency value 
        of the specified octave bands, using the following 
        formulas,

            f_centre = 10^3 * 2^p ,
            f_min = f_centre / sqrt(2) ,
            f_max = f_centre * sqrt(2) ,

        where p = band_number - 5

        Args:
            band_min: int
                Lower octave band
            band_max: int
                Upper octave band

        Returns:
            fcentre: numpy array
                Central frequency of each band (in Hz)
            flow: numpy array
                Minimum frequency of each band (in Hz)
            fhigh: numpy array
                Maximum frequency of each band (in Hz)

        Example:
            >>> from ketos.utils import octave_bands
            >>> 
            >>> fc, fmin, fmax = octave_bands(1, 3)
            >>> print(fc)
            [ 62.5 125.  250. ]
    """
    p = np.arange(band_min-5., band_max-4.)
    fcentre = np.power(10.,3) * np.power(2.,p)
    fd = np.sqrt(2.)
    flow = fcentre / fd
    fhigh = fcentre * fd
    return fcentre, flow, fhigh

def octave_bands_json(band_min, band_max):
    """ Produce a string of the specified octave bands
        in json format

        Args:
            band_min: int
                Lower octave band
            band_max: int
                Upper octave band

        Returns:
            s: str
                json format string

        Example:
            >>> from ketos.utils import octave_bands_json
            >>> 
            >>> s = octave_bands_json(1, 2)
    """
    fcentre, flow, fhigh = octave_bands(band_min, band_max)

    s = "\"frequency_bands\": [\n"
    n = len(flow)
    for i in range(n):
        s += "\t{\n"
        s += "\t\t\"name\": \"{0:.0f}Hz\",\n".format(fcentre[i])
        s += "\t\t\"range\": [\"{0:.1f}Hz\", \"{1:.1f}Hz\"]".format(flow[i],fhigh[i])
        endpar = "\n\t}"
        if i < n-1:
            endpar += ","

        s += endpar + "\n"

    s += "]"
    return s

def morlet_func(time, frequency, width, displacement, norm=True, dfdt=0):
    """ Compute Morlet wavelet function

        The function is implemented as in Eq. (15) in John Ashmead, "Morlet Wavelets in Quantum Mechanics",
        Quanta 2012; 1: 58-70, with the replacement f -> 2*pi*f*s, to allow f to be identified with the 
        physical frequency.

        Args:
            time: float or numpy array
               Time in seconds at which the function is to be evaluated
            frequency: float
                Wavelet frequency in Hz
            width: float
                Wavelet width in seconds (1-sigma width of the Gaussian envelope function)
            displacement: float
                Wavelet centroid in seconds
            norm: bool
                Include [pi^1/4*sqrt(sigma)]^-1 normalization factor
            dfdt: float
                Rate of change in frequency as a function of time in Hz per second.
                If dfdt is non-zero, the frequency is computed as 
                    
                    f = frequency + (time - displacement) * dfdt 

        Returns:
            y: float or numpy array
                Value of Morlet wavelet function at time t

        Example:
            >>> from ketos.utils import morlet_func
            >>> 
            >>> time = np.array([-1., 0., 0.5])
            >>> f = morlet_func(time=time, frequency=10, width=3, displacement=0)
            >>> print(f)
            [0.41022718 0.43366254 0.42768108]
    """
    if dfdt != 0:
        frequency += (time - displacement) * dfdt
    
    assert np.all(frequency > 0), "Frequency must be a strictly positive float"
    assert width > 0, "Width must be a strictly positive float"

    t = time
    w = 2 * np.pi * frequency * width
    s = width
    l = displacement
    x = (t-l)/s

    y = (np.exp(1j*w*x) - np.exp(-0.5*(w**2))) * np.exp(-0.5*(x**2))

    if norm:
        y *= (s * np.sqrt(np.pi) * (1 + np.exp(-w**2) - 2*np.exp(-0.75*w**2)) )**-0.5

    return np.real(y)
