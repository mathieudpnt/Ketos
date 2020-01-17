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

""" Axis module within the ketos library

    This module provides utilities to convert bin numbers to 
    continuous variable values and vice versa.

    Contents:
        LinearAxis class:
        Log2Axis class
"""
import numpy as np

def bin_number(x, pos_func, bins, truncate=False):
    """ Compute bin number corresponding to a given value.

        If the value lies outside the axis range, a negative 
        bin number or a bin number above N-1 will be returned. 
        This behaviour can be changed using the argument 'truncate'.

        Args:
            x: array-like
                Value
            pos_func: function
                Calculates the position on the axis of any given input value.
            bins: int
                Number of bins
            truncate: bool
                Return 0 if x is below the lower axis boundary 
                and N-1 if x is above the upper boundary. Default 
                is False.

        Returns: 
            b: array-like
                Bin number
    """
    if np.ndim(x) == 0:
        scalar = True
        x = [x]
    else:
        scalar = False

    if isinstance(x, list):
        x = np.array(x)

    b = pos_func(x)

    if truncate:
        b[b < 0] = 0
        b[b >= bins] = bins - 1
    else:
        b[b < 0] = b[b < 0] - 1
        b[b == bins] = b[b == bins] - 1

    b = b.astype(dtype=int, copy=False)

    if scalar:
        b = b[0]

    return b

class LinearAxis():
    """ Linear axis.

        Bins are numbered 0,1,2,3,...,N-1 from lower to 
        higher values, where N is the number of bins.

        Args: 
            bins: int
                Number of bins
            extent: tuple(float,float)
                Axis range, e.g. (0., 100.)

        Attributes:
            bins: int
                Number of bins
            x_min: float
                Left edge of first bin
            x_max: float
                Right edge of last bin            
            dx: float
                Bin width
    """
    def __init__(self, bins, extent):
        self.bins = int(bins)
        self.x_min = extent[0]
        self.x_max = extent[1]
        self.dx = (self.x_max - self.x_min) / self.bins

    def _pos_func(self, x):
        """ Compute the position of a given input value on the axis.

            Args:
                x: array-like
                    Value

            Returns: 
                : array-like
                    Position
        """  
        return (x - self.x_min) / self.dx    

    def bin(self, x, truncate=False):
        """ Get bin number corresponding to a given value.

            If the value lies outside the axis range, a negative 
            bin number or a bin number above N-1 will be returned. 
            This behaviour can be changed using the argument 'truncate'.

            Args:
                x: array-like
                    Value
                truncate: bool
                    Return 0 if x is below the lower axis boundary 
                    and N-1 if x is above the upper boundary. Default 
                    is False.

            Returns: 
                b: array-like
                    Bin number

            Example:
                >>> from ketos.audio_processing.axis import LinearAxis
                >>> #Linear axis between 0. and 100. with 200 bins.
                >>> ax = LinearAxis(bins=200, extent=(0.,100.))
                >>> #Get bin number corresponding to x=0.6
                >>> b = ax.bin(0.6)
                >>> print(b)
                1
                >>> #Get several bin numbes in one call 
                >>> b = ax.bin([0.6,11.1])
                >>> print(b)
                [ 1 22]
                >>> #Get bin number for values at bin edges
                >>> b = ax.bin([0.0,0.5,1.0,100.])
                >>> print(b)
                [  0   1   2 199]
                >>> #Note that when the value sits between two bins, 
                >>> #the higher bin number is returned, expect if the 
                >>> #value sits at the upper edge of the last bin, in 
                >>> #which case the lower bin number (i.e. the last bin)
                >>> #is returned.  
                >>> 
                >>> #Get bin numbers outside the axis range
                >>> b = ax.bin([-2.1, 100.1])
                >>> print(b)
                [ -5 200]
                >>> b = ax.bin([-2.1, 100.1], truncate=True)
                >>> print(b)
                [  0 199]
        """
        b = bin_number(x, pos_func=self._pos_func, bins=self.bins, truncate=truncate)
        return b

    def low_edge(self, b):
        """ Get the lower-edge value of a given bin.

            Args:
                b: array-like
                    Bin number.
            
            Returns: 
                x: array-like
                    Lower-edge bin value
        """
        if isinstance(b, list):
            b = np.array(b)

        x = self.x_min + b * self.dx
        return x

    def min(self):
        return self.x_min

    def max(self):
        return self.x_max

    def bin_width(self, i):
        return self.dx

class Log2Axis():
    """ Logarithmic axis with base 2.
    
        Bins are numbered :math:`0,1,2,3,...,N-1` from lower to 
        higher values, where :math:`N` is the number of bins.

        The lower-edge value of bin no. :math:`i` is calculated 
        from the formula,

        .. math:: 
            x_{i} = 2^{i / m} \cdot x_{0}

        where :math:`m` is the number of bins per octave and 
        :math:`x_0` is the lower-edge value of the first bin.

        Args: 
            num_oct: int
                Number of octaves
            bins_per_oct: int
                Number of bins per octave
            min_value: float
                Left edge of first bin

        Attributes:
            bins: int
                Number of bins
            num_oct: int
                Number of octaves
            bins_per_oct: int
                Number of bins per octave
            x_min: float
                Left edge of first bin
    """
    def __init__(self, num_oct, bins_per_oct, min_value):
        self.num_oct = int(num_oct)
        self.bins_per_oct = int(bins_per_oct)
        self.x_min = min_value
        self.bins = self.num_oct * self.bins_per_oct

    def _pos_func(self, x):
        """ Compute the position of a given input value on the axis.

            Args:
                x: array-like
                    Value

            Returns: 
                : array-like
                    Position
        """  
        return self.bins_per_oct * np.log2(x / self.x_min)

    def bin(self, x, truncate=False):
        """ Get bin number corresponding to a given value.

            If the value lies outside the axis range, a negative 
            bin number or a bin number above N-1 will be returned. 
            This behaviour can be changed using the argument 'truncate'.

            Args:
                x: array-like
                    Value
                truncate: bool
                    Return 0 if x is below the lower axis boundary 
                    and N-1 if x is above the upper boundary. Default 
                    is False.

            Returns: 
                b: array-like
                    Bin number

            Example:
                >>> from ketos.audio_processing.axis import LinearAxis
                >>> ax = LinearAxis(bins=100, extent=(0.,100.))
        """
        b = bin_number(x, pos_func=self._pos_func, bins=self.bins, truncate=truncate)
        return b

    def low_edge(self, b):
        """ Get the lower-edge value of a given bin.

            Args:
                b: array-like
                    Bin number.
            
            Returns: 
                x: array-like
                    Lower-edge bin value
        """
        if isinstance(b, list):
            b = np.array(b)

        x = 2**(b / self.bins_per_oct) * self.x_min
        return x

    def min(self):
        return self.x_min

    def max(self):
        x = self.low_edge(self.bins)
        return x

    def bin_width(self, i):
        w = self.low_edge(i+1) - self.low_edge(i)
        return w

