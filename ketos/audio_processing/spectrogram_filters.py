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

""" spectrogram_filters module within the ketos library

    This module contains various filters that can be applied to spectrograms.

    Filters are implemented as classes, which are required to have the 
    attribute name and the method apply. 

    Contents:
        HarmonicFilter class
        FrequencyFilter class
        WindowFilter class
        WindowSubtractionFilter class
"""

import numpy as np
import pandas as pd
from ketos.audio_processing.spectrogram import Spectrogram
from ketos.utils import nearest_values


class HarmonicFilter():
    """ Performs Fast Fourier Transform (FFT) of the frequency axis.

        Attributes:
            name: str
                Filter name
    """

    def __init__(self):

        self.name = "Harmonic"

    def apply(self, spec):
        """ Apply harmonic filter to spectrogram

            Args:
                spec: Spectrogram
                    Spectrogram to which filter will be applied 
        """
        h = np.abs(np.fft.rfft(spec.image, axis=1))
        new_img = h[:,1:]

        flabels = list()
        for i in range(new_img.shape[1]):
            f = 2 * float(i) / float(new_img.shape[1]) * (spec.fmax() - spec.fmin)
            flabels.append("{0:.1f}Hz".format(f))

        spec.image = new_img
        spec.flabels = flabels


class CroppingFilter():
    """ Crops along the frequency dimension.

        Args:
            flow: float
                Lower bound on frequency in Hz
            fhigh: float
                Upper bound on frequency in Hz

        Attributes:
            name: str
                Filter name
    """

    def __init__(self, flow=None, fhigh=None):

        self.flow = flow
        self.fhigh = fhigh
        self.name = "Cropping"

    def apply(self, spec):
        """ Apply frequency cropping filter to spectrogram

            Args:
                spec: Spectrogram
                    Spectrogram to which filter will be applied 
        """
        spec.crop(flow=self.flow, fhigh=self.fhigh)


class FrequencyFilter():
    """ Splits the frequency axis into bands and computes 
        the average sound magnitude within each band.

        Args:
            bands: list(Interval)
                Frequency bands in Hz
            names: list(str)
                Names of the frequency bands

        Attributes:
            name: str
                Filter name
    """

    def __init__(self, bands, names=None):

        self.bands = bands
        self.names = names
        self.name = "Frequency"

    def apply(self, spec):
        """ Apply frequency filter to spectrogram

            Args:
                spec: Spectrogram
                    Spectrogram to which filter will be applied 
        """
        nt = spec.tbins()
        nf = len(self.bands)

        new_img = np.ma.zeros(shape=(nt,nf))

        for i in range(nf):
            b = self.bands[i]
            new_img[:,i] = spec.average(axis=1, flow=b.low, fhigh=b.high)

        # discard frequency bands with NaN's
        not_nan = ~np.any(np.isnan(new_img), axis=0)
        new_img = new_img[:, not_nan]
        names = [i for (i, v) in zip(self.names, not_nan) if v]

        # mask zeros
        new_img = np.ma.masked_values(new_img, 0)

        spec.image = new_img
        spec.flabels = names


class WindowFilter():
    """ Applies a windowed median/average filter to the spectrogram.

        Args:
            window_size: float
                Window size in seconds
            step_size: float
                Step size in seconds
            filter_func: 
                Filtering function. Default is np.ma.median

        Attributes:
            name: str
                Filter name
    """

    def __init__(self, window_size, step_size, filter_func=np.ma.median):

        self.window_size = window_size
        self.step_size = step_size
        self.filter_func = filter_func

        if filter_func is np.ma.median:
            self.name = "Median"
        elif filter_func is np.ma.average:
            self.name = "Average"
        else:
            self.name = "Window"
            

    def apply(self, spec):
        """ Apply filter to spectrogram

            Args:
                spec: Spectrogram
                    Spectrogram to which filter will be applied 
        """
        step = int(np.ceil(self.step_size / spec.tres))
        window = int(np.ceil(self.window_size / spec.tres))
        
        nt = spec.tbins()
        nf = spec.fbins()
        
        n = int(np.ceil(nt/step))

        img = spec.image
        new_img = np.zeros(shape=(n,nf))

        for i in range(n):
            i1 = i * step
            i2 = min(nt-1, i1 + window)
            new_img[i,:] = self.filter_func(img[i1:i2+1,:], axis=0)  # ignore entries with value=0

        # mask zeros
        new_img = np.ma.masked_values(new_img, 0)    

        spec.image = new_img
        spec.tres = step * spec.tres


class WindowSubtractionFilter():
    """ Applied a windowed median/average subtraction filter to the spectrogram.

        Args:
            window_size: float
                Window size in seconds
            filter_func: 
                Filtering function. Default is np.ma.median

        Attributes:
            name: str
                Filter name
    """

    def __init__(self, window_size, filter_func=np.ma.median):

        self.window_size = window_size
        self.filter_func = filter_func

        if filter_func is np.ma.median:
            self.name = "Median Subtraction"
        elif filter_func is np.ma.average:
            self.name = "Average Subtraction"
        else:
            self.name = "Window Subtraction"

    def apply(self, spec):
        """ Apply filter to spectrogram

            Args:
                spec: Spectrogram
                    Spectrogram to which filter will be applied 
        """
        window = 1 + 2 * int(0.5 * self.window_size / spec.tres)
        
        nt = spec.tbins()
        nf = spec.fbins()

        img = spec.image
        new_img = np.zeros(shape=(nt,nf))

        # loop over time bins
        for i in range(nt):

            # local median
            v = nearest_values(x=img, i=i, n=window)
            med = self.filter_func(v, axis=0)
            
            new_img[i,:] = img[i,:] - med
        
        spec.image = new_img

