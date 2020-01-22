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

""" Image module within the ketos library

    This module provides utilities to manipulate spectrogram 
    images.
"""
import numpy as np

def enhance_image(img, enhancement=1.):
    """ Enhance regions of high intensity while suppressing regions of low intensity.

        Multiplies each pixel value by the factor,

            s(x) = 1 / (exp(-(x-x0)/w) + 1)

        where x is the pixel value, x0 = threshold * std(img), and w = 1./enhancement * std(img).

        Some observations:
          
         * s(x) is a smoothly increasing function from 0 to 1.
         * s(x0) = 0.5 (i.e. x0 demarks the transition from "low intensity" to "high intensity")
         * The smaller the value of w, the faster the transition from 0 to 1.

        Args:
            img : numpy array
                Image to be processed. 
            enhancement: float
                Parameter determining the amount of enhancement.

        Returns:
            img_en: numpy array
                Enhanced image.
    """
    if enhancement > 0:
        std = np.std(img)
        half = np.median(img)
        wid = (1. / enhancement) * std
        scaling = 1. / (np.exp(-(img - half) / wid) + 1.)

    else:
        scaling = 1.

    img_en = img * scaling
    return img_en

def tonal_noise_reduction(self, method='MEDIAN', **kwargs):
    """ Reduce continuous tonal noise produced by e.g. ships and slowly varying background noise

        Currently, offers the following two methods:

            1. MEDIAN: Subtracts from each row the median value of that row.
            
            2. RUNNING_MEAN: Subtracts from each row the running mean of that row.
            
        The running mean is computed according to the formula given in Baumgartner & Mussoline, JASA 129, 2889 (2011); doi: 10.1121/1.3562166

        Args:
            method: str
                Options are 'MEDIAN' and 'RUNNING_MEAN'
        
        Optional args:
            time_constant: float
                Time constant used for the computation of the running mean (in seconds).
                Must be provided if the method 'RUNNING_MEAN' is chosen.

        Example:
            >>> # read audio file
            >>> from ketos.audio_processing.audio import AudioSignal
            >>> aud = AudioSignal.from_wav('ketos/tests/assets/grunt1.wav')
            >>> # compute the spectrogram
            >>> from ketos.audio_processing.spectrogram import MagSpectrogram
            >>> spec = MagSpectrogram(aud, winlen=0.2, winstep=0.02, decibel=True)
            >>> # keep only frequencies below 800 Hz
            >>> spec.crop(fhigh=800)
            >>> # show spectrogram as is
            >>> fig = spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec_before_tonal.png")
            >>> plt.close(fig)
            >>> # tonal noise reduction
            >>> spec.tonal_noise_reduction()
            >>> # show modified spectrogram
            >>> fig = spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec_after_tonal.png")
            >>> plt.close(fig)

            .. image:: ../../../../ketos/tests/assets/tmp/spec_before_tonal.png

            .. image:: ../../../../ketos/tests/assets/tmp/spec_after_tonal.png

    """
    if method is 'MEDIAN':
        self.image = self.image - np.median(self.image, axis=0)
    
    elif method is 'RUNNING_MEAN':
        assert 'time_constant' in kwargs.keys(), 'method RUNNING_MEAN requires time_constant input argument'
        self.image = self._tonal_noise_reduction_running_mean(kwargs['time_constant'])

    else:
        print('Invalid tonal noise reduction method:',method)
        print('Available options are: MEDIAN, RUNNING_MEAN')
        print('Spectrogram is unchanged')

def _tonal_noise_reduction_running_mean(self, time_constant):
    """ Reduce continuous tonal noise produced by e.g. ships and slowly varying background noise 
        by subtracting from each row a running mean, computed according to the formula given in 
        Baumgartner & Mussoline, Journal of the Acoustical Society of America 129, 2889 (2011); doi: 10.1121/1.3562166

        Args:
            time_constant: float
                Time constant used for the computation of the running mean (in seconds).

        Returns:
            new_img : 2d numpy array
                Corrected spetrogram image
    """
    dt = self.tres
    T = time_constant
    eps = 1 - np.exp((np.log(0.15) * dt / T))
    nx, ny = self.image.shape
    rmean = np.average(self.image, axis=0)
    new_img = np.zeros(shape=(nx,ny))
    for ix in range(nx):
        new_img[ix,:] = self.image[ix,:] - rmean # subtract running mean
        rmean = (1 - eps) * rmean + eps * self.image[ix,:] # update running mean

    return new_img


def filter_isolated_spots(img, struct=np.array([[1,1,1],[1,1,1],[1,1,1]])):
    """ Remove isolated spots from the image.

        Args:
            img : numpy array
                An array like object representing an image. 
            struct : numpy array
                A structuring pattern that defines feature connections.
                Must be symmetric.

        Returns:
            filtered_array : numpy array
                An array containing the input image without the isolated spots.

        Example:

            >>> from ketos.audio_processing.audio_processing import filter_isolated_spots
            >>> img = np.array([[0,0,1,1,0,0],
            ...                 [0,0,0,1,0,0],
            ...                 [0,1,0,0,0,0],
            ...                 [0,0,0,0,0,0],
            ...                 [0,0,0,1,0,0]])
            >>> # remove pixels without neighbors
            >>> img_fil = filter_isolated_spots(img)
            >>> print(img_fil)
            [[0 0 1 1 0 0]
             [0 0 0 1 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]
    """
    filtered_array = np.copy(img)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(img, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

def blur_image(img, size=20, sigma=5, gaussian=True):
    """ Smooth the input image using a median or Gaussian blur filter.
        
        Note that the input image is recasted as np.float32.

        This is essentially a wrapper around the scipy.ndimage.median_filter 
        and scipy.ndimage.gaussian_filter methods. 

        For further details, see https://docs.scipy.org/doc/scipy/reference/ndimage.html

        Args:
            img : numpy array
                Image to be processed. 
            size: int
                Only used by the median filter. Describes the shape that is taken from the input array,
                at every element position, to define the input to the filter function.
            sigma: float or array
                Only used by the Gaussian filter. Standard deviation for Gaussian kernel. May be given as a 
                single number, in which case all axes have the same standard deviation, or as an array, allowing 
                for the axes to have different standard deviations.
            Gaussian: bool
                Switch between median and Gaussian (default) filter

        Returns:
            blur_img: numpy array
                Blurred image.

        Example:

            >>> from ketos.audio_processing.audio_processing import blur_image
            >>> img = np.array([[0,0,0],
            ...                 [0,1,0],
            ...                 [0,0,0]])
            >>> # blur using Gaussian filter with sigma of 0.5
            >>> img_blur = blur_image(img, sigma=0.5)
            >>> img_blur = np.around(img_blur, decimals=2) # only keep up to two decimals
            >>> print(img_blur)
            [[0.01 0.08 0.01]
             [0.08 0.62 0.08]
             [0.01 0.08 0.01]]
    """

    try:
        assert img.dtype == "float32", "img type {0} shoult be 'float32'".format(img.dtype)
    except AssertionError:
        img = img.astype(dtype = np.float32)    
    
    if (gaussian):
        img_blur = ndimage.gaussian_filter(img, sigma=sigma)
    else:
        img_blur = ndimage.median_filter(img, size=size)

    return img_blur

def apply_median_filter(img, row_factor=3, col_factor=4):
    """ Discard pixels that are lower than the median threshold. 

        The resulting image will have 0s for pixels below the threshold and 1s for the pixels above the threshold.

        Note: Code adapted from Kahl et al. (2017)
            Paper: http://ceur-ws.org/Vol-1866/paper_143.pdf
            Code:  https://github.com/kahst/BirdCLEF2017/blob/master/birdCLEF_spec.py 

        Args:
            img : numpy array
                Array containing the img to be filtered. 
                OBS: Note that contents of img are modified by call to function.
            row_factor: int or float
                Factor by which the row-wise median pixel value will be multiplied in orther to define the threshold.
            col_factor: int or float
                Factor by which the col-wise median pixel value will be multiplied in orther to define the threshold.

        Returns:
            filtered_img: numpy array
                The filtered image with 0s and 1s.

        Example:

            >>> from ketos.audio_processing.audio_processing import apply_median_filter
            >>> img = np.array([[1,4,5],
            ...                 [3,5,1],
            ...                 [1,0,9]])
            >>> img_fil = apply_median_filter(img, row_factor=1, col_factor=1)
            >>> print(img_fil)
            [[0 0 0]
             [0 1 0]
             [0 0 1]]
    """

    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    img[img <= row_median * row_factor] = 0
    img[img <= col_median * col_factor] = 0 
    filtered_img = img
    filtered_img[filtered_img > 0] = 1

    return filtered_img

def apply_preemphasis(sig, coeff=0.97):
    """ Apply pre-emphasis to signal

        Args:
            sig : numpy array
                1-d array containing the signal.
            coeff: float
                The preemphasis coefficient. If set to 0,
                no preemphasis is applied (the output will be the same as the input).

        Returns:
            emphasized_signal : numpy array
                The filtered signal.

        Example:

            >>> from ketos.audio_processing.audio_processing import apply_preemphasis
            >>> sig = np.array([1,2,3,4,5])
            >>> sig_new = apply_preemphasis(sig, coeff=0.95)
            >>> print(sig_new)
            [1.   1.05 1.1  1.15 1.2 ]
    """
    emphasized_signal = np.append(sig[0], sig[1:] - coeff * sig[:-1])
    
    return emphasized_signal

def inv_magphase(mag, angle):
    """ Computes complex value from magnitude and phase angle.

        Args:
            mag: numpy array
                Magnitude
            angle: float or numpy array
                Phase angle

        Returns:
            c: numpy array
                Complex value
    """
    phase = np.cos(angle) + 1.j * np.sin(angle)
    c = mag * phase
    return c  
