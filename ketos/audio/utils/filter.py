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

""" 'audio.utils.filter' module within the ketos library

    This module provides utilities for manipulating and filtering waveforms and
    spectrograms.
"""
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def plot_image(img, fig, ax, extent=None, xlabel='', ylabel=''):
    """ Draw the image.

        Args:
            img: numpy array
                Pixel values
            fig: matplotlib.figure.Figure
                Figure object
            ax: matplotlib.axes.Axes
                Axes object
            extent: tuple(float,float,float,float)
                Extent of axes, optional.
            xlabel: str
                Label for x axis, optional.
            ylabel: str
                Label for y axis, optional.

        Returns:
            None
    """
    img_plt = ax.imshow(img.T, aspect='auto', origin='lower', extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(img_plt, ax=ax, format='%.1f')

def enhance_signal(img, enhancement=1.):
    """ Enhance the contrast between regions of high and low intensity, while preserving 
        the range of pixel values.

        Multiplies each pixel value by the factor,

        .. math::
            f(x) = ( e^{-(x - m_x - \sigma_m) / w} + 1)^{-1}

        where :math:`x` is the pixel value, :math:`m_x` is the pixel value median of 
        the image, and :math:`w = \sigma_x / \epsilon`, where :math:`\sigma_x`
        is the pixel value standard deviation of the image and :math:`\epsilon` is the 
        enhancement parameter.

        Some observations:
          
         * :math:`f(x)` is a smoothly increasing function from 0 to 1.
         * :math:`f(m_x)=0.5`, i.e. the median :math:`m_x` demarks the transition from "low intensity" to "high intensity".
         * The smaller the width, :math:`w`, the faster the transition from 0 to 1.

        Args:
            img : numpy array
                Image to be processed. 
            enhancement: float
                Parameter determining the amount of enhancement.

        Returns:
            img_en: numpy array
                Enhanced image.

        Example:
            >>> from ketos.audio.utils.filter import enhance_signal, plot_image
            >>> #create an image 
            >>> x = np.linspace(-4,4,100)
            >>> y = np.linspace(-6,6,100)
            >>> x,y = np.meshgrid(x,y,indexing='ij')
            >>> img = np.exp(-(x**2+y**2)/(2*0.5**2)) #symmetrical Gaussian 
            >>> img += 0.2 * np.random.rand(100,100)  #add some noise
            >>> # apply enhancement
            >>> img_enh = enhance_signal(img, enhancement=3.0)
            >>> #draw the original image and its enhanced version
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) #create canvas to draw on
            >>> plot_image(img,fig,ax1,extent=(-4,4,-6,6))
            >>> plot_image(img_enh,fig,ax2,extent=(-4,4,-6,6))
            >>> fig.savefig("ketos/tests/assets/tmp/image_enhancement1.png")

            .. image:: ../../../ketos/tests/assets/tmp/image_enhancement1.png
    """
    if enhancement > 0:
        med = np.median(img)
        std = np.std(img)
        wid = (1. / enhancement) * std
        scaling = 1. / (np.exp(-(img - med - std) / wid) + 1.)

    else:
        scaling = 1.

    img_en = img * scaling
    return img_en

def reduce_tonal_noise(img, method='MEDIAN', **kwargs):
    """ Reduce continuous tonal noise produced by e.g. ships and slowly varying 
        background noise

        Currently, offers the following two methods:

            1. MEDIAN: Subtracts from each row the median value of that row.
            
            2. RUNNING_MEAN: Subtracts from each row the running mean of that row.
            
        The running mean is computed according to the formula given in 
        Baumgartner & Mussoline, JASA 129, 2889 (2011); doi: 10.1121/1.3562166

        Args:
            img: numpy.array
                Spectrogram image
            method: str
                Options are 'MEDIAN' and 'RUNNING_MEAN'
        
        Optional args:
            time_const_len: int
                Time constant in number of samples, used for the computation of the running mean.
                Must be provided if the method 'RUNNING_MEAN' is chosen.

        Returns:
            img_new: numpy array
                Corrected spectrogram image

        Example:
            >>> import numpy as np
            >>> from ketos.audio.utils.filter import reduce_tonal_noise, plot_image
            >>> #create an image 
            >>> x = np.linspace(-4,4,100)
            >>> y = np.linspace(-6,6,100)
            >>> x,y = np.meshgrid(x,y,indexing='ij')
            >>> img = np.exp(-(x**2+y**2)/(2*0.5**2)) #symmetrical Gaussian 
            >>> img += 0.2 * np.random.rand(100,100)  #add some flat noise
            >>> #add tonal noise that exhibits sudden increase in amplitude
            >>> img += 0.2 * (1 + np.heaviside(x,0.5)) * np.exp(-(y + 2.)**2/(2*0.1**2))
            >>> #reduce tonal noise 
            >>> img_m = reduce_tonal_noise(img, method='MEDIAN')
            >>> img_r = reduce_tonal_noise(img, method='RUNNING_MEAN', time_const_len=30)
            >>> #draw the resulting images along with the original one 
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4)) #create canvas to draw on
            >>> ext = (-4,4,-6,6)
            >>> plot_image(img,fig,ax1,extent=ext)
            >>> plot_image(img_m,fig,ax2,extent=ext)
            >>> plot_image(img_r,fig,ax3,extent=ext)
            >>> fig.savefig("ketos/tests/assets/tmp/image_tonal_noise_red1.png")

            .. image:: ../../../ketos/tests/assets/tmp/image_tonal_noise_red1.png
    """
    if method == 'MEDIAN':
        img_new = img - np.median(img, axis=0)
    
    elif method == 'RUNNING_MEAN':
        assert 'time_const_len' in kwargs.keys(), 'method RUNNING_MEAN requires time_constant input argument'
        img_new = reduce_tonal_noise_running_mean(img, kwargs['time_const_len'])

    else:
        print('Invalid tonal noise reduction method:',method)
        print('Available options are: MEDIAN, RUNNING_MEAN')

    return img_new

def reduce_tonal_noise_running_mean(img, time_const_len):
    """ Reduce continuous tonal noise produced by e.g. ships and slowly varying background noise 
        by subtracting from each row a running mean, computed according to the formula given in 
        Baumgartner & Mussoline, Journal of the Acoustical Society of America 129, 2889 (2011); doi: 10.1121/1.3562166

        Args:
            img: numpy.array
                Spectrogram image
            time_const_len: int
                Time constant in number of samples, used for the computation of the running mean.
                Must be provided if the method 'RUNNING_MEAN' is chosen.

        Returns:
            img_new : 2d numpy array
                Corrected spetrogram image
    """
    T = time_const_len
    eps = 1 - np.exp((np.log(0.15) * 1. / T))
    rmean = np.average(img, axis=0)
    img_new = np.zeros(img.shape)
    nx = img.shape[0]
    for ix in range(nx):
        img_new[ix,:] = img[ix,:] - rmean # subtract running mean
        rmean = (1 - eps) * rmean + eps * img[ix,:] # update running mean

    return img_new

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
            >>> from ketos.audio.utils.filter import filter_isolated_spots
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
            >>> from ketos.audio.utils.filter import blur_image
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
            >>> from ketos.audio.utils.filter import apply_median_filter
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

            >>> from ketos.audio.utils.filter import apply_preemphasis
            >>> sig = np.array([1,2,3,4,5])
            >>> sig_new = apply_preemphasis(sig, coeff=0.95)
            >>> print(sig_new)
            [1.   1.05 1.1  1.15 1.2 ]
    """
    emphasized_signal = np.append(sig[0], sig[1:] - coeff * sig[:-1])
    
    return emphasized_signal