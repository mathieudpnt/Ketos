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

""" Augmentation module within the ketos library

    This module provides utilities to modify existing audio 
    data and/or synthetize new audio data.

    These utilities may either operate in the time domain 
    (waveform) or in the frequency domain (spectrogram), or 
    both.
"""
import numpy as np

def enhance_image(img, threshold=1., enhancement=1.):
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
            threshold: float
                Parameter determining where the transition from low to high takes place.
            enhancement: float
                Parameter determining the amount of enhancement.

        Returns:
            img_en: numpy array
                Enhanced image.
    """
    std = np.std(img)
    half = treshold * std
    wid = (1. / enhancement) * std
    scaling = 1. / (np.exp(-(img - half) / wid) + 1.)
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


def interbreed(specs1, specs2, num, smooth=True, smooth_par=5,\
            scale=(1,1), t_scale=(1,1), f_scale=(1,1), seed=1,\
            validation_function=None, progress_bar=False,\
            min_peak_diff=None, reduce_tonal_noise=False,\
            output_file=None, max_size=1E9, max_annotations=10, mode='a'):
    """ Interbreed spectrograms to create new ones.

        Interbreeding consists in adding/superimposing two spectrograms on top of each other.

        If the spectrograms have different lengths, the shorter of the two will be placed 
        within the larger one with a randomly generated time offset.

        The shorter spectrogram may also be subject to re-scaling along any of its dimensions, as 
        specified via the arguments t_scale_min, t_scale_max, f_scale_min, f_scale_max, scale_min, scale_max.

        Note that the spectrograms must have the same time and frequency resolution. Otherwise an assertion error will be thrown.

        Args:
            specs1: list
                First group of input spectrograms.
            specs2: list
                Second group of input spectrograms.
            num: int
                Number of spectrograms that will be created
            smooth: bool
                If True, a smoothing operation will be applied 
                to avoid sharp discontinuities in the resulting spetrogram
            smooth_par: int
                Smoothing parameter. The larger the value, the less 
                smoothing. Only applicable if smooth is set to True
            scale: tuple
                Scale the spectrogram that is being added by a random 
                number between [a,b)
            t_scale: tuple
                Scale the time axis of the spectrogram that is being added 
                by a random number between [a,b)
            f_scale: tuple
                Scale the frequency axis of the spectrogram that is being added 
                by a random number between [a,b)
            seed: int
                Seed for numpy's random number generator
            validation_function:
                This function is applied to each new spectrogram. 
                The function must accept 'spec1', 'spec2', and 'new_spec'. 
                Returns True or False. If True, the new spectrogram is accepted; 
                if False, it gets discarded.
            progress_bar: bool
                Option to display progress bar.
            min_peak_diff: float
                If specified, the following validation criterion is used:
                max(spec2) > max(spec1) + min_peak_diff
            reduce_tonal_noise: bool
                Reduce continuous tonal noise produced by e.g. ships and slowly varying background noise
            output_file: str
                Full path to output database file (*.h5). If no output file is 
                provided (default), the spectrograms created are kept in memory 
                and passed as return argument; If an output file is provided, 
                the spectrograms are saved to disk.
            max_annotations: int
                Maximum number of annotations allowed for any spectrogram. 
                Only applicable if output_file is specified.
            max_size: int
                Maximum size of output database file in bytes
                If file exceeds this size, it will be split up into several 
                files with _000, _001, etc, appended to the filename.
                The default values is max_size=1E9 (1 Gbyte)
                Only applicable if output_file is specified.
            mode: str
                The mode to open the file. It can be one of the following:
                    ’r’: Read-only; no data can be modified.
                    ’w’: Write; a new file is created (an existing file with the same name would be deleted).
                    ’a’: Append; an existing file is opened for reading and writing, and if the file does not exist it is created.
                    ’r+’: It is similar to ‘a’, but the file must already exist.
            
        Returns:   
            specs: Spectrogram or list of Spectrograms
                Created spectrogram(s). Returns None if output_file is specified.

        Examples:
            >>> # extract saved spectrograms from database file
            >>> import tables
            >>> import ketos.data_handling.database_interface as di
            >>> db = tables.open_file("ketos/tests/assets/morlet.h5", "r") 
            >>> spec1 = di.load_specs(di.open_table(db, "/spec1"))[0]
            >>> spec2 = di.load_specs(di.open_table(db, "/spec2"))[0]
            >>> db.close()
            >>> 
            >>> # interbreed the two spectrograms once to make one new spectrogram
            >>> from ketos.audio_processing.spectrogram import interbreed
            >>> new_spec = interbreed([spec1], [spec2], num=1)
            >>>
            >>> # plot the original spectrograms and the new one
            >>> fig = spec1.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec1.png")
            >>> fig = spec2.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec2.png")
            >>> fig = new_spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/new_spec.png")

            .. image:: ../../../../ketos/tests/assets/tmp/spec1.png

            .. image:: ../../../../ketos/tests/assets/tmp/spec2.png

            .. image:: ../../../../ketos/tests/assets/tmp/new_spec.png

            >>> # Interbreed the two spectrograms to make 3 new spectrograms.
            >>> # Apply a random scaling factor between 0.0 and 5.0 to the spectrogram 
            >>> # that is being added.
            >>> # Only accept spectrograms with peak value at least two times 
            >>> # larger than either of the two parent spectrograms
            >>> def func(spec1, spec2, new_spec):
            ...     m1 = np.max(spec1.image)
            ...     m2 = np.max(spec2.image)
            ...     m = np.max(new_spec.image)
            ...     return m >= 2 * max(m1, m2)
            >>> new_specs = interbreed([spec1], [spec2], num=3, scale=(0,5), validation_function=func)
            >>>
            >>> # plot the first of the new spectrograms
            >>> fig = new_specs[0].plot()
            >>> fig.savefig("ketos/tests/assets/tmp/new_spec_x.png")

            .. image:: ../../../../ketos/tests/assets/tmp/new_spec_x.png

    """
    if output_file:
        from ketos.data_handling.database_interface import SpecWriter
        writer = SpecWriter(output_file=output_file, max_size=max_size, max_annotations=max_annotations, mode=mode)

    # set random seed
    np.random.seed(seed)

    # default validation function always returns True
    if validation_function is None:        
        def always_true(spec1, spec2, new_spec):
            return True

        validation_function = always_true

    if progress_bar:
        import sys
        nprog = max(1, int(num / 100.))

    specs_counter = 0
    specs = list()
    while specs_counter < num:
        
        N = num - len(specs)
        N = num - specs_counter

        # randomly select spectrograms
        _specs1 = np.random.choice(specs1, N, replace=True)
        _specs2 = np.random.choice(specs2, N, replace=True)

        # randomly sampled scaling factors
        sf_t = random_floats(size=N, low=t_scale[0], high=t_scale[1], seed=seed)
        sf_f = random_floats(size=N, low=f_scale[0], high=f_scale[1], seed=seed)
        sf = random_floats(size=N, low=scale[0], high=scale[1], seed=seed)
        seed += 1

        if N == 1:
            sf_t = [sf_t]
            sf_f = [sf_f]
            sf = [sf]

        for i in range(N):

            if progress_bar:
                if specs_counter % nprog == 0:
                    sys.stdout.write('{0:.0f}% \r'.format(specs_counter / num * 100.))

            s1 = _specs1[i]
            s2 = _specs2[i]

            # time offset
            dt = s1.duration() - s2.duration()
            if dt != 0:
                rndm = np.random.random_sample()
                delay = np.abs(dt) * rndm
            else:
                delay = 0

            spec = s1.copy() # make a copy

            if min_peak_diff is not None:
                diff = sf[i] * np.max(s2.image) - np.max(s1.image)
                if diff < min_peak_diff:
                    continue

            # add the two spectrograms
            spec.add(spec=s2, delay=delay, scale=sf[i], make_copy=True,\
                    smooth=smooth, smooth_par=smooth_par, t_scale=sf_t[i], f_scale=sf_f[i])

            if validation_function(s1, s2, spec):

                if reduce_tonal_noise:
                    spec.tonal_noise_reduction()

                if output_file:
                    writer.cd('/spec')
                    writer.write(spec)
                else:                    
                    specs.append(spec)
                
                specs_counter += 1

            if specs_counter >= num:
                break

    if progress_bar:
        print('100%')

    if output_file:
        writer.close()
        return None

    else:
        # if list has length 1, return the element rather than the list
        if len(specs) == 1:
            specs = specs[0]

        return specs
