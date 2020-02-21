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

""" 'audio.audio_provider' module within the ketos library

    This module contains the utilities for loading waveforms and computing spectrograms.

    Contents:
        AudioProvider class
"""
import os
import numpy as np
import librosa
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram,MagSpectrogram,PowerSpectrogram,MelSpectrogram,CQTSpectrogram
from ketos.data_handling.data_handling import find_wave_files

class AudioProvider():
    """ Load waveforms or compute spectrograms from raw audio (*.wav) files.

        TODO: Add annotations to created Waveform/Spectrogram objects

        Args:
            path: str
                Full path to audio file (*.wav) or folder containing audio files
            seg_dur: float
                Segment length in seconds.
            seg_step: float
                Separation between consecutive segments in seconds. If None, the separation 
                equals the segment length.
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            rep: str
                Audio data representation. Options are Waveform, MagSpectrogram, PowerSpectrogram, MelSpectrogram, CQTSpectrogram.
            rate: float
                Sampling rate in Hz. If specified, audio data will be resampled at this rate
            window: float
                Window size used for computing the spectrogram in seconds. Only relevant for 
                STFT spectrograms (Mag, Power, Mel).
            step: float
                Step size used for computing the spectrogram in seconds.
            bins_per_oct: int
                Number of bins per octave. Only relevant for CQT spectrograms.
            freq_min: float
                Lower value of the frequency axis in Hz
            freq_max: float
                Upper value of the frequency axis in Hz
            window_func: str
                Window function (optional). Select between
                    * bartlett
                    * blackman
                    * hamming (default)
                    * hanning
            annot: pandas DataFrame
                Annotation table
    """
    def __init__(self, path, seg_dur, seg_step=None, channel=0, rep='MagSpectrogram',
        rate=None, window=None, step=None, bins_per_oct=None, freq_min=None, freq_max=None, 
        window_func=None, annot=None):

        self.channel = channel
        self.rep = rep
        self.seg_dur = seg_dur
        if seg_step is None: self.seg_step = seg_dur
        else: self.seg_step = seg_step
        self.annot = annot
        self.freq_cut = {'freq_min':freq_min, 'freq_max':freq_max}

        if rep is 'Waveform':
            self.config = {'rate': rate}
        elif rep is 'CQTSpectrogram':
            if freq_min is None: freq_min = 1
            self.config = {'rate': rate, 'step':step, 'freq_min':freq_min, 'freq_min':freq_max, 
                'bins_per_oct':bins_per_oct, 'window_func':window_func}
        elif rep in ['MagSpectrogram', 'PowerSpectrogram', 'MelSpectrogram']:
            self.config = {'rate': rate, 'window': window, 'step':step, 'window_func':window_func}

        # get all wav files in the folder, including subfolders
        ext = os.path.splitext(path)[1].lower()
        if ext == '.wav':
            assert os.path.exists(path), '{0} could not find {1}'.format(self.__class__.__name__, path)
            self.files = [path]
        else:
            self.files = find_wave_files(path=path, fullpath=True, subdirs=True)
            assert len(self.files) > 0, '{0} did not find any wave files in {1}'.format(self.__class__.__name__, path)

        self.file_id = -1
        self._next_file()

    def __iter__(self):
        return self

    def __next__(self):
        """ Load next waveform segment or compute next spectrogram.

            Returns: 
                seg: Waveform or Spectrogram
                    Next segment
        """
        seg = self.get(time=self.time, file_id=self.file_id) #get next segment
        self.time += self.seg_step #increment time        
        self.seg_id += 1 #increment segment ID
        if self.seg_id == self.num_segs: self._next_file() #if this was the last segment, jump to the next file
        return seg

    def reset(self):
        """ Go back to the beginning of the first file.
        """
        self.jump(0)

    def jump(self, file_id=0):
        """ Go to the beginning of the selected file.

            Args:
                file_id: int
                    File ID
        """
        self.file_id = file_id - 1
        self._next_file()

    def get(self, time=0, file_id=0):
        """ Load audio segment for specified file and time.

            Args:
                time: float
                    Start time of the spectrogram in seconds, measured from the 
                    beginning of the file.
                file_id: int
                    Integer file identifier.
        
            Returns: 
                : BaseAudio
                    Audio segment
        """
        path = self.files[file_id]
        seg = eval(self.rep).from_wav(path=path, channel=self.channel, offset=time, 
            duration=self.seg_dur, **self.config)
        if isinstance(seg, Spectrogram): seg.crop(**self.freq_cut)
        return seg

    def _next_file(self):
        """ Jump to next file. 
        """
        # increment file ID
        self.file_id = (self.file_id + 1) % len(self.files)

        # check if file exists
        f = self.files[self.file_id]
        exists = os.path.exists(f)

        # if not, jump to the next file
        if not exists: self._next_file()

        # number of segments
        file_duration = librosa.get_duration(filename=f)
        self.num_segs = int(np.ceil((file_duration - self.seg_dur) / self.seg_step)) + 1

        # reset segment ID and time
        self.seg_id = 0
        self.time = 0


