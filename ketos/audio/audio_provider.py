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
import copy
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
            window: float
                Segment length in seconds.
            step: float
                Separation between consecutive segments in seconds. If None, the step size 
                equals the segment length.
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            annot: pandas DataFrame
                Annotation table
            repres: dict
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the from_wav method.  
                
                    * Waveform: 
                        (rate), (resample_method)
                    
                    * MagSpectrogram, PowerSpectrogram, MelSpectrogram: 
                        window, step, (window_func), (rate), (resample_method)
                    
                    * CQTSpectrogram:
                        step, bins_per_oct, (freq_min), (freq_max), (window_func), (rate), (resample_method)
    """
    def __init__(self, path, window, step=None, channel=0, annot=None, repres={'type': 'Waveform'}):

        repres = copy.deepcopy(repres)
        self.channel = channel
        self.typ = repres.pop('type')
        self.cfg = repres
        self.window = window
        if step is None: self.step = window
        else: self.step = step
        self.annot = annot

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
        self.time += self.step #increment time        
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
        return eval(self.typ).from_wav(path=path, channel=self.channel, offset=time, 
            duration=self.window, **self.cfg)

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
        self.num_segs = int(np.ceil((file_duration - self.window) / self.step)) + 1

        # reset segment ID and time
        self.seg_id = 0
        self.time = 0


