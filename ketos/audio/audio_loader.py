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

""" 'audio.audio_loader' module within the ketos library

    This module contains the utilities for loading waveforms and computing spectrograms.

    Contents:
        AudioLoader class:
        AudioSelectionLoader class:
        AudioSequenceLoader class
"""
import os
import copy
import numpy as np
import librosa
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram,MagSpectrogram,PowerSpectrogram,MelSpectrogram,CQTSpectrogram
from ketos.data_handling.data_handling import find_wave_files
from ketos.data_handling.selection_table import query

class AudioLoader():
    """ Base class for AudioSelectionLoader and AudioSequenceLoader.

        TODO: Add annotations to created Waveform/Spectrogram objects

        Args:
            path: str
                Full path to audio file (*.wav) or folder containing audio files
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            annotations: pandas DataFrame
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
    def __init__(self, path, channel=0, annotations=None, repres={'type': 'Waveform'}):

        repres = copy.deepcopy(repres)
        self.channel = channel
        self.typ = repres.pop('type')
        self.cfg = repres
        self.annot = annotations

        # get all wav files in the folder, including subfolders
        ext = os.path.splitext(path)[1].lower()
        if ext == '.wav':
            assert os.path.exists(path), '{0} could not find {1}'.format(self.__class__.__name__, path)
            self.files = [path]
        else:
            self.files = find_wave_files(path=path, fullpath=True, subdirs=True)
            assert len(self.files) > 0, '{0} did not find any wave files in {1}'.format(self.__class__.__name__, path)

        # create map: filename -> full path
        self.path_dict = {os.path.basename(f) : f for f in self.files}

    def __iter__(self):
        return self

    def __next__(self):
        """ Load next waveform segment or compute next spectrogram.

            Returns: 
                seg: Waveform or Spectrogram
                    Next segment
        """
        offset, duration, fname = self._next_segment()
        return self.load_segment(offset, duration, fname)

    def load_segment(self, offset, duration, path):
        """ Load audio segment for specified file and time.

            Args:
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                path: str
                    Full path to wav file.
        
            Returns: 
                seg: BaseAudio
                    Audio segment
        """
        # load audio
        seg = eval(self.typ).from_wav(path=path, channel=self.channel, offset=offset, 
            duration=duration, **self.cfg)
    
        # add annotations
        if self.annot is not None:
            q = query(self.annot, filename=os.path.basename(path), start=offset, end=offset+duration)
            if len(q) > 0:
                q['start'] = np.maximum(0, q['start'].values - offset)
                q['end']   = np.minimum(q['end'].values - offset, seg.duration())
                seg.annotate(df=q)             

        return seg

    def _next_segment():
        """ Returns offset, duration, and filename of the next segment.
        
            Must be implemented in child classes.

            Returns:
                : float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                : float
                    Duration of segment in seconds.
                : str
                    Full path to wav file.
        """
        pass


class AudioSequenceLoader(AudioLoader):
    """ Load segments of audio data from *.wav files. 

        Several representations of the audio data are possible, including 
        waveform, magnitude spectrogram, power spectrogram, mel spectrogram, 
        and CQT spectrogram.

        Segments of length 'window' will be loaded in chronological order, with 
        every segment displaced by an amount 'step' relative to the previous 
        segment. (If 'step' is not specified, it is set equal to 'window'.)

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
            annotations: pandas DataFrame
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
    def __init__(self, path, window, step=None, channel=0, annotations=None, repres={'type': 'Waveform'}):

        super().__init__(path=path, channel=channel, annotations=annotations, repres=repres)

        self.window = window
        if step is None: self.step = window
        else: self.step = step

        self.file_id = -1
        self._next_file()

    def _next_segment(self):
        """ Returns start and end times and file name of the next segment.
        
            Returns:
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                path: str
                    Full path to wav file.
        """
        offset = self.time
        path   = self.files[self.file_id]
        self.time += self.step #increment time       
        self.seg_id += 1 #increment segment ID
        if self.seg_id == self.num_segs: self._next_file() #if this was the last segment, jump to the next file
        return offset, self.window, path

    def _next_file(self):
        """ Jump to next file. 
        """
        self.file_id = (self.file_id + 1) % len(self.files) #increment file ID
        file_duration = librosa.get_duration(filename=self.files[self.file_id]) #file duration
        self.num_segs = int(np.ceil((file_duration - self.window) / self.step)) + 1  #number of segments
        self.seg_id = 0 #reset
        self.time = 0 #reset


class AudioSelectionLoader(AudioLoader):
    """ Load segments of audio data from *.wav files. 

        Several representations of the audio data are possible, including 
        waveform, magnitude spectrogram, power spectrogram, mel spectrogram, 
        and CQT spectrogram.

        The segments to be loaded are specified via a selection table.

        Args:
            path: str
                Full path to audio file (*.wav) or folder containing audio files
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            selections: pandas DataFrame
                Selection table
            annotations: pandas DataFrame
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
    def __init__(self, path, selections, channel=0, annotations=None, repres={'type': 'Waveform'}):

        super().__init__(path=path, channel=channel, annotations=annotations, repres=repres)

        self.selec  = selections
        self.row_id = 0

    def _next_segment(self):
        """ Returns start and end times and file name of the next segment.
        
            Returns:
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                path: str
                    Full path to wav file.
        """
        fname    = self.selec.index.values[self.row_id][0]
        path     = self.path_dict[fname]
        s = self.selec.iloc[self.row_id]
        offset   = s['start']
        duration = s['end'] - s['start']
        self.row_id = (self.row_id + 1) % len(self.selec)
        return offset, duration, path
