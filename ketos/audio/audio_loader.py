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
import warnings
from ketos.audio.waveform import Waveform
from ketos.audio.spectrogram import Spectrogram,MagSpectrogram,PowerSpectrogram,MelSpectrogram,CQTSpectrogram
from ketos.data_handling.data_handling import find_wave_files
from ketos.data_handling.selection_table import query


""" Audio representation dictionary 
"""
audio_repres_dict = {'Waveform':Waveform,
                     'MagSpectrogram':MagSpectrogram, 
                     'Mag':MagSpectrogram,
                     'PowerSpectrogram':PowerSpectrogram,
                     'Power':PowerSpectrogram,
                     'Pow':PowerSpectrogram,
                     'MelSpectrogram':MelSpectrogram,
                     'Mel':MelSpectrogram,
                     'CQTSpectrogram':CQTSpectrogram,
                     'CQT':CQTSpectrogram}


class SelectionGenerator():
    """ Template class for selection generators.
    """
    def __iter__(self):
        return self

    def __next__(self):
        """ Returns offset, duration, data_dir, filename, and label (if available) 
            of the next audio selection.
        
            Must be implemented in child class.

            Returns:
                : float
                    Start time of the selection in seconds, measured from the 
                    beginning of the file.
                : float
                    Duration of the selection in seconds.
                : str
                    Data directory
                : str
                    Filename or relative path
                : int
                    Label (if available)
        """
        pass

    def num(self):
        """ Returns total number of selections.
        
            Must be implemented in child class.

            Returns:
                : int
                    Total number of selections.
        """
        pass

    def reset(self):
        """ Resets the selection generator to the beginning.
        """        
        pass

class SelectionTableIterator(SelectionGenerator):
    """ Iterates over entries in a selection table.

        Args: 
            data_dir: str
                Path to top folder containing audio files.
            selection_table: pandas DataFrame
                Selection table
            duration: float
                Use this argument to enforce uniform duration of all selections.
                Any selection longer than the specified duration will be shortened
    """
    def __init__(self, data_dir, selection_table, duration=None):
        self.sel = selection_table
        self.duration = duration
        self.dir = data_dir
        self.row_id = 0

    def __next__(self):
        """ Returns offset, duration, file path, and label (if available) 
            of the next audio selection.
        
            Returns:
                offset: float
                    Start time of the selection in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of the selection in seconds.
                data_dir: str
                    Data directory
                filename: str
                    Filename or relative path
                label: int
                    Label
        """
        filename = self.sel.index.values[self.row_id][0]
        # current row
        s = self.sel.iloc[self.row_id]
        # start time
        if 'start' in s.keys(): offset = s['start']
        else: offset = 0
        # duration
        if self.duration is not None: duration = self.duration
        elif 'end' in s.keys(): duration = s['end'] - offset
        else: duration = None
        # label
        if 'label' in self.sel.columns.values: label = s['label']
        else: label = None
        # update row no.
        self.row_id = (self.row_id + 1) % len(self.sel)
        return offset, duration, self.dir, filename, label

    def num(self):
        """ Returns total number of selections.
        
            Returns:
                : int
                    Total number of selections.
        """
        return len(self.sel)

    def reset(self):
        """ Resets the selection generator to the beginning of the selection table.
        """        
        self.row_id = 0

class FrameStepper(SelectionGenerator):
    """ Generates selections with uniform duration 'frame', with successive selections 
        displaced by a fixed amount 'step' (If 'step' is not specified, it is set equal 
        to 'frame'.)

        Args: 
            frame: float
                Frame length in seconds.
            step: float
                Separation between consecutive frames in seconds. If None, the step size 
                equals the frame length.
            path: str
                Path to folder containing .wav files. If None is specified, the current directory will be used.
            filename: str or list(str)
                Relative path to a single .wav file or a list of .wav files. Optional.
    """
    def __init__(self, frame, step=None, path=None, filename=None):
        self.frame = frame
        if step is None: self.step = frame
        else: self.step = step

        if path is None: path = os.getcwd()

        # get all wav files in the folder, including subfolders
        if filename is None:
            self.dir = path
            self.files = find_wave_files(path=path, return_path=True, search_subdirs=True)
            assert len(self.files) > 0, '{0} did not find any wave files in {1}'.format(self.__class__.__name__, path)

        else:
            if isinstance(filename, str):
                fullpath = os.path.join(path,filename)
                assert os.path.exists(fullpath), '{0} could not find {1}'.format(self.__class__.__name__, fullpath)
                self.dir = os.path.dirname(fullpath)
                self.files = [os.path.basename(fullpath)]
            else:                
                assert isinstance(filename, list), 'filename must be str or list(str)'        
                self.dir = path
                self.files = filename

        # get file durations
        file_durations = np.array([librosa.get_duration(filename=os.path.join(self.dir, f)) for f in self.files])

        # discard any files with 0 second duration
        self.files = np.array(self.files)[file_durations > 0].tolist()
        file_durations = file_durations[file_durations > 0].tolist()

        # obtain file durations and compute number of frames for each file
        self.num_segs = [int(np.ceil((dur - self.frame) / self.step)) + 1 for dur in file_durations]
        self.num_segs_tot = np.sum(np.array(self.num_segs))

        self.reset()

    def __next__(self):
        """ Returns offset, duration, and file path of the next audio selection.
        
            Returns:
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                data_dir: str
                    Data directory
                filename: str
                    Filename or relative path
                : None
        """
        offset   = self.time
        filename = self.files[self.file_id]
        self.time += self.step #increment time       
        self.seg_id += 1 #increment segment ID
        if self.seg_id == self.num_segs[self.file_id]: self._next_file() #if this was the last segment, jump to the next file
        return offset, self.frame, self.dir, filename, None

    def num(self):
        """ Returns total number of selections.
        
            Returns:
                : int
                    Total number of selections.
        """
        return self.num_segs_tot

    def _next_file(self):
        """ Jump to next file. 
        """
        self.file_id = (self.file_id + 1) % len(self.files) #increment file ID
        self.seg_id = 0 #reset
        self.time = 0 #reset

    def reset(self):
        """ Resets the selection generator to the beginning of the first file.
        """        
        self.file_id = -1
        self._next_file()

class AudioLoader():
    """ Class for loading segments of audio data from .wav files. 

        Several representations of the audio data are possible, including 
        waveform, magnitude spectrogram, power spectrogram, mel spectrogram, 
        and CQT spectrogram.

        Args:
            selection_gen: SelectionGenerator
                Selection generator
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

                Optionally, may also contain the key 'normalize_wav' which can have value True or False. 
                If True, the waveform is normalized zero mean (mean=0) and (std=1) unity standard deviation.

        Examples:
            See child classes :class:`audio.audio_loader.AudioFrameLoader` and 
            :class:`audio.audio_loader.AudioSelectionLoader`.            
    """
    def __init__(self, selection_gen, channel=0, annotations=None, repres={'type': 'Waveform'}):

        repres = copy.deepcopy(repres)
        self.channel = channel
        self.typ = repres.pop('type')
        if 'duration' in repres.keys(): repres.pop('duration')
        self.cfg = repres
        self.sel_gen = selection_gen
        self.annot = annotations

    def __iter__(self):
        return self

    def __next__(self):
        """ Load next waveform segment or compute next spectrogram.

            Returns: 
                : Waveform or Spectrogram
                    Next segment
        """
        offset, duration, data_dir, filename, label = next(self.sel_gen)
        return self.load(offset, duration, data_dir, filename, label)

    def num(self):
        """ Returns total number of segments.
        
            Returns:
                : int
                    Total number of segments.
        """
        return self.sel_gen.num()

    def load(self, offset, duration, data_dir, filename, label):
        """ Load audio segment for specified file and time.

            Args:
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                data_dir: str
                    Data directory
                filename: str
                    Filename or relative path
                label: int
                    Integer label
        
            Returns: 
                seg: BaseAudio
                    Audio segment
        """
        path = os.path.join(data_dir, filename)

        # load audio
        # (ignore warnings from the from_wav method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            seg = audio_repres_dict[self.typ].from_wav(path=path, channel=self.channel, offset=offset, 
                                                        duration=duration, id=filename, **self.cfg)
    
        # add annotations
        if label is not None:
            seg.label = label

        if self.annot is not None:
            q = query(self.annot, filename=os.path.basename(path), start=offset, end=offset+duration)
            if len(q) > 0:
                q['start'] = np.maximum(0, q['start'].values - offset)
                q['end']   = np.minimum(q['end'].values - offset, seg.duration())
                seg.annotate(df=q)             

        return seg

    def reset(self):
        """ Resets the audio loader to the beginning.
        """        
        self.sel_gen.reset()

class AudioFrameLoader(AudioLoader):
    """ Load segments of audio data from .wav files. 

        Loads segments of uniform duration 'frame', with successive segments
        displaced by an amount 'step'. (If 'step' is not specified, it is 
        set equal to 'frame'.)

        Args:
            frame: float
                Segment duration in seconds. Can also be specified via the 'duration' 
                item of the 'repres' dictionary.
            step: float
                Separation between consecutive segments in seconds. If None, the step size 
                equals the segment duration.
            path: str
                Path to folder containing .wav files. If None is specified, the current directory will be used.
            filename: str or list(str)
                relative path to a single .wav file or a list of .wav files. Optional
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            annotations: pandas DataFrame
                Annotation table
            repres: dict
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the from_wav method.  
            batch_size: int or str
                Load segments in batches rather than one at the time. 
                Increasing the batch size can help reduce computational time.
                The default batch size is 1. 
                You can also specify batch_size='file' to load one wav file at the time.

        Examples:
            >>> import librosa
            >>> from ketos.audio.audio_loader import AudioFrameLoader
            >>> # specify path to wav file
            >>> filename = 'ketos/tests/assets/2min.wav'
            >>> # check the duration of the audio file
            >>> print(librosa.get_duration(filename=filename))
            120.832
            >>> # specify the audio representation
            >>> rep = {'type':'MagSpectrogram', 'window':0.2, 'step':0.02, 'window_func':'hamming', 'freq_max':1000.}
            >>> # create an object for loading 30-s long spectrogram segments, using a step size of 15 s (50% overlap) 
            >>> loader = AudioFrameLoader(frame=30., step=15., filename=filename, repres=rep)
            >>> # print number of segments
            >>> print(loader.num())
            8
            >>> # load and plot the first segment
            >>> spec = next(loader)
            >>>
            >>> import matplotlib.pyplot as plt
            >>> fig = spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec_2min_0.png")
            >>> plt.close(fig)
            
            .. image:: ../../../../ketos/tests/assets/tmp/spec_2min_0.png
    """
    def __init__(self, frame=None, step=None, path=None, filename=None, channel=0, 
                    annotations=None, repres={'type': 'Waveform'}, batch_size=1):

        assert 'duration' in repres.keys() or frame is not None, 'duration must be specified either via the frame argument or the duration item of the repres dictionary'

        if frame is None: frame = repres['duration']

        if 'duration' in repres.keys() and repres['duration'] is not None and repres['duration'] != frame:
            print("Warning: Mismatch between frame size ({0:.3f} s) and duration ({1:.3f} s). The latter value will be ignored.")

        assert (isinstance(batch_size, int) and batch_size >= 1) or (isinstance(batch_size, str) and batch_size.lower() == 'file'), 'Batch size must be a positive integer or have the string value file'

        super().__init__(selection_gen=FrameStepper(frame=frame, step=step, path=path, filename=filename), 
            channel=channel, annotations=annotations, repres=repres)

        if isinstance(batch_size, int):
            self.max_batch_size = batch_size
        else:
            self.max_batch_size = np.inf

        if self.max_batch_size > 1:
            self.offset, _, self.data_dir, self.filename, _ = next(self.sel_gen)
            self.load_next_batch()

    def __next__(self):
        """ Load next waveform segment or compute next spectrogram.

            Returns: 
                : Waveform or Spectrogram
                    Next segment
        """
        if self.max_batch_size == 1:
            return super().__next__()        
        else:
            return self.next_in_batch()

    def load_next_batch(self):
        """ Load the next batch of waveforms or spectrograms.
        """
        self.batch_size = 0
        self.counter = 0
        offset = np.inf
        data_dir = self.data_dir
        filename = self.filename
        while data_dir == self.data_dir and filename == self.filename and offset > self.offset and self.batch_size < self.max_batch_size:
            self.batch_size += 1
            offset, _, data_dir, filename, _ = next(self.sel_gen)

        duration = self.sel_gen.frame + self.sel_gen.step * (self.batch_size - 1)
        self.batch = self.load(self.offset, duration, self.data_dir, self.filename, label=None)
        self.batch = self.batch.segment(window=self.sel_gen.frame, step=self.sel_gen.step)

        self.offset = offset
        self.data_dir = data_dir
        self.filename = filename

    def next_in_batch(self):
        """ Load the next waveform or spectrogram in the batch.
        
            Returns: 
                a: Waveform or Spectrogram
                    Next segment
        """
        if self.counter >= self.batch.num_objects(): self.load_next_batch()
        a = self.batch.get(self.counter)
        self.counter += 1
        return a


class AudioSelectionLoader(AudioLoader):
    """ Load segments of audio data from .wav files. 

        The segments to be loaded are specified via a selection table.

        Args:
            selections: pandas DataFrame
                Selection table
            path: str
                Path to folder containing .wav files
            filename: str or list(str)
                relative path to a single .wav file or a list of .wav files. Optional
            annotations: pandas DataFrame
                Annotation table
            repres: dict
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the from_wav method.  
    """
    def __init__(self, path, selections, channel=0, annotations=None, repres={'type': 'Waveform'}):

        if 'duration' in repres.keys(): duration = repres['duration']
        else: duration = None

        super().__init__(selection_gen=SelectionTableIterator(data_dir=path, selection_table=selections, duration=duration), 
            channel=channel, annotations=annotations, repres=repres)