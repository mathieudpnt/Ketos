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
from ketos.audio.gammatone import GammatoneFilterBank,AuralFeatures
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
                     'CQT':CQTSpectrogram,
                     'AuralFeatures': AuralFeatures,
                     'Aural': AuralFeatures,
                     'GammatoneFilterBank': GammatoneFilterBank,
                     'Gammatone': GammatoneFilterBank}


class SelectionGenerator():
    """ Template class for selection generators.
    """
    def __iter__(self):
        return self

    def __next__(self):
        """ Returns the next audio selection.

            An audio selection is represented as a dictionary 
            with two required keys (data_dir, filename) and 
            an unlimited number of optional keys, which typically 
            include offset, duration, and label.
        
            Must be implemented in child class.

            Returns:
                : dict()
                    Next audio selection
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
            include_attrs: bool
                If True, load data from all attribute columns in the selection table. Default is False.
            attrs: list(str)
                Specify the names of the attribute columns that you wish to load data from. 
                Overwrites include_attrs if specified. If None, all columns will be loaded provided that 
                include_attrs=True.
    """
    def __init__(self, data_dir, selection_table, duration=None, include_attrs=False, attrs=None):
        self.sel = selection_table
        self.duration = duration
        self.dir = data_dir
        self.row_id = 0

        all_attrs = list(self.sel.columns.values)
        for col in ['start', 'end', 'label']: 
            if col in all_attrs: all_attrs.remove(col)

        if attrs is not None:
            for col in attrs: 
                if col not in all_attrs: attrs.remove(col)
            self.attrs = attrs
        elif include_attrs:
            self.attrs = all_attrs
        else:
            self.attrs = []

    def __next__(self):
        """ Returns the next audio selection.

            Returns:
                audio_sel: dict
                    Audio selection
        """
        audio_sel = self.get_selection(id=self.row_id)
        self.row_id = (self.row_id + 1) % len(self.sel) #update row no.
        return audio_sel

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
        
    def get_selection(self, id):
        """ Returns the audio selection with a given id.

            Args:
                id: int
                    The id within the selection table to be searched        
            Returns:
                audio_sel: dict
                    Audio selection
        """
        audio_sel = {'data_dir': self.dir}
        audio_sel['filename'] = self.sel.index.values[id][0]
        # current row
        s = self.sel.iloc[id]
        # start time
        if 'start' in s.keys(): offset = s['start']
        else: offset = 0
        audio_sel['offset'] = offset
        # duration
        if self.duration is not None: audio_sel['duration'] = self.duration
        elif 'end' in s.keys(): audio_sel['duration'] = s['end'] - offset
        # label
        if 'label' in self.sel.columns.values: audio_sel['label'] = s['label']
        # attribute columns
        for col in self.attrs: audio_sel[col] = s[col]
        return audio_sel


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
        """ Returns the next audio selection.
        
            Returns:
                audio_sel: dict
                    Audio selection
        """
        audio_sel = {'data_dir':self.dir, 'filename': self.files[self.file_id], 'offset':self.time, 'duration':self.frame}
        self.time += self.step #increment time       
        self.seg_id += 1 #increment segment ID
        if self.seg_id == self.num_segs[self.file_id]: self._next_file() #if this was the last segment, jump to the next file
        return audio_sel

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
                It is also possible to specify multiple audio presentations as a list.
        Examples:
            See child classes :class:`audio.audio_loader.AudioFrameLoader` and 
            :class:`audio.audio_loader.AudioSelectionLoader`.            
    """
    def __init__(self, selection_gen, channel=0, annotations=None, repres={'type': 'Waveform'}, **kwargs):
        repres = copy.deepcopy(repres)
        if not isinstance(repres, list): repres = [repres]
        self.typ, self.cfg = [], []
        for r in repres:
            self.typ.append(r.pop('type'))
            if 'duration' in r.keys(): r.pop('duration')
            self.cfg.append(r)
        self.channel = channel
        self.sel_gen = selection_gen
        self.annot = annotations
        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        """ Load next waveform segment or compute next spectrogram.

            Returns: 
                : Waveform or Spectrogram
                    Next segment
        """
        audio_sel = next(self.sel_gen)
        return self.load(**audio_sel, **self.kwargs)

    def num(self):
        """ Returns total number of segments.
        
            Returns:
                : int
                    Total number of segments.
        """
        return self.sel_gen.num()

    def load(self, data_dir, filename, offset=0, duration=None, label=None, **kwargs):
        """ Load audio segment for specified file and time.

            Args:
                data_dir: str
                    Data directory
                filename: str
                    Filename or relative path
                offset: float
                    Start time of the segment in seconds, measured from the 
                    beginning of the file.
                duration: float
                    Duration of segment in seconds.
                label: int
                    Integer label
        
            Returns: 
                seg: BaseAudio
                    Audio segment
        """
        path = os.path.join(data_dir, filename)

        # load audio
        # (ignore warnings from the from_wav method)
        segs = []
        for typ,cfg in zip(self.typ, self.cfg):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")        
                _kwargs = kwargs.copy()
                _kwargs.update(cfg)
                seg = audio_repres_dict[typ].from_wav(path=path, channel=self.channel, offset=offset, 
                                                            duration=duration, id=filename, **_kwargs)
        
            # add annotations
            if label is not None:
                seg.label = label

            if self.annot is not None:
                q = query(self.annot, filename=os.path.basename(path), start=offset, end=offset+duration)
                if len(q) > 0:
                    q['start'] = np.maximum(0, q['start'].values - offset)
                    q['end']   = np.minimum(q['end'].values - offset, seg.duration())
                    seg.annotate(df=q)  

            segs.append(seg)           

        if len(segs)==1: segs=segs[0]

        return segs

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
                It is also possible to specify multiple audio presentations as a list. These 
                presentations must have the same duration.
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
                    annotations=None, repres={'type': 'Waveform'}, batch_size=1, **kwargs):

        if isinstance(repres, list): r0 = repres[0]
        else: r0 = repres

        assert 'duration' in r0.keys() or frame is not None, 'duration must be specified either via the frame argument or the duration item of the repres dictionary'

        if frame is None: frame = r0['duration']

        if 'duration' in r0.keys() and r0['duration'] is not None and r0['duration'] != frame:
            print("Warning: Mismatch between frame size ({0:.3f} s) and duration ({1:.3f} s). The latter value will be ignored.")

        assert (isinstance(batch_size, int) and batch_size >= 1) or (isinstance(batch_size, str) and batch_size.lower() == 'file'), 'Batch size must be a positive integer or have the string value file'

        super().__init__(selection_gen=FrameStepper(frame=frame, step=step, path=path, filename=filename), 
            channel=channel, annotations=annotations, repres=repres, **kwargs)

        if isinstance(batch_size, int):
            self.max_batch_size = batch_size
        else:
            self.max_batch_size = np.inf

        if self.max_batch_size > 1:
            audio_sel = next(self.sel_gen)
            self.offset = audio_sel['offset']
            self.data_dir = audio_sel['data_dir']
            self.filename = audio_sel['filename']
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
            audio_sel = next(self.sel_gen)
            offset = audio_sel['offset']
            data_dir = audio_sel['data_dir']
            filename = audio_sel['filename']            

        duration = self.sel_gen.frame + self.sel_gen.step * (self.batch_size - 1)
        self.batch = self.load(data_dir=self.data_dir, filename=self.filename, offset=self.offset, duration=duration, label=None, **self.kwargs)
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
        if self.counter >= len(self.batch): self.load_next_batch()
        a = self.batch[self.counter]
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
                It is also possible to specify multiple audio presentations as a list.
            include_attrs: bool
                If True, load data from all attribute columns in the selection table. Default is False.
            attrs: list(str)
                Specify the names of the attribute columns that you wish to load data from. 
                Overwrites include_attrs if specified. If None, all columns will be loaded provided that 
                include_attrs=True.
    """
    def __init__(self, path, selections, channel=0, annotations=None, repres={'type': 'Waveform'}, 
        include_attrs=False, attrs=None, **kwargs):

        if isinstance(repres, list): r0 = repres[0]
        else: r0 = repres

        if 'duration' in r0.keys(): duration = r0['duration']
        else: duration = None

        super().__init__(selection_gen=SelectionTableIterator(data_dir=path, selection_table=selections, duration=duration, include_attrs=include_attrs, attrs=attrs), 
            channel=channel, annotations=annotations, repres=repres, **kwargs)