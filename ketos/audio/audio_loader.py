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
                Any selection longer than the specified duration will be shortened at the end.
                Any selection shorter than the specified duration will be extended at the end.
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
    """ Generates selections with uniform length 'duration', with successive selections 
        displaced by a fixed amount 'step' (If 'step' is not specified, it is set equal 
        to 'duration'.)

        Args: 
            duration: float
                Selection length in seconds.
            step: float
                Separation between consecutive selections in seconds. If None, the step size 
                equals the selection length.
            path: str
                Path to folder containing .wav files. If None is specified, the current directory will be used.
            filename: str or list(str)
                Relative path to a single .wav file or a list of .wav files. Optional.
            frame: float
                Same as duration. Only included for backward compatibility. Will be removed in future versions.
    """
    def __init__(self, duration=None, step=None, path=None, filename=None, frame=None):
        assert duration is not None or frame is not None, "Either duration or frame must be specified"
        
        if frame is not None:
            print("Warning: frame is deprecated and will be removed in a future versions. Use duration instead")
            if duration is None:
                duration = frame
            
        self.duration = duration
        if step is None: self.step = duration
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
        self.file_durations = np.array([librosa.get_duration(filename=os.path.join(self.dir, f)) for f in self.files])

        # discard any files with 0 second duration
        self.files = np.array(self.files)[self.file_durations > 0].tolist()
        self.file_durations = self.file_durations[self.file_durations > 0].tolist()

        # obtain file durations and compute number of frames for each file
        self.num_segs = [int(np.ceil((dur - self.duration) / self.step)) + 1 for dur in self.file_durations]
        self.num_segs_tot = np.sum(np.array(self.num_segs))

        self.reset()

    def __next__(self):
        """ Returns the next audio selection.
        
            Returns:
                audio_sel: dict
                    Audio selection
        """
        audio_sel = {'data_dir':self.dir, 'filename': self.files[self.file_id], 'offset':self.time, 'duration':self.duration}
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

    def get_file_paths(self, fullpath=True):
        """ Get the paths to the audio files associated with this instance.

            Args:
                fullpath: bool
                    Whether to return the full path (default) or only the filename.

            Returns:
                ans: list
                    List of file paths
        """
        if fullpath:
            ans = [os.path.join(self.dir, f) for f in self.files]
        else:
            ans = self.files

        return ans

    def get_file_durations(self):
        """ Get the durations of the audio files associated with this instance.

            Returns:
                ans: list
                    List of file durations in seconds
        """
        return self.file_durations


class AudioLoader():
    """ Class for loading segments of audio data from .wav files. 

        Several representations of the audio data are possible, including 
        waveform, magnitude spectrogram, power spectrogram, mel spectrogram, 
        and CQT spectrogram.

        TODO: Change default value of `stop` argument to True.

        Args:
            selection_gen: SelectionGenerator
                Selection generator
            channel: int
                For stereo recordings, this can be used to select which channel to read from
            annotations: pandas DataFrame
                Annotation table
            repres: dict
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the `from_wav` method.  
                
                    * Waveform: 
                        (rate), (resample_method)
                    
                    * MagSpectrogram, PowerSpectrogram, MelSpectrogram: 
                        window, step, (window_func), (rate), (resample_method)
                    
                    * CQTSpectrogram:
                        step, bins_per_oct, (freq_min), (freq_max), (window_func), (rate), (resample_method)

                Optionally, may also contain the key 'normalize_wav' which can have value True or False. 
                If True, the waveform is normalized zero mean (mean=0) and (std=1) unity standard deviation.
                It is also possible to specify multiple audio presentations as a list.
            batch_size: int
                Load segments in batches rather than one at the time. 
            stop: bool
                Raise StopIteration when all selections have been loaded. Default is False.

        Attributes:
            cfg: list(dict)
                Audio representation dictionaries.
        
        Examples:
            See child classes :class:`audio.audio_loader.AudioFrameLoader` and 
            :class:`audio.audio_loader.AudioSelectionLoader`.            
    """
    def __init__(self, selection_gen, channel=0, annotations=None, repres={'type': 'Waveform'}, 
                        batch_size=1, stop=False, **kwargs):
        repres = copy.deepcopy(repres)
        if not isinstance(repres, list): repres = [repres]
        self.typ, self.cfg = [], []
        for r in repres:
            self.typ.append(r.pop('type'))
            if 'duration' in r.keys(): r.pop('duration')
            self.cfg.append(r)

        self.channel = channel
        self.selection_gen = selection_gen
        self.annot = annotations
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.stop = stop
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        """ Load next audio segment or batch of audio segments.

            Depending on how the loader was initialized, the return value can either be 
            an instance of :class:`BaseAudio <ketos.audio.base_audio.BaseAudio>` (or, 
            more commonly, a instance of one of its derived classes such as the 
            :class:`Waveform <ketos.audio.waveform.Waveform>` or 
            :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`
            classes), a list of such objects, or a nested listed of such objects. 

            Some examples:

             * If the loader was initialized with the audio representation `repres={'type':'Waveform'}` 
               and with `batch_size=1` (default), the return value will be a single 
               instance of :class:`Waveform <ketos.audio.waveform.Waveform>`.

             * If the loader was initialized with the audio representation 
               `repres=[{'type':'Waveform'}, {'type':'MagSpectrogram', 'window':0.1,'step':0.02}]` 
               and with `batch_size=1` (default), the return value will be a list 
               of length 2, where the first entry holds an instance of 
               :class:`Waveform <ketos.audio.waveform.Waveform>` and the second entry holds an instance 
               of :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`.

             * If the loader was initialized with the audio representation 
               `repres=[{'type':'Waveform'}, {'type':'MagSpectrogram', 'window':0.1,'step':0.02}]` 
               and with `batch_size>1`, the return value will be a nested list with outer 
               length equal to `batch_size` and inner length 2, corresponding to the number of 
               audio representations.

            If the loader was initialized with `stop=True` this method will raise `StopIteration` 
            when all the selections have been loaded.

            Returns: 
                a: BaseAudio, list(BaseAudio), or list(list(BaseAudio))
                    Next segment or next batch of segments
        """
        return self._next_batch(load=True)

    def skip(self):
        """ Skip to the next audio segment or batch of audio segments
            without loading the current one.
        """        
        self._next_batch(load=False)

    def _next_batch(self, load=True):
        """ Load next audio segment or batch of audio segments.

            Helper function for :meth:`__next()__` and :meth:`skip()`.

            Args:
                load: bool
                    Whether to load the audio data.
        """
        if self.counter == self.num():
            if self.stop:
                raise StopIteration
            else:
                self.reset()

        a = []
        for _ in range(self.batch_size):
            if self.counter < self.num():
                selection = next(self.selection_gen)
                if load:
                    a.append(self.load(**selection, **self.kwargs))
                self.counter += 1

        if load:
            if self.batch_size == 1: a = a[0]
            return a

    def num(self):
        """ Returns total number of segments.
        
            Returns:
                : int
                    Total number of segments.
        """
        return self.selection_gen.num()

    def load(self, data_dir, filename, offset=0, duration=None, label=None, apply_transforms=True, **kwargs):
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
                apply_transforms: bool
                    Apply transforms. Default is True.
        
            Returns: 
                seg: BaseAudio or list(BaseAudio)
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
                if not apply_transforms and 'transforms' in _kwargs.keys(): del _kwargs['transforms']
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

        if len(segs) == 1: segs = segs[0]

        return segs

    def reset(self):
        """ Resets the audio loader to the beginning.
        """        
        self.selection_gen.reset()
        self.counter = 0


class AudioFrameLoader(AudioLoader):
    """ Load audio segments by sliding a fixed-size frame across the recording.

        The frame size is specified with the 'duration' argument, while the 'step'
        argument may be used to specify the step size. (If 'step' is not specified, 
        it is set equal to 'duration'.)

        TODO: Remove frame argument
        TODO: Change default value of `stop` argument to True.

        Args:
            duration: float
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
            batch_size: int
                Load segments in batches rather than one at the time. 
            stop: bool
                Raise StopIteration if the iteration exceeds the number of available selections. Default is False.
            frame: float
                Same as duration. Only included for backward compatibility. Will be removed in future versions.

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
            >>> loader = AudioFrameLoader(duration=30., step=15., filename=filename, repres=rep)
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
            
            .. image:: ../../../ketos/tests/assets/tmp/spec_2min_0.png
    """
    def __init__(self, duration=None, step=None, path=None, filename=None, channel=0, 
                    annotations=None, repres={'type': 'Waveform'}, batch_size=1, 
                    stop=False, frame=None, **kwargs):

        if frame != None:
            print("Warning: frame is deprecated and will be removed in a future versions. Use duration instead")
            if duration == None: duration = frame

        if batch_size > 1:
            print("Warning: batch_size > 1 results in different behaviour for ketos versions >= 2.4.2 than earlier \
                   versions. You may want to check out the AudioFrameEfficientLoader class.")

        duration = _get_duration(repres, duration)

        assert duration != None, 'duration must be specified either with the duration \
            argument or in the audio representation dictionary'

        super().__init__(selection_gen=FrameStepper(duration=duration, step=step, path=path, filename=filename), 
            channel=channel, annotations=annotations, repres=repres, batch_size=batch_size, stop=stop, **kwargs)

    def get_file_paths(self, fullpath=True):
        """ Get the paths to the audio files associated with this instance.

            Args:
                fullpath: bool
                    Whether to return the full path (default) or only the filename.

            Returns:
                ans: list
                    List of file paths
        """
        return self.selection_gen.get_file_paths(fullpath=fullpath)

    def get_file_durations(self):
        """ Get the durations of the audio files associated with this instance.

            Returns:
                ans: list
                    List of file durations in seconds
        """
        return self.selection_gen.get_file_durations()


class AudioFrameEfficientLoader(AudioFrameLoader):
    """ Load audio segments by sliding a fixed-size frame across the recording.

        AudioFrameEfficientLoader implements a more efficient approach to loading 
        overlapping audio segments and converting them to spectrograms. 
        Rather than loading and converting one frame at the time, the 
        AudioFrameEfficientLoader loads a longer frame and converts it to a 
        spectrogram which is split up into the desired shorter frames.

        Use the `num_frames` argument to specify how many frames are loaded into 
        memory at a time.

        While the segments are loaded into memory in batches, they are by default 
        returned one at a time. Use the `return_as_batch` argument to change this
        behaviour.

        Args:
            duration: float
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
                Annotation table. Optional.
            repres: dict
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the from_wav method.  
                It is also possible to specify multiple audio presentations as a list. These 
                presentations must have the same duration.
            num_frames: int
                Load segments in batches of size `num_frames` rather than one at the time. 
                Increasing `num_frames` can help reduce computational time.
                You can also specify `num_frames='file'` to load one wav file at the time.
            return_as_batch: bool
                Whether to return the segments individually or in batches of size `num_frames`.
                The default behaviour is to return the segments individually.
    """
    def __init__(self, duration=None, step=None, path=None, filename=None, channel=0, 
                    annotations=None, repres={'type': 'Waveform'}, num_frames=12, 
                    return_as_batch=False, **kwargs):

        assert (isinstance(num_frames, int) and num_frames >= 1) or \
            (isinstance(num_frames, str) and num_frames.lower() == 'file'), \
            'Argument `num_frames` must be a positive integer or have the string value `file`'

        super().__init__(duration=duration, step=step, path=path, filename=filename, 
                    channel=channel, annotations=annotations, repres=repres, **kwargs)

        self.return_as_batch = return_as_batch

        self.transforms_list = []
        for config in self.cfg:
            transforms = config['transforms'] if 'transforms' in config.keys() else []
            self.transforms_list.append(transforms)

        if isinstance(num_frames, int):
            self.max_batch_size = num_frames
        else:
            self.max_batch_size = np.inf

        audio_sel = next(self.selection_gen)
        self.offset = audio_sel['offset']
        self.data_dir = audio_sel['data_dir']
        self.filename = audio_sel['filename']

    def __next__(self):
        """ Load the next audio segment or batch of audio segments.

            Depending on how the loader was initialized, the return value can either be 
            an instance of :class:`BaseAudio <ketos.audio.base_audio.BaseAudio>` (or, 
            more commonly, a instance of one of its derived classes such as the 
            :class:`Waveform <ketos.audio.waveform.Waveform>` or 
            :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`
            classes), a list of such objects, or a nested listed of such objects. 

            Some examples:

             * If the loader was initialized with the audio representation `repres={'type':'Waveform'}` 
               and with `return_as_batch=False` (default), the return value will be a single 
               instance of :class:`Waveform <ketos.audio.waveform.Waveform>`.

             * If the loader was initialized with the audio representation 
               `repres=[{'type':'Waveform'}, {'type':'MagSpectrogram', 'window':0.1,'step':0.02}]` 
               and with `return_as_batch=False` (default), the return value will be a list 
               of length 2, where the first entry holds an instance of 
               :class:`Waveform <ketos.audio.waveform.Waveform>` and the second entry holds an instance 
               of :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`.

             * If the loader was initialized with the audio representation 
               `repres=[{'type':'Waveform'}, {'type':'MagSpectrogram', 'window':0.1,'step':0.02}]` 
               and with `return_as_batch=True`, the return value will be a nested list with outer 
               length equal to `num_frames` and inner length 2, corresponding to the number of 
               audio representations.

            Returns: 
                : BaseAudio, list(BaseAudio), or list(list(BaseAudio))
                    Next segment or next batch of segments
        """
        if self.return_as_batch:
            self.load_next_batch()
            return self.batch
        else:           
            return self.next_in_batch()

    def next_in_batch(self):
        """ Load the next audio segment.
        
            Returns: 
                a: BaseAudio or list(BaseAudio)
                    Next audio segment
        """
        if self.counter == 0 or self.counter >= len(self.batch): 
            self.load_next_batch()
        
        a = self.batch[self.counter]
        self.counter += 1
        return a

    def load_next_batch(self):
        """ Load the next batch of audio objects.
        """
        self.batch_size = 0
        self.counter = 0
        offset = np.inf
        data_dir = self.data_dir
        filename = self.filename
        while data_dir == self.data_dir and filename == self.filename and offset > self.offset and self.batch_size < self.max_batch_size:
            self.batch_size += 1
            audio_sel = next(self.selection_gen)
            offset = audio_sel['offset']
            data_dir = audio_sel['data_dir']
            filename = audio_sel['filename']            

        duration = self.selection_gen.duration + self.selection_gen.step * (self.batch_size - 1)

        # load the data without applying transforms
        self.batch = self.load(data_dir=self.data_dir, filename=self.filename, offset=self.offset, 
            duration=duration, label=None, apply_transforms=False, **self.kwargs)

        if not isinstance(self.batch, list): self.batch = [self.batch]

        # loop over the representations
        for i in range(len(self.transforms_list)):

            # segment the data
            self.batch[i] = self.batch[i].segment(window=self.selection_gen.duration, step=self.selection_gen.step)

            # apply the transforms to each segment separately 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")        
                for j in range(len(self.batch[i])):
                    self.batch[i][j].apply_transforms(self.transforms_list[i])

        if len(self.batch) == 1: self.batch = self.batch[0]

        self.offset = offset
        self.data_dir = data_dir
        self.filename = filename


class AudioSelectionLoader(AudioLoader):
    """ Load segments of data from audio files. 

        The segments to be loaded are specified via a selection table.

        Note: It is possible to enforce all segments to have the same length by 
        specifyng a `duration` parameter in the audio representation dictionary.
        Selections that are shorter than the specified duration will be extended
        at the end, and selections that are longer will be cropped.
        For example, specifying

            >>> repres = {'type': 'Waveform', 'duration': '3.0s'}

        would result in all loaded segments having a length of precisely 3 seconds, 
        even if some of the selections deviate from this length. For example, the 
        selection [9, 14] would be shortened to [9, 12] and the selection [0.2, 2.0]
        would be extended to [0.2, 3.2].

        TODO: Change default value of `stop` argument to True.

        Args:
            selections: pandas DataFrame
                Selection table.
            path: str
                Path to folder containing the audio files.
            filename: str or list(str)
                Relative path to a single audio file or a list of audio files. Optional.
            annotations: pandas DataFrame
                Annotation table. Optional.
            repres: dict or list(dict)
                Audio data representation. Must contain the key 'type' as well as any arguments 
                required to initialize the class using the `from_wav` method.  
                It is also possible to specify multiple audio presentations as a list.
                The default representation is the raw, unaltered waveform.
            include_attrs: bool
                If True, load data from all attribute columns in the selection table. Default is False.
            attrs: list(str)
                Specify the names of the attribute columns that you wish to load data from. 
                Overwrites include_attrs if specified. If None, all columns will be loaded if 
                `include_attrs=True`.
            batch_size: int
                Load segments in batches rather than one at the time. 
            stop: bool
                Raise StopIteration if the iteration exceeds the number of available selections. Default is False.
    """
    def __init__(self, path, selections, channel=0, annotations=None, repres={'type': 'Waveform'}, 
        include_attrs=False, attrs=None, batch_size=1, stop=False, **kwargs):

        duration = _get_duration(repres)

        super().__init__(selection_gen=SelectionTableIterator(data_dir=path, 
            selection_table=selections, duration=duration, include_attrs=include_attrs, 
            attrs=attrs), channel=channel, annotations=annotations, repres=repres, 
            batch_size=batch_size, stop=stop, **kwargs)

    def get_selection(self, id):
        """ Returns the audio selection with a given id.

            Args:
                id: int
                    The id within the selection table to be searched        
            Returns:
                audio_sel: dict
                    Audio selection
        """
        return self.selection_gen.get_selection(id)


def _get_duration(repres, duration=None):
    """ Helper function.
    
        Extracts the duration parameter from the audio presentation 
        dictionary, if available.

        If the expected duration is specified via the `duration` argument, 
        the function checks if it is consistent with the value extracted 
        from the dictionary. If this is not the case, a warning is issued 
        and the return value is the expected value.

        If the audio representation does not contain a duration parameter
        and the expected duration is not specified, the return value is None.

        Args:
            repres: dict or list(dict)
                One or several audio representations.
            duration: float
                Duration in seconds. Optional

        Returns:
            duration: float
                Duration in seconds
    """
    r0 = repres[0] if isinstance(repres, list) else repres

    if duration is not None:
        if 'duration' in r0.keys() and r0['duration'] is not None and r0['duration'] != duration:
            print("Warning: Mismatch between duration argument ({0:.3f} s) and duration parameter in \
                    audio representation ({1:.3f} s). The latter value will be ignored.")

    else:
        duration = r0['duration'] if 'duration' in r0.keys() else None

    return duration