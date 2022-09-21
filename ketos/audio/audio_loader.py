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

    The audio representations currently implemented in Ketos are: 

    - :class:`Waveform <ketos.audio.waveform.Waveform>`
    - :class:`magnitude spectrogram <ketos.audio.spectrogram.MagSpectrogram>`
    - :class:`power spectrogram <ketos.audio.spectrogram.PowSpectrogram>`
    - :class:`mel spectrogram <ketos.audio.spectrogram.MelSpectrogram>`
    - :class:`CQT spectrogram <ketos.audio.spectrogram.CQTSpectrogram>`
    - :class:`CQTSpectrogram <ketos.audio.spectrogram.CQTSpectrogram>`
    - :class:`GammatoneFilterBank <ketos.audio.gammatone.GammatoneFilterBank>`
    - :class:`AuralFeatures <ketos.audio.gammatone.AuralFeatures>`
"""
import os
import pandas as pd
import numpy as np
import soundfile as sf
import tarfile
import warnings
import shutil
from ketos.audio.waveform import Waveform, get_duration
from ketos.data_handling.data_handling import find_wave_files
from ketos.data_handling.selection_table import query
from ketos.utils import floor_round_up, ceil_round_down, user_format_warning


class ArchiveManager():
    ''' Class for extracting files from a .tar file.

        Use the method :meth:`ketos.audio.audio_loader.ArchiveManager.extract` to extract one or 
        several files from the .tar file to a temporary directory.

        Every time a file extraction request is submitted, the contents of the temporary directory 
        are updated as follows: 

         * Requested files *not* already present in the directory are extracted.
         * Requested files already present in the directory are left untouched.
         * Files present in the directory that are not part of the request are removed.

        At any given time, the location of the temporary directory and the paths of the files stored 
        within the directory can be accessed via the attributes @extract_dir and @extracted_files.

        Args:
            tar_path: str
                Path to the .tar file
            extract_dir: str
                Path to the directory where the extracted files are temporarily stored. The directory 
                is automatically created. If a directory already exists at the specified path, all its 
                contents will be deleted. By default, audio files are extracted to the folder `kt-tmp` 
                within the current working directory.

        Attributes:
            tar: TarFile
                tar object
            tar_path: str
                Path to the .tar file
            extract_dir: str
                Path to the directory where the extracted files are temporarily stored
            extracted_files: list
                Relative paths to the currently extracted files
    '''
    def __init__(self, tar_path, extract_dir="kt-tmp"):
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path)
        self.extract_dir = extract_dir
        self.extracted_files = []
        self.close() 

    def _extract_files(self, paths):
        """ Helper function for extracting files.

            Issues a UserWarning if a file does not exist at the specified 
            path within the tar archive.

            Args:
                paths: list
                    Relative paths to the files to be extracted from within the tar archive
        """ 
        for path in paths:
            try:
                self.tar.extract(member=path, path=self.extract_dir)
                self.extracted_files.append(path)
            except KeyError as e:
                warnings.formatwarning = user_format_warning
                warnings.warn(f"{path} not found in {self.tar_path}")

    def _remove_files(self, paths):
        """ Helper function for removing files from the extraction directory.

            Args:
                paths: list
                    Relative paths to the files to be removed
        """ 
        for path in paths:
            dst = os.path.join(self.extract_dir, path)
            os.remove(dst)
            self.extracted_files.remove(path)

    def extract(self, paths):
        """ Update the files in the extraction directory.

            Every time this method is called, the contents of the temporary directory 
            are updated as follows: 

                * Requested files *not* already present in the directory are extracted.
                * Requested files already present in the directory are left untouched.
                * Files present in the directory that are not part of the request are removed.

            Args:
                paths: str or list
                    Relative path(s) of the files within the tar archive that we want to 
                    be available in the extraction directory
        """
        if isinstance(paths, str):
            paths = [paths]

        paths_extract = [path for path in paths if path not in self.extracted_files]
        paths_remove = [path for path in self.extracted_files if path not in paths]
        self._extract_files(paths_extract)
        self._remove_files(paths_remove)

    def close(self):
        """ Remove the extraction directory and its contents 
        """
        if os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir)

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
                Path to top folder containing audio files, or a .tar archive file.
            selection_table: pandas DataFrame
                Selection table
            include_attrs: bool
                If True, load data from all attribute columns in the selection table. Default is False.
            attrs: list(str)
                Specify the names of the attribute columns that you wish to load data from. 
                Overwrites include_attrs if specified. If None, all columns will be loaded provided that 
                include_attrs=True.
            extract_dir: str
                Temporary directory for storing audio files extracted from a tar archive file. 
                Only relevant if @data_dir points to a .tar file. The directory will be automatically 
                created. If a directory already exists at the specified path, all its contents will be 
                deleted. By default, audio files are extracted to the folder `kt-tmp` within the current
                working directory. Note that this folder must be deleted manually when it is no longer needed.
    """
    def __init__(self, data_dir, selection_table, include_attrs=False, attrs=None, extract_dir="kt-tmp"):
        self.sel = selection_table

        if os.path.isfile(data_dir) and tarfile.is_tarfile(data_dir):
            self.tar = ArchiveManager(data_dir, extract_dir)
            self.dir = self.tar.extract_dir
        else:
            self.tar = None
            self.dir = data_dir

        self.counter = 0

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

        # determine if the selection table has been formatted according to 
        # the new ketos style (>=2.6.0) or the old style 
        self._new_style = (self.sel.index.names[0] == "sel_id")

        if self._new_style:
            self.sel_ids = self.sel.index.get_level_values(0).unique()
            self.num_sel = len(self.sel_ids)
        else:
            self.num_sel = len(self.sel)

    def __next__(self):
        """ Returns the next audio selection.

            Returns:
                audio_sel: dict
                    Audio selection
        """
        audio_sel = self.get_selection(self.counter)

        if self.tar is not None:
            self.tar.extract(audio_sel['filename'])

        self.counter = (self.counter + 1) % self.num() #update selection counter
        return audio_sel

    def num(self):
        """ Returns total number of selections.
        
            Returns:
                : int
                    Total number of selections.
        """
        return self.num_sel

    def reset(self):
        """ Resets the selection generator to the beginning of the selection table.
        """        
        self.counter = 0
        if self.tar is not None:
            self.tar.close()
        
    def get_selection(self, n):
        """ Returns the n-th audio selection in the table.

            Args:
                n: int
                    The index (0,1,2,...) of the desired selection.

            Returns:
                res: dict
                    The selection
        """
        res = {'data_dir': self.dir}

        if self._new_style:
            selection = self.sel.loc[self.sel_ids[n]]
            res['filename'] = selection.index.values            
        
        else:
            selection = self.sel.iloc[n]
            res['filename'] = self.sel.index.values[n][0]
        
        # start time
        if 'start' in selection.keys(): 
            offset = selection['start']
        else: 
            offset = 0

        # duration
        if 'end' in selection.keys(): 
            duration = selection['end'] - offset
        else:
            duration = None

        # pass offset and duration to return dict
        res['offset'] = offset
        if duration is not None:
            res['duration'] = duration

        # label
        if 'label' in self.sel.columns.values: 
            res['label'] = selection['label']

        # attribute columns
        for col in self.attrs: 
            res[col] = selection[col]

        # for new style, convert pandas Series to numpy arrays
        if self._new_style:
            for key in res.keys():
                if isinstance(res[key], pd.Series):
                    res[key] = res[key].values
                    if key in ["offset", "duration"]:
                        res[key] = res[key].astype(float) #ensure float

        # OBS: for labels and attributes, use only the first entry
        for col in ["label"] + self.attrs:
            if col in res.keys() and np.ndim(res[col]) > 0:
                res[col] = res[col][0]
        
        return res


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
            pad: bool
                If True (default), the last segment is allowed to extend beyond the endpoint of the audio file.
            frame: float
                Same as duration. Only included for backward compatibility. Will be removed in future versions.
    """
    def __init__(self, duration=None, step=None, path=None, filename=None, pad=True, frame=None):
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
        self.file_durations = np.array(get_duration([os.path.join(self.dir, f) for f in self.files]))

        # discard any files with 0 second duration
        self.files = np.array(self.files)[self.file_durations > 0].tolist()
        self.file_durations = self.file_durations[self.file_durations > 0]

        # obtain file durations and compute number of frames for each file
        self.num_segs = np.maximum((self.file_durations - self.duration) / self.step + 1, 1)
        if pad:
            self.num_segs = ceil_round_down(self.num_segs, decimals=6).astype(int)        
        else:
            self.num_segs = floor_round_up(self.num_segs, decimals=6).astype(int)               

        self.num_segs_tot = np.sum(self.num_segs)

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
        return self.file_durations.tolist()


def _file_limits_warning(start, end, file_path, file_duration):
    """ Helper function for the AudioLoader class.
    
        Generates warning messages if the selection start or end time is outside of the audio file's limits.

        Args:
            start: float
                Selection start time with respect to beginning of file in seconds.
            end: float
                Selection end time with respect to beginning of file in seconds.
            file_path: str
                Full path of the audio file.
            file_duration: float
                Audio file length in seconds.
    """
    # total length of selection
    len_tot = end - start
    # determine how much of the selection is outside the file
    len_outside = max(0, -start) + max(0, end - file_duration)

    warnings.formatwarning = user_format_warning

    # print warnings if selection end is zero or negative
    file_info = f"While processing {os.path.basename(file_path)}"
    if (end <= 0):
        warnings.warn(f"{file_info}: selection has negative end time ({end:.2f}s).")

    # print warnings if selection start is later than the file end time
    elif (start > file_duration):
        warnings.warn(f"{file_info}: selection start time exceeds file duration ({start:.2f}s).")

    #print a warning that the selection has 0 or negative length (end before start)
    elif (len_tot <= 0):
        warnings.warn(f"{file_info}: selection has negative duration ({start:.2f},{end:.2f}).")
         
    # print a warning that a fraction larger than 50% of the selection is outside the file
    elif (len_outside > 0.5 * len_tot):
        warnings.warn(f"{file_info}: over 50% of the selection falls outside the audio file ({start:.2f}s,{end:.2f}s).")


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
            representation: class or list of classes
                Audio data representation. This is a class that must receive the raw audio data and will transform the data 
                into the specified audio representation object.
                
                Classes available in ketos:

                    * Waveform: 
                        (rate), (resample_method)
                    
                    * MagSpectrogram, PowerSpectrogram, MelSpectrogram: 
                        audio, window, step, (window_func), (rate), (resample_method)
                    
                    * CQTSpectrogram:
                        audio, step, bins_per_oct, (freq_min), (freq_max), (window_func), (rate), (resample_method)

                It is also possible to specify multiple audio presentations as a list.
            representation_params: dict or list of dict
                Dictionary containing any required and optional arguments for the representation class. If more than one
                representation is given `representation_params` must be a list of the same length and in the same order.
            batch_size: int
                Load segments in batches rather than one at the time. 
            stop: bool
                Raise StopIteration when all selections have been loaded. Default is True.
        
        Examples:

            Creating an AudioLoader to load selections:         
            
            >>> from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator
            >>> from ketos.data_handling.selection_table import use_multi_indexing
            >>> import pandas as pd
            >>> # Load the audio representation you want to pass
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> # specify the audio representation
            >>> rep = {'window':0.2, 'step':0.02, 'window_func':'hamming'}
            >>> # Load selections
            >>> sel = pd.DataFrame({'filename':["2min.wav", "2min.wav"],'start':[0.10,0.12],'end':[0.46,0.42]})
            >>> sel = use_multi_indexing(sel, 'sel_id')
            >>> # create a generator for iterating over all the selections 
            >>> generator = SelectionTableIterator(data_dir="ketos/tests/assets/", selection_table=sel)
            >>> # Create a loader by passing the generator and the representation to the AudioLoader
            >>> loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep)
            >>> # print number of segments
            >>> print(loader.num())
            2
            >>> # load and plot the first selection
            >>> spec = next(loader)
            >>>
            >>> import matplotlib.pyplot as plt
            >>> fig = spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec_loader_2min_0.png")
            >>> plt.close(fig)
            
            .. image:: ../../../ketos/tests/assets/tmp/spec_loader_2min_0.png

            Creating an AudioLoader to load selections made from annotations:   

            >>> from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator
            >>> from ketos.data_handling.selection_table import standardize
            >>> import pandas as pd
            >>> # Load the audio representation you want to pass
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> # specify the audio representation
            >>> rep = {'window':0.2, 'step':0.02, 'window_func':'hamming'}
            >>> # Load selections
            >>> annot = pd.DataFrame([{"filename":"2min.wav", "start":2.0, "end":3.0, "label":0},
            ...         {"filename":"2min.wav", "start":5.0, "end":6.0, "label":0},
            ...         {"filename":"2min.wav", "start":21.0, "end":22.0, "label":0},
            ...         {"filename":"2min.wav", "start":25.0, "end":27.0, "label":0}])
            >>> annot_std = standardize(table=annot)
            >>> # create a generator for iterating over all the selections 
            >>> generator = SelectionTableIterator(data_dir="ketos/tests/assets/", selection_table=annot_std)
            >>> # Create a loader by passing the generator and the representation to the AudioLoader
            >>> loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep)
            >>> # print number of segments
            >>> print(loader.num())
            4
            >>> # load and plot the first selection
            >>> spec = next(loader)
            >>>
            >>> import matplotlib.pyplot as plt
            >>> fig = spec.plot()
            >>> fig.savefig("ketos/tests/assets/tmp/spec_loader_2min_1.png")
            >>> plt.close(fig)
            
            .. image:: ../../../ketos/tests/assets/tmp/spec_loader_2min_1.png

            For more examples see child class :class:`audio.audio_loader.AudioFrameLoader`   

    """
    def __init__(self, selection_gen, channel=0, annotations=None, representation=Waveform, representation_params=None, 
                        batch_size=1, stop=True, **kwargs):

        self.representation = representation
        self.representation_params = representation_params

        if not isinstance(self.representation, list):
            self.representation = [self.representation]
            self.representation_params = [self.representation_params]
        
        for i in range(len(self.representation)):
            if self.representation_params[i] == None: # If no parameters are given then create an empty dict (this will use the default params)
                self.representation_params[i] = {}
        self.channel = channel
        self.selection_gen = selection_gen
        self.annot = annotations
        # QUESTION: kwargs is carrying more optional arguments. such as compute phase... it feels very wrong. shouldnt the phase be another representation? or an arugment of the spectrogram class?
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.stop = stop
        self.file_durations = dict()
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

             * If the loader was initialized with the audio representation `representation=Waveform`,   
               `representation_params=None` (default) and with `batch_size=1` (default), the return  
               value will be a single instance of :class:`Waveform <ketos.audio.waveform.Waveform>`.

             * If the loader was initialized with the audio representation 
               `representation=[Waveform, MagSpectrogram]`, `representation_params=[None, {'window':0.1,'step':0.02}]` 
               and with `batch_size=1` (default), the return value will be a list 
               of length 2, where the first entry holds an instance of 
               :class:`Waveform <ketos.audio.waveform.Waveform>` and the second entry holds an instance 
               of :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`.

             * If the loader was initialized with the audio representation 
               `representation=[Waveform, MagSpectrogram]`, `representation_params=[None, {'window':0.1,'step':0.02}]` 
               and with `batch_size>1`, the return value will be a nested list with outer 
               length equal to `batch_size` and inner length 2, corresponding to the number of 
               audio representations.

            If the loader was initialized with `stop=True` this method will raise `StopIteration` 
            when all the selections have been loaded.

            Returns: 
                a: BaseAudio, list(BaseAudio), or list(list(BaseAudio))
                    Next segment or next batch of segments
        """
        try:
            return self._next_batch(load=True)
        except Exception as e:
            warnings.formatwarning = user_format_warning
            warnings.warn(f"While loading entry no. {self.counter}: {str(e)}", category=UserWarning)
            
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
                seg: BaseAudio or list(BaseAudio)
                    Audio segment
        """
        # convert scalar args to arrays
        if np.ndim(filename) == 0:
            filename = [filename]
            offset = np.array([offset], dtype=float)
            if duration is None:
                duration = [None]
            else:
                duration = np.array([duration], dtype=float)


        path = [os.path.join(data_dir, fname) for fname in filename]
        id = filename[0]

        # issue warnings if selections extend beyond file limits
        for i in range(len(path)):  
            p = path[i]          
            file_duration = self.file_durations.get(p)
            # If file duration does not exist in the dict, get file duration and add it to the dict
            if file_duration == None:
                file_duration = get_duration(p)[0]
                self.file_durations[p] = file_duration

            start = offset[i]
            end = file_duration - start if duration[i] is None else start + duration[i]
            _file_limits_warning(start=start, end=end, file_path=p, file_duration=file_duration)


        # load audio
        segs = []
        for i in range(len(self.representation)): # For each representation
            self.representation_params[i]['duration'] = duration # The duration for the representation is defined by each segment
            seg = self.representation[i].from_wav(path=path, channel=self.channel, offset=offset, id=id, **self.representation_params[i], **kwargs) 

            # add label
            if label is not None:
                seg.label = label

            # add annotations
            if self.annot is not None:
                file_offset = np.concatenate([[0],np.cumsum(duration)[:-1]])
                for j in range(len(filename)):
                    q = query(self.annot, filename=filename[j], start=offset[j], end=offset[j]+duration[j])
                    if len(q) > 0:
                        q['start'] += file_offset[j] - offset[j]
                        q['end']   += file_offset[j] - offset[j]
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

        Args:
            duration: float
                Segment duration in seconds. 
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
            representation: class or list of classes
                Audio data representation. This is a class that must receive the raw audio data 
                and will transform the data into the specified audio representation object.  
                It is also possible to specify multiple audio presentations as a list. These 
                presentations must have the same duration.
            representation_params: dict or list of dict
                Dictionary containing any required and optional arguments for the representation class. If more than one
                representation is given `representation_params` must be a list of the same length and in the same order.
            batch_size: int
                Load segments in batches rather than one at the time. 
            stop: bool
                Raise StopIteration if the iteration exceeds the number of available selections. Default is False.
            pad: bool
                If True (default), the last segment is allowed to extend beyond the endpoint of the audio file.

        Examples:
            >>> from ketos.audio.audio_loader import AudioFrameLoader
            >>> # Load the audio representation you want to pass
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> # specify path to wav file
            >>> filename = 'ketos/tests/assets/2min.wav'
            >>> # check the duration of the audio file
            >>> from ketos.audio.waveform import get_duration
            >>> print(get_duration(filename)[0])
            120.832
            >>> # specify the audio representation parameters
            >>> rep = {'window':0.2, 'step':0.02, 'window_func':'hamming', 'freq_max':1000.}
            >>> # create an object for loading 30-s long spectrogram segments, using a step size of 15 s (50% overlap) 
            >>> loader = AudioFrameLoader(duration=30., step=15., filename=filename, representation=MagSpectrogram, representation_params=rep)
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
    def __init__(self, duration, step=None, path=None, filename=None, channel=0, 
                    annotations=None, representation=Waveform, representation_params=None, batch_size=1, 
                    stop=True, pad=True):

        if batch_size > 1:
            print("Warning: batch_size > 1 results in different behaviour for ketos versions >= 2.4.2 than earlier \
                   versions. You may want to check out the AudioFrameEfficientLoader class.")

        super().__init__(selection_gen=FrameStepper(duration=duration, step=step, path=path, pad=pad, filename=filename), 
            channel=channel, annotations=annotations, representation=representation, representation_params=representation_params, 
            batch_size=batch_size, stop=stop)

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
            representation: class or list of classes
                Audio data representation. This is a class that must receive the raw audio data 
                and will transform the data into the specified audio representation object.  
                It is also possible to specify multiple audio presentations as a list. These 
                presentations must have the same duration.
            representation_params: dict or list of dict
                Dictionary containing any required and optional arguments for the representation class. If more than one
                representation is given `representation_params` must be a list of the same length and in the same order.
            num_frames: int
                Load segments in batches of size `num_frames` rather than one at the time. 
                Increasing `num_frames` can help reduce computational time.
                You can also specify `num_frames='file'` to load one wav file at the time.
            return_as_batch: bool
                Whether to return the segments individually or in batches of size `num_frames`.
                The default behaviour is to return the segments individually.
    """
    def __init__(self, duration=None, step=None, path=None, filename=None, channel=0, 
                    annotations=None, representation=Waveform, representation_params=None,
                    num_frames=12, return_as_batch=False):

        assert (isinstance(num_frames, int) and num_frames >= 1) or \
            (isinstance(num_frames, str) and num_frames.lower() == 'file'), \
            'Argument `num_frames` must be a positive integer or have the string value `file`'

        super().__init__(duration=duration, step=step, path=path, filename=filename, 
                    channel=channel, annotations=annotations, representation=representation, 
                    representation_params=representation_params)

        self.return_as_batch = return_as_batch
        self.transforms_list = []
        
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

             * If the loader was initialized with the audio representation `representation=Waveform`,   
               `representation_params=None` (default) and with `return_as_batch=False` (default), 
               the return value will be a single instance of :class:`Waveform <ketos.audio.waveform.Waveform>`.

             * If the loader was initialized with the audio representation 
               `representation=[Waveform, MagSpectrogram]`, `representation_params=[None, {'window':0.1,'step':0.02}]` 
               and with `return_as_batch=False` (default), the return value will be a list 
               of length 2, where the first entry holds an instance of 
               :class:`Waveform <ketos.audio.waveform.Waveform>` and the second entry holds an instance 
               of :class:`MagSpectrogram <ketos.audio.spectrogram.MagSpectrogram>`.

             * If the loader was initialized with the audio representation 
               `representation=[Waveform, MagSpectrogram]`, `representation_params=[None, {'window':0.1,'step':0.02}]` 
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
            duration=duration, label=None)

        if not isinstance(self.batch, list): self.batch = [self.batch]

        # loop over the representations
        for i in range(len(self.representation)):
            transforms = self.representation_params[i]['transforms'] if 'transforms' in self.representation_params[i].keys() else []
            self.transforms_list.append(transforms)
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