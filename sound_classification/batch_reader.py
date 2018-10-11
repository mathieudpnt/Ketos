
import os
import math
import pandas as pd
import datetime
from sound_classification.data_handling import get_wave_files, parse_datetime
from sound_classification.audio_signal import TimeStampedAudioSignal


class BatchReader:
    """ Reads audio file(s) and serves them in batches of specified size.

        If the file names do not have date-time information, the files will 
        be sorted in alphabetical order and smoothly joined to one another.

        Otherwise, the date-time information will be extracted from the file 
        names and used to sort the files chronologically. Any gaps will be 
        filled with zeros.

        Args:
            source: str or list
                File name, list of file names, or directory name 
            recursive_search: bool
                Include files from all subdirectories 
            rate: float
                Sampling rate in Hz
            datetime_fmt: str
                Format for parsing date-time data from file names
            n_smooth: int
                Size of region (number of samples) used for smoothly joining audio signals 
            verbose: bool
                Print progress messages during processing 
    """
    def __init__(self, source, recursive_search=False, rate=None, datetime_fmt=None, n_smooth=100, verbose=False):
        self.rate = rate
        self.n_smooth = n_smooth
        self.times = list()
        self.files = list()
        self.batch = None
        self.signal = None
        self.index = -1
        self.time = None
        self.eof = False
        self.verbose = verbose
        self.load(source=source, recursive_search=recursive_search, datetime_fmt=datetime_fmt)

    def load(self, source, recursive_search=False, datetime_fmt=None):
        """
            Reset the reader and load new data.
            
            Args:
                source: str or list
                    File name, list of file names, or directory name 
                datetime_fmt: str
                    Format for parsing date-time data from file names
        """
        self.files.clear()

        # get list of file names
        fnames = list()
        if isinstance(source, list):
            fnames = source
        else:
            if source[-4:] == '.wav':
                fnames = [source]
            else:
                fnames = get_wave_files(path=source, subdirs=recursive_search)
        
        # check that files exist
        for f in fnames:
            assert os.path.exists(f), " Could not find {0}".format(f)

        # check that we have at least 1 file
        assert len(fnames) > 0, " No wave files found in {0}".format(source)

        # default time stamp
        t0 = datetime.datetime.today()
        t0 = datetime.datetime.combine(t0, datetime.datetime.min.time())

        # time stamps
        for f in fnames:
            t = parse_datetime(f, datetime_fmt)
            if t is None:
                t = t0
            self.files.append([f, t])

        # sort signals in chronological order
        def sorting(y):
            return y[1]
        self.files.sort(key=sorting)

        # reset the reader
        self.reset()

    def read_file(self, i):
    
        assert i < len(self.files), "attempt to read file with id {0} but only {1} files have been loaded".format(i, len(self.files))

        if self.verbose:
            print(' File {0} of {1}'.format(i+1, len(self.files)), end="\r")
            
        f = self.files[i]
        s = TimeStampedAudioSignal.from_wav(path=f[0], time_stamp=f[1]) # read in audio data from wav file
        
        if self.rate is not None:
            s.resample(new_rate=self.rate) # resamples

        return s
        
    def _read_next_file(self):
        """
            Read next file, increment file counter, and update time.
        """
        self.index += 1 # increment counter
        if self.index < len(self.files):
            self.signal = self.read_file(self.index) # read audio file
            self.time = self.signal.begin() # start time of this audio file
        else:
            self.signal = None
            self.time = None
            self.eof = True

    def _add_to_batch(self, size, new_batch):
        """
            Add audio from current file to batch.
            
            Args:
                size: int
                    Maximum batch size (number of samples) 
                new_batch: bool
                    Start a new batch or add to existing
        """
        if self.signal.empty():
            self._read_next_file()

        file_is_new = self.signal.begin() == self.time # check if we have already read from this file

        if new_batch:
            if self.batch is not None:
                t_prev = self.batch.end() # end time of the previous batch
            else:
                t_prev = None

            self.batch = self.signal.clip(s=size) # create a new batch

            if t_prev is not None and self.batch.begin() < t_prev: 
                self.batch.time_stamp = t_prev # ensure that new batch starts after the end of the previous batch
                
            if file_is_new: 
                self.times.append(self.batch.begin()) # collect times
        else:
            l = len(self.signal.data)
            if file_is_new:
                n_smooth = self.n_smooth
            else:
                n_smooth = 0
            t = self.batch.append(signal=self.signal, n_smooth=n_smooth, max_length=size) # add to existing batch
            if file_is_new and (self.signal.empty() or len(self.signal.data) < l): 
                self.times.append(t) # collect times
        
        if self.signal.empty() and self.index == len(self.files) - 1: # check if there is more data
            self.eof = True 

    def next(self, size=math.inf):
        """
            Read next batch of audio files and merge into a single audio signal. 
            
            If no maximum size is given, all loaded files will be read and merged.
            
            Args:
                size: int
                    Maximum batch size (number of samples) 
                    
            Returns:
                batch: TimeStampedAudioSignal
                    Merged audio signal
        """
        if self.finished():
            return None
        
        length = 0
        
        while length < size and not self.finished():

            self._add_to_batch(size, new_batch=(length==0))
            
            length = len(self.batch.data)

        if self.finished() and self.verbose:
            print(' Successfully processed {0} files'.format(len(self.files)))

        return self.batch
        
                
    def finished(self):
        """
            Reader has read all load data.
            
            Returns: 
                x: bool
                True if all data has been process, False otherwise
        """
        return self.eof
    
    def reset(self):
        """
            Go back and start reading from the beginning of the first file.
            
        """
        # reset 
        self.index = -1
        self.times.clear()
        self.eof = False
        self.batch = None

        # read the first file 
        self._read_next_file()

    def log(self):
        """
            Generate summary of all processed data.

            Returns:
                df: pandas DataFrame
                    Table with file names and time stamps
            
        """
        fnames = [x[0] for x in self.files]
        df = pd.DataFrame(data={'time':self.times,'file':fnames})
        return df