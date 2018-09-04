
import os
import math
from sound_classification.data_handling import get_wave_files
from sound_classification.audio_signal import TimeStampedAudioSignal


class BatchReader:
    """ Audio file batch reader.

        Reads audio file(s) and serves them in batches.

        Args:
            source: str or list
                File name, list of file names, or directory name 
            rate: float
                Sampling rate in Hz
            datetime_fmt: str
                Format for parsing date-time data from file names
            n_smooth: int
                Size of region (number of samples) used for smoothly joining audio signals 
    """
    def __init__(self, source, rate=None, datetime_fmt=None, n_smooth=100):
        self.rate = rate
        self.n_smooth = n_smooth
        self.times = list()
        self.files = list()
        self.batch = None
        self.signal = None
        self.index = -1
        self.time = None
        self.eof = False
        self.load(source=source, datetime_fmt=datetime_fmt)

    def load(self, source, datetime_fmt=None):
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
                fnames = get_wave_files(source)
        
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
            t = None
            if datetime_fmt is not None:
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
            
        f = self.files[i]
        s = TimeStampedAudioSignal.from_wav(path=f[0], time_stamp=f[1]) # read in audio data from wav file
        
        if self.rate is not None:
            s.resample(new_rate=self.rate) # resample

        return s
        
    def _read_next_file(self):
        """
            Read next file, increment file counter, and update time.
        """
        self.index += 1 # increment counter
        if self.index < len(self.files):
            self.signal = self.read_file(self.index) # read audio file
            self.time = self.signal.begin() # start time
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
            t = self.batch.append(signal=self.signal, n_smooth=self.n_smooth, max_length=size) # add to existing batch
            if file_is_new: 
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
        

# this function should be placed in data_handling module        
import datetime
def parse_datetime(fname, fmt):

    date, time = list(), list()

    # time
    p = fname.rfind("HMS")    
    if p >= 0:
        for n in range(3):
            p = p + 1 + fname[p:].find("_")
            time.append(int(fname[p:p+2])) # hour, min, sec

    # date
    p = fname.rfind("DMY")
    if p >= 0:
        for n in range(3):
            p = p + 1 + fname[p:].find("_")
            date.append(int(fname[p:p+2])) # day, month, year

    # create datetime object
    dt = default_time_stamp
    if len(date) == 3 and len(time) == 3:
        dt = datetime.datetime(year=2000+date[2], month=date[1], day=date[0], hour=time[0], minute=time[1], second=time[2])

    return dt

