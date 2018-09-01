
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
            overlap: int
                Size of overlap region (number of samples) used for smoothly joining audio signals 
    """
    def __init__(self, source, rate, datetime_fmt=None, overlap=100):
        self.rate = rate
        self.overlap = overlap
        self.index = 0
        self.times = list()
        self.files = list()
        self.signal = None
        load(source=source, datetime_fmt=datetime_fmt)

    def load(source, datetime_fmt=None):
        """
            Reset the reader and load new data.
            
            Args:
                source: str or list
                    File name, list of file names, or directory name 
                datetime_fmt: str
                    Format for parsing date-time data from file names
        """
        files.clear()

        # get list of file names
        fnames = list()
        if isinstance(source, list):
            fnames = source
        else:
            if source[-3:] == '.wav':
                fnames = [source]
            else:
                fnames = get_wave_files(source)
        
        # check that files exist
        for f in fnames:
            assert os.path.exists(path), " Could not find {0}".format(f)

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

    def read_file(i):
    
        assert i < len(files):
            
        f = self.files[i]
        s = TimeStampedAudioSignal.from_wav(path=f[0], time_stamp=f[1]) # read in audio data from wav file
        s.resample(new_rate=self.rate) # resample

        return s

    def next(max_size=None):

        if self.index == len(files) - 1:
            return None

        batch = self.signal
        self.times.append(self.time)

        # loop over files
        n = len(files)
        while self.index < n:

            signal = read_file(self.index) # read audio file
            self.index += 1
            
            delay = batch.delay(signal) # compute delay

            # if delay is negative, reduce the overlap region to a managable size and make smooth transition
            if delay < 0:
                delay = -self.overlap / signal.rate

            batch_size = batch.merged_length(signal=signal, delay=delay) # compute the size of the merged audio signal

            t = batch.end() + datetime.timedelta(microseconds=1E6*delay) # start time of appended signal / new batch

            if batch_size > max_size:
                self.signal = signal
                self.time = t
                return batch
            else:
                batch.append(signal=signal, delay=delay)
                times.append(t)

        return batch
        
                
    def finished():
        """
            Reader has read all load data.
            
            Returns: 
                res: bool
                True if all data has been process, False otherwise
        """
        res = self.index == len(files)
        return res
    
    def reset()
        """
            Go back and start reading from the beginning of the first file.
            
        """
        # reset 
        self.index = 0
        self.times.clear()

        # read the first file 
        self.signal = read_file(0)
        self.time = self.signal.begin()
        self.index += 1

    def log():
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

