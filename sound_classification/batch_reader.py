
from sound_classification.data_handling import get_wave_files


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
        self.reset()
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

    def next(int max_size=None):
        # loop over files, read and merge, stop when size > max_size
        # return copy of merged signal and update file counter
        # clear local copy of merged signal
        # return None, if there are no more files to read
        
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
        self.index = 0
        self.times.clear()

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
def parse_datetime(fname, fmt):
    return 0
