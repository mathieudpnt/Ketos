""" Data handling module within the ketos library

    This module provides utilities to load and handle data files.

    
    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
             Project goal: The ketos library provides functionalities for handling data, processing audio signals and
             creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
import pandas as pd
import librosa
import os
import math
import errno
import tables
from subprocess import call
import scipy.io.wavfile as wave
import ketos.external.wavfile as wave_bit
from ketos.utils import tostring
import datetime
import datetime_glob
import re



def parse_datetime(to_parse, fmt=None, replace_spaces='0'):
    """Parse date-time data from string.
       
       Returns None if parsing fails.
        
        Args:
            to_parse: str
                String with date-time data to parse.
            fmt: str
                String defining the date-time format. 
                Example: %d_%m_%Y* would capture "14_3_1999.txt"
                See https://pypi.org/project/datetime-glob/ for a list of valid directives
                
            replace_spaces: str
                If string contains spaces, replaces them with this string

        Returns:
            datetime: datetime object

        Examples:
            >>> #This will parse dates in the day/month/year format,
            >>> #separated by '/'. It will also ignore any test after the year,
            >>> # (such as a file extension )
            >>> fmt = "%d/%m/%Y*"
            >>> result = parse_datetime("10/03/1942.txt", fmt)
            >>> result.year
            1942
            >>> result.month
            3
            >>> result.day
            10
            >>>
            >>> # Now with the time (hour:minute:second) separated from the date by un underscore
            >>> fmt = "%H:%M:%S_%d/%m/%Y*"
            >>> result = parse_datetime("15:43:03_10/03/1918.wav", fmt)
            >>> result.year
            1918
            >>> result.month
            3
            >>> result.day
            10
            >>> result.hour
            15
            >>> result.minute
            43
            >>> result.second
            3
    """

    # replace spaces
    to_parse = to_parse.replace(' ', replace_spaces)
    
    if fmt is not None:
        matcher = datetime_glob.Matcher(pattern=fmt)
        match = matcher.match(path=to_parse)
        if match is None:
            return None
        else:
            return match.as_datetime()

    return None

def find_files(path, substr, fullpath=True, subdirs=False):
    """ Find all files in the specified directory containing the specified substring in their file name

        Args:
            path: str
                Directory path
            substr: str
                Substring contained in file name
            fullpath: bool
                If True, return relative path to each file. If false, only return the file names 
            subdirs: bool
                If True, search all subdirectories

        Returns:
            files: list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> # Find files that contain 'super' in the name;
            >>> # Do not return the relative path
            >>> find_files(path="ketos/tests/assets", substr="super", fullpath=False)
            ['super_short_1.wav', 'super_short_2.wav']

            >>> # find all files with '.h5" in the name
            >>> # Return the relative path
            >>> find_files(path="ketos/tests/assets", substr=".h5")
            ['ketos/tests/assets/15x_same_spec.h5', 'ketos/tests/assets/cod.h5', 'ketos/tests/assets/morlet.h5']
    """
    # find all files
    allfiles = list()
    if not subdirs:
        f = os.listdir(path)
        for fil in f:
            if fullpath:
                x = path
                if path[-1] is not '/':
                    x += '/'
                allfiles.append(os.path.join(x, fil))
            else:
                allfiles.append(fil)
    else:
        for r, _, f in os.walk(path):
            for fil in f:
                if fullpath:
                    allfiles.append(os.path.join(r, fil))
                else:
                    allfiles.append(fil)

    # select those that contain specified substring
    files = list()
    for f in allfiles:
        if substr in f:
            files.append(f)

    # sort alphabetically
    files.sort()

    return files


def find_wave_files(path, fullpath=True, subdirs=False):
    """ Find all wave files in the specified directory

        Args:
            path: str
                Directory path
            fullpath: bool
                Return relative path to each file or just the file name 

        Returns:
            wavefiles: list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> find_wave_files(path="ketos/tests/assets", fullpath=False)
            ['2min.wav', 'empty.wav', 'grunt1.wav', 'super_short_1.wav', 'super_short_2.wav']

    """
    wavefiles = find_files(path, '.wav', fullpath, subdirs)
    return wavefiles


def read_wave(file, channel=0):
    """ Read a wave file in either mono or stereo mode

        Args:
            file: str
                path to the wave file
            channel: int
                Which channel should be used in case of stereo data (0: left, 1: right) 

        Returns: (rate,data)
            rate: int
                The sampling rate
            data: numpy.array (float)
                A 1d array containing the audio data
        
        Examples:
            >>> rate, data = read_wave("ketos/tests/assets/2min.wav")
            >>> type(rate)
            <class 'int'>
            >>> rate
            2000
            >>> type(data)
            <class 'numpy.ndarray'>
            >>> len(data)
            241664
    """
    try:
        rate, signal, _ = wave_bit.read(file)
    except TypeError:
        rate, signal = wave.read(file)
           
    if len(signal.shape) == 2:
        data = signal[:, channel]
    else:
        data = signal[:]
    return rate, data

def create_dir(dir):
    """ Create a new directory if it does not exist

        Will also create any intermediate directories that do not exist
        Args:
            dir: str
               The path to the new directory
     """
    os.makedirs(dir, exist_ok=True)

def to1hot(value,depth):
    """Converts the binary label to one hot format

            Args:
                value: scalar or numpy.array | int or float
                    The the label to be converted.
                depth: int
                    The number of possible values for the labels 
                    (number of categories).
                                
            Returns:
                one_hot:numpy array (dtype=float64)
                    A len(value) by depth array containg the one hot encoding
                    for the given value(s).

            Example:
                >>> values = np.array([0,1])
                >>> to1hot(values,depth=2)
                array([[1., 0.],
                       [0., 1.]])
     """
    value = np.int64(value)
    one_hot = np.eye(depth)[value]
    return one_hot

def from1hot(value):
    """Converts the one hot label to binary format

            Args:
                value: scalar or numpy.array | int or float
                    The the label to be converted.
            
            Returns:
                output: int or numpy array (dtype=int64)
                    An int representing the category if 'value' has 1 dimension or an
                    array of m ints if  input values is an n by m array.

            Example:
                >>> from1hot(np.array([0,0,0,1,0]))
                3
                >>> from1hot(np.array([[0,0,0,1,0],
                ...   [0,1,0,0,0]]))
                array([3, 1])

     """

    if value.ndim > 1:
        output = np.apply_along_axis(arr=value, axis=1, func1d=np.argmax)
        output.dtype = np.int64
    else:
        output = np.argmax(value)

    return output


def check_data_sanity(images, labels):
    """ Check that all images have same size, all labels have values, 
        and number of images and labels match.
     
        Args:
            images: numpy array or pandas series
                Images
            labels: numpy array or pandas series
                Labels
        Raises:
            ValueError:
                If no images or labels are passed;
                If the number of images and labels is different;
                If images have different shapes;
                If any labels are NaN.

       Returns:
            True if all checks pass.

        Examples:
            >>> # Load a database with images and integer labels
            >>> data = pd.read_pickle("ketos/tests/assets/pd_img_db.pickle")
            >>> images = data['image']
            >>> labels = data['label']
            >>> # When all the images and labels  pass all the quality checks,
            >>> # The function returns True            
            >>> check_data_sanity(images, labels)
            True
            >>> # If something is wrong, like if the number of labels
            >>> # is different from the number of images, and exeption is raised
            >>> labels = data['label'][:10] 
            >>> check_data_sanity(images, labels=labels)
            Traceback (most recent call last):
                File "/usr/lib/python3.6/doctest.py", line 1330, in __run
                    compileflags, 1), test.globs)
                File "<doctest data_handling.check_data_sanity[5]>", line 1, in <module>
                    check_data_sanity(images, labels=labels)
                File "ketos/data_handling/data_handling.py", line 599, in check_data_sanity
                    raise ValueError("Image and label columns have different lengths")
            ValueError: Image and label columns have different lengths



    """
    checks = True
    if images is None or labels is None:
        raise ValueError(" Images and labels cannot be None")
        

    # check that number of images matches numbers of labels
    if len(images) != len(labels):
        raise ValueError("Image and label columns have different lengths")

    # determine image size and check that all images have same size
    image_shape = images[0].shape
    if not all(x.shape == image_shape for x in images):
        raise ValueError("Images do not all have the same size")

    # check that all labels have values
    b = np.isnan(labels)    
    n = np.count_nonzero(b)
    if n != 0:
        raise ValueError("Some labels are NaN")
    
    return checks

def get_image_size(images):
    """ Get image size and check that all images have same size.
     
        Args:
            images: numpy array or pandas series
                Images

        Results:
            image_size: tuple (int,int)
                Image size

        Examples:
            >>> # Load a database with images and integer labels
            >>> data = pd.read_pickle("ketos/tests/assets/pd_img_db.pickle")
            >>> images = data['image']
            >>> get_image_size(images)
            (20, 20)

    """
    # determine image size and check that all images have same size
    image_shape = images[0].shape
    assert all(x.shape == image_shape for x in images), "Images do not all have the same size"

    return image_shape


def parse_seg_name(seg_name):
    """ Retrieves the segment id and label from the segment name

        Args:
            seg_name: str
            Name of the segment in the format id_*_*_l_*.wav, where 'id' is 
            followed by base name of the audio file from which the segment was extracted, '_',
            and a sequence number. The 'l' is followed by any number of characters describing the label(s).

        Returns:
            (id,label) : tuple (str,str)
            A tuple with the id and label strings.

    """
    id, labels = None, None
    if seg_name != '':

        splits = seg_name.split("_")
        if len(splits) >= 5:
            id = seg_name.split("_")[1] + "_" + seg_name.split("_")[2]
            tmp = seg_name.split("_")[4]
            labels = tmp.split(".")[0]

    return (id,labels)




def divide_audio_into_segs(audio_file, seg_duration, save_to, annotations=None, start_seg=None, end_seg=None):
    """ Divides a large .wav file into a sequence of smaller segments with the same duration.
        Names the resulting segments sequentially and save them as .wav files in the specified directory.

        Note: segments will be saved following the name pattern "id_*_*_l_*.wav",
            where 'id_' is followed by the name of the original file, underscore ('_') 
            and the a sequence name. 'l_' is followed by the label(s) associated with that segment.
            Ex: 'id_rec03_87_l_[1,3]', 'id_rec03_88_l_[0]

            The start_seg and end_seg arguments can be used to segment only part of audio files,
            which is usefule when processing large files in parallel.
            
        Args:
            audio_file:str
            .wav file name (including path).

            seg_duration: float
            desired duration for each segment

            annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "orig_file": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file. 
            If None, the segments will be created and file names will have 'NULL' as labels. 
            Ex: 'id_rec03_87_l[NULL].wav.
                    
            save_to: str
            path to the directory where segments will be saved.

            start_seg: int
                Indicates the number of the segment on which the segmentation will start.
                A value of 3 would indicate the 3rd segment in a sequence(if 'seg_duration' is set to 2.0,
                that would corresponfd to 6.0 seconds from the beginning of the file')
            end_seg:int
                Indicates the number of the segment where the segmentation will stop.
                A value of 6 would indicate the 3rd segment in a sequence(if 'seg_duration' is set to 2.0,
                that would correspond to 12.0 seconds from the beginning of the file'
                        
         Returns:
            None   
    """
    create_dir(save_to)
    orig_audio_duration = librosa.get_duration(filename=audio_file)
    n_seg = round(orig_audio_duration/seg_duration)

    prefix = os.path.basename(audio_file).split(".wav")[0]

    if start_seg is None:
        start_seg = 0
    if end_seg is None:
        end_seg = n_seg - 1

    for s in range(start_seg, end_seg + 1):
        start = s * seg_duration - seg_duration
        end = start + seg_duration

        if annotations is None:
            label = '[NULL]'
        else:
            label =  get_labels(prefix, start, end, annotations)

        out_name = "id_" + prefix + "_" + str(s) + "_l_" + label + ".wav"
        path_to_seg = os.path.join(save_to, out_name)    
        sig, rate = librosa.load(audio_file, sr=None, offset=start, duration=seg_duration)
        print("Creating segment......", path_to_seg)
        librosa.output.write_wav(path_to_seg, sig, rate)

def _filter_annotations_by_orig_file(annotations, orig_file_name):
    """ Filter the annotations DataFrame by the base of the original file name (without the path or extension)

        Args:
        file: str
           The original audio file name without path or extensions.
           Ex: 'file_1' will match the entry './data/sample_a/file_1.wav" in the orig_file
           column of the annotations DataFrame.

        annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "orig_file": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file.

        Returns:
            filtered annotations: pandas.DataFrame
            A subset of the annotations DataFrame containing only the entries for the specified file.
            

    """
    filtered_indices = annotations.apply(axis=1, func= lambda row: os.path.basename(row.orig_file).split(".wav")[0] == orig_file_name)
    filtered_annotations = annotations[filtered_indices]
    return filtered_annotations

def get_labels(file, start, end, annotations, not_in_annotations=0):
    """ Retrieves the labels that fall in the specified interval.
    
        Args:
        file: str
           The base name (without paths or extensions) for the original audio file. Will be used to match the 'orig_file' field
           in the annotations Dataframe. Important: The name of the files must be
           unique within the annotations, even if the path is different.
           Ex: '/data/sample_a/file_1.wav' and '/data/sample_b/file_1.wav'

        annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "orig_file": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file.
            
        not_in_annotations: str
            Label to be used if the segment is not included in the annotations.

        Returns:
            labels: str
                The labels corresponding to the interval specified.
                if the interval is not in the annotations, the value 
                specified in 'not_in_annotations' will be used.

    """
    interval_start = start
    interval_end = end

    data = _filter_annotations_by_orig_file(annotations, file)
    query_results = data.query("(@interval_start >= start & @interval_start <= end) | (@interval_end >= start & @interval_end <= end) | (@interval_start <= start & @interval_end >= end)")
    #print(query_results)
    
    if query_results.empty:
        label = [not_in_annotations]
    else:
        label=[]
        for l in query_results.label:
          label.append(l)
                 
    return str(label)


def seg_from_time_tag(audio_file, start, end, name, save_to):
    """ Extracts a segment from the audio_file according to the start and end tags.

        Args:
            audio_file:str
            .wav file name (including path).

            start:float
            Start point for segment (in seconds from start of the source audio_file)

            end:float
            End point for segment (in seconds from start of the source audio_file)
            
            save_to: str
            Path to the directory where segments will be saved.
            
            name: str
            Name of segment file name (including '.wav')
            

         Returns:

            None   

    """
    out_seg = os.path.join(save_to, name)
    sig, rate = librosa.load(audio_file, sr=None, offset=start, duration=end - start)
    librosa.output.write_wav(out_seg, sig, rate)


def segs_from_annotations(annotations, save_to):
    """ Generates segments based on the annotations DataFrame.

        Args:
            annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "orig_file": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file.
            save_to: str
            path to the directory where segments will be saved.
            
            
         Returns:
            None   

    """ 
    create_dir(save_to)
    for i, row in annotations.iterrows():
        start = row.start
        end= row.end
        base_name = os.path.basename(row.orig_file).split(".wav")[0]
        seg_name = "id_" + base_name + "_" + str(i) + "_l_[" + str(row.label) + "].wav"
        seg_from_time_tag(row.orig_file, row.start, row.end, seg_name, save_to)
        print("Creating segment......", save_to, seg_name)

def pad_signal(signal,rate, length):
    """Pad a signal with zeros so it has the specified length

        Zeros will be added before and after the signal in approximately equal quantities.
        Args:
            signal: numpy.array
            The signal to be padded

            rate: int
            The sample rate

            length: float
            The desired length for the signal

         Returns:
            padded_signal: numpy.array
            Array with the original signal padded with zeros.

        
    """
    length = length * rate
    input_length = signal.shape[0] 
    
    difference = ( length - input_length) 
    pad1_len =  int(np.ceil(difference/2))
    pad2_len = int(difference - pad1_len)

    pad1 =  np.zeros((pad1_len))
    pad2 =  np.zeros((pad2_len))

    padded_signal =  np.concatenate([pad1,signal,pad2])
    return padded_signal


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
            fmt = datetime_fmt
            p_unix = f.rfind('/')
            p_win = f.rfind('\\')
            p = max(p_unix, p_win)
            folder = f[:p+1]
            if folder is not None and fmt is not None:
                fmt = folder + fmt
            t = parse_datetime(f, fmt)
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
        from ketos.audio_processing.audio import TimeStampedAudioSignal
    
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
            
        if self.signal is None:
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
           
        if self.signal is None:
            return

        file_is_new = self.signal.begin() == self.time # check if we have already read from this file

        if new_batch:
            if self.batch is not None:
                t_prev = self.batch.end() # end time of the previous batch
            else:
                t_prev = None

            self.batch = self.signal.split(s=size) # create a new batch

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
            
            if self.batch is not None:
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
        n = len(self.times)
        fnames = [x[0] for x in self.files]
        df = pd.DataFrame(data={'time':self.times,'file':fnames[:n]})
        return df
