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

""" Data handling module within the ketos library

    This module provides utilities to load and handle data files.
"""
import numpy as np
import pandas as pd
import librosa
import os
import math
import errno
import tables
from subprocess import call
import soundfile as sf
from ketos.utils import tostring
import datetime
import datetime_glob
import re
import soundfile


def rel_path_unix(path, start=None):
    """ Return a relative unix filepath to path either from the current 
        directory or from an optional start directory.

        Args:
            path: str
                Path. Can be unix or windows format.
            start: str
                Optional start directory. Can be unix or windows format.

        Returns:
            u: str
                Relative unix filepath

        Examples:
            >>> from ketos.data_handling.data_handling import rel_path_unix      
            >>> path = "/home/me/documents/projectX/file1.pdf"
            >>> start = "/home/me/documents/"
            >>> u = rel_path_unix(path, start)
            >>> print(u)
            /projectX/
    """
    rel = os.path.relpath(path, start)
    h,t = os.path.split(rel)
    u = '/'
    while len(h) > 0:
        h,t = os.path.split(h)
        u = '/' + t + u

    return u

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
            >>> #separated by '/'. It will also ignore any text after the year,
            >>> # (such as a file extension )
            >>>
            >>> from ketos.data_handling.data_handling import parse_datetime           
            >>> fmt = "%d/%m/%Y*"
            >>> result = parse_datetime("10/03/1942.txt", fmt)
            >>> result.year
            1942
            >>> result.month
            3
            >>> result.day
            10
            >>>
            >>> # Now with the time (hour:minute:second) separated from the date by an underscore
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
            >>> from ketos.data_handling.data_handling import find_files
            >>>
            >>> # Find files that contain 'super' in the name;
            >>> # Do not return the relative path
            >>> find_files(path="ketos/tests/assets", substr="super", fullpath=False)
            ['super_short_1.wav', 'super_short_2.wav']
            >>>
            >>> # find all files with '.h5" in the name
            >>> # Return the relative path
            >>> find_files(path="ketos/tests/assets", substr=".h5")
<<<<<<< HEAD
            ['ketos/tests/assets/15x_same_spec.h5', 'ketos/tests/assets/cod.h5', 'ketos/tests/assets/humpback.h5', 'ketos/tests/assets/morlet.h5', 'ketos/tests/assets/vectors_1_0.h5']
=======
            ['ketos/tests/assets/11x_same_spec.h5', 'ketos/tests/assets/15x_same_spec.h5', 'ketos/tests/assets/cod.h5', 'ketos/tests/assets/humpback.h5', 'ketos/tests/assets/morlet.h5']
>>>>>>> selection_iterator
    """
    # find all files
    allfiles = list()
    if not subdirs:
        f = os.listdir(path)
        for fil in f:
            if fullpath:
                x = path
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
            >>> from ketos.data_handling.data_handling import find_wave_files
            >>>
            >>> find_wave_files(path="ketos/tests/assets", fullpath=False)
            ['2min.wav', 'empty.wav', 'grunt1.wav', 'super_short_1.wav', 'super_short_2.wav']

    """
    wavefiles = find_files(path, '.wav', fullpath, subdirs)
    wavefiles += find_files(path, '.WAV', fullpath, subdirs)
    return wavefiles

def read_wave(file, channel=0, start=0, stop=None):
    """ Read a wave file in either mono or stereo mode.

        Wrapper method around 
        
            https://pysoundfile.readthedocs.io/en/latest/index.html#soundfile.read

        Args:
            file: str
                path to the wave file
            channel: int
                Which channel should be used in case of stereo data (0: left, 1: right) 
            start: int (optional)
                Where to start reading. A negative value counts from the end. 
                Defaults to 0.
            stop: int (optional)
                The index after the last time step to be read. A negative value counts 
                from the end.

        Returns: (rate,data)
            rate: int
                The sampling rate
            data: numpy.array (float)
                A 1d array containing the audio data
        
        Examples:
            >>> from ketos.data_handling.data_handling import read_wave
            >>> rate, data = read_wave("ketos/tests/assets/2min.wav")
            >>> # the function returns the sampling rate (in Hz) as an integer
            >>> type(rate)
            <class 'int'>
            >>> rate
            2000
            >>> # And the actual audio data is a numpy array
            >>> type(data)
            <class 'numpy.ndarray'>
            >>> len(data)
            241664
            >>> # Since each item in the vector is one sample,
            >>> # The duration of the audio in seconds can be obtained by
            >>> # dividing the the vector length by the sampling rate
            >>> len(data)/rate
            120.832
    """
    signal, rate = sf.read(file=file, start=start, stop=stop, always_2d=True)               
    data = signal[:, channel]
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
                >>> from ketos.data_handling.data_handling import to1hot
                >>>
                >>> # An example with two possible labels (0 or 1)
                >>> values = np.array([0,1])
                >>> to1hot(values,depth=2)
                array([[1., 0.],
                       [0., 1.]])
                >>>
                >>> # The same example with 4 possible labels (0,1,2 or 3)
                >>> values = np.array([0,1])
                >>> to1hot(values,depth=4)
                array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.]])
     """
    value = np.int64(value)
    one_hot = np.eye(depth)[value]
    return one_hot

def from1hot(value):
    """Converts the one hot label to binary format

            Args:
                value: scalar or numpy.array | int or float
                    The  label to be converted.
            
            Returns:
                output: int or numpy array (dtype=int64)
                    An int representing the category if 'value' has 1 dimension or an
                    array of m ints if values is an n by m array.

            Example:
                >>> from ketos.data_handling.data_handling import from1hot
                >>>
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
            >>> from ketos.data_handling.data_handling import check_data_sanity
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
            >>> from ketos.data_handling.data_handling import get_image_size
            >>> import pandas as pd
            >>>
            >>> # Load a dataset with images and integer labels
            >>> data = pd.read_pickle("ketos/tests/assets/pd_img_db.pickle")
            >>>
            >>> # Select only the images from the dataset
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

        Raise:
            ValueError
            If seg_name is empty or in wrong format
        Returns:
            (id,label) : tuple (str,str)
            A tuple with the id and label strings.

        Examples:
            >>> from ketos.data_handling.data_handling import parse_seg_name
            >>> seg_name = "id_hydr06_23_l_[2].wav"
            >>> id, label = parse_seg_name(seg_name)
            >>> id
            'hydr06_23'
            >>> label
            '[2]'
            >>>
            >>> seg_name = "id_hydr05_279_l_[2,1].mp3"
            >>> id, label = parse_seg_name(seg_name)
            >>> id
            'hydr05_279'
            >>> label
            '[2,1]'

    """
    id, labels = None, None
    pattern=re.compile(r'id_(.+)_(.+)_l_\[(.+)\].*')
    if not pattern.match(seg_name):
       raise ValueError("seg_name must follow the format  id_*_*_l_[*].")


    splits = seg_name.split("_")
    if len(splits) >= 5:
        id = seg_name.split("_")[1] + "_" + seg_name.split("_")[2]
        tmp = seg_name.split("_")[4]
        labels = tmp.split(".")[0]
    
    return (id,labels)



def divide_audio_into_segs(audio_file, seg_duration, save_to, annotations=None, start_seg=None, end_seg=None, verbose=False):
    """ Divide a large .wav file into a sequence of smaller segments with the same duration.
        

        Name the resulting segments sequentially and save them as .wav files in the specified directory.
        If annotations are provided, this function will check if the segment created emcompasses any labels. If so,
        the label information will be added to the segment name.
        
        Note: segments will be saved following the name pattern "id_*_*_l_*.wav",
            where 'id_' is followed by the name of the original file, underscore ('_') 
            and the a sequence name. 'l_' is followed by the label(s) associated with that segment.
            Ex: 'id_rec03_87_l_[1,3]', 'id_rec03_88_l_[0]

            The start_seg and end_seg arguments can be used to segment only part of audio files,
            which is useful when processing large files in parallel.
            
        Args:
            audio_file:str
            .wav file name (including path).

            seg_duration: float
            desired duration for each segment

            annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "filename": the file name. Must be the the same as audio_file
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
            verbose:bool
                If True, print "Creating segment .... name_of_segment" for each segment.
                        
         Returns:
            None   

        Examples:
            >>> from ketos.data_handling.data_handling import divide_audio_into_segs
            >>> from glob import glob
            >>> import os
            >>>
            >>> # Define the paths to the audio file that will be segmented
            >>> # And the folder where the segments will be saved
            >>> audio_file = "ketos/tests/assets/2min.wav"
            >>> save_dir = "ketos/tests/assets/tmp/divided_segs"
            >>> # Create that folder (if it does not exist)
            >>> os.makedirs(save_dir, exist_ok=True)
            >>> # Difine an annotations dataframe
            >>> annotations = pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
            ...                    'label':[1,2,1], 'start':[5.0, 70.34, 105.8],
            ...                    'end':[6.0,75.98,110.0]})
            >>>
            >>> # Devide the wav file into 2 seconds segments.
            >>> # Uses the annotations dataframe to determine if each segment
            >>> # includes a label names the segments accordingly
            >>> divide_audio_into_segs(audio_file=audio_file,
            ... seg_duration=2.0, annotations=annotations, save_to=save_dir)
            >>> # Count all files have been created in the destination folder
            >>> n_seg = len(glob(save_dir + "/id_2min*.wav"))
            >>> #60 files have been created
            >>> n_seg
            60
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
        if verbose:
            print("Creating segment......", path_to_seg)
        soundfile.write(path_to_seg, sig, rate)

def _filter_annotations_by_filename(annotations, filename):
    """ Filter the annotations DataFrame by the base of the original file name (without the path or extension)

        Args:
        file: str
           The original audio file name without path or extensions.
           Ex: 'file_1' will match the entry './data/sample_a/file_1.wav" in the filename
           column of the annotations DataFrame.

        annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "filename": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file.

        Returns:
            filtered annotations: pandas.DataFrame
            A subset of the annotations DataFrame containing only the entries for the specified file.
            
        Examples:
            >>> from ketos.data_handling.data_handling import _filter_annotations_by_filename
            >>> import pandas as pd
            >>> # Create an annotations dataframe
            >>> annotations = pd.DataFrame({'filename':['2min_01.wav','2min_01.wav','2min_02.wav','2min_02.wav','2min_02.wav'],
            ...                     'label':[1,2,1,1,1], 'start':[5.0, 100.5, 105.0, 80.0, 90.0],
            ...                     'end':[6.0,103.0,108.0, 87.0, 94.0]})
            >>> # Filter the annotations associated with file "2min_01"
            >>> annot_01 = _filter_annotations_by_filename(annotations,'2min_01')
            >>> # enforce desired column ordering
            >>> annot_01 = annot_01[['filename','label','start','end']]
            >>> annot_01
                  filename  label  start    end
            0  2min_01.wav      1    5.0    6.0
            1  2min_01.wav      2  100.5  103.0
                                 

    """
    filtered_indices = annotations.apply(axis=1, func= lambda row: os.path.basename(row.filename).split(".wav")[0] == filename)
    filtered_annotations = annotations[filtered_indices]
    return filtered_annotations

def get_labels(file, start, end, annotations, not_in_annotations=0):
    """ Retrieves the labels that fall in the specified interval.
    
        Args:
        file: str
           The base name (without paths or extensions) for the original audio file. Will be used to match the 'filename' field
           in the annotations Dataframe. Important: The name of the files must be
           unique within the annotations, even if the path is different.
           Ex: '/data/sample_a/file_1.wav' and '/data/sample_b/file_1.wav'

        annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "filename": the file name. Must be the the same as audio_file
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

        Examples:
            >>> from ketos.data_handling.data_handling import get_labels
            >>> import pandas as pd
            >>> audio_file="2min"
            >>> # Create an annotations dataframe
            >>> annotations = pd.DataFrame({'filename':['2min.wav','2min.wav','2min.wav'],
            ...                            'label':[1,2,1], 'start':[5.0, 100.5, 105.0],
            ...                            'end':[6.0,103.0,108.0]})
            >>> # Find all labels between time 4.0 seconds and time 5.0 seconds.
            >>> get_labels(file='2min',start=4.0, end=5.5,
            ...                        annotations=annotations, not_in_annotations=0)
            '[1]'
            >>>
            >>> # Find all labels between time 99.0 seconds and time 110.0 seconds.
            >>> get_labels(file='2min',start=99.0, end=110.0,
            ...                        annotations=annotations, not_in_annotations=0)
            '[2, 1]'

    """
    interval_start = start
    interval_end = end

    data = _filter_annotations_by_filename(annotations, file)
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
    """ Extract a segment from the audio_file according to the start and end tags.

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

        Examples:
            >>> import os
            >>> from ketos.data_handling.data_handling import read_wave
            >>> from ketos.data_handling.data_handling import seg_from_time_tag
            >>>
            >>> # Define the audio file and the destination folder
            >>> audio_file = "ketos/tests/assets/2min.wav"
            >>> save_dir = "ketos/tests/assets/tmp/segs_from_tags"
            >>> # Create the folder
            >>> os.makedirs(save_dir, exist_ok=True)
            >>>
            >>> # Create a segmet starting at 0.5 seconds and ending at 2.5 seconds
            >>> seg_from_time_tag(audio_file=audio_file, start=0.5, end=2.5 , name="seg_1.wav", save_to=save_dir )
            >>>
            >>> # Read the created segment and check its duration
            >>> rate, sig  = read_wave(os.path.join(save_dir, "seg_1.wav"))
            >>> duration = len(sig)/rate
            >>> duration
            2.0

    """
    out_seg = os.path.join(save_to, name)
    sig, rate = librosa.load(audio_file, sr=None, offset=start, duration=end - start)
    soundfile.write(out_seg, sig, rate)


def segs_from_annotations(annotations, save_to):
    """ Generates segments based on the annotations DataFrame.

        Args:
            annotations: pandas.DataFrame
            DataFrame with the the annotations. At least the following columns are expected:
                "filename": the file name. Must be the the same as audio_file
                "label": the label value for each annotaded event
                "start": the start time relative to the beginning of the audio_file.
                "end": the end time relative to the beginning of the file.
            save_to: str
            path to the directory where segments will be saved.
            
            
        Returns:
            None
            
        Examples:
            >>> import os
            >>> from glob import glob
            >>> import pandas as pd
            >>> from ketos.data_handling.data_handling import segs_from_annotations
            >>>
            >>> # Define the audio file and the destination folder
            >>> audio_file_path = "ketos/tests/assets/2min.wav"
            >>> save_dir = "ketos/tests/assets/tmp/from_annot"
            >>>
            >>> # Create a dataframe with annotations
            >>> annotations = pd.DataFrame({'filename':[audio_file_path,audio_file_path,audio_file_path],
            ...                            'label':[1,2,1], 'start':[5.0, 70.5, 105.0],
            ...                            'end':[6.0,73.0,108.0]})
            >>>
            >>> # Segemnt the audio file according with the annotations           
            >>> segs_from_annotations(annotations,save_dir)
            Creating segment...... ketos/tests/assets/tmp/from_annot id_2min_0_l_[1].wav
            Creating segment...... ketos/tests/assets/tmp/from_annot id_2min_1_l_[2].wav
            Creating segment...... ketos/tests/assets/tmp/from_annot id_2min_2_l_[1].wav
            
            

    """ 
    create_dir(save_to)
    for i, row in annotations.iterrows():
        start = row.start
        end= row.end
        base_name = os.path.basename(row.filename).split(".wav")[0]
        seg_name = "id_" + base_name + "_" + str(i) + "_l_[" + str(row.label) + "].wav"
        seg_from_time_tag(row.filename, row.start, row.end, seg_name, save_to)
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

        Examples:
            >>> from ketos.data_handling.data_handling import read_wave
            >>> from ketos.data_handling.data_handling import pad_signal
            >>>
            >>> #Read a very short audio signal
            >>> rate, sig = read_wave("ketos/tests/assets/super_short_1.wav")
            >>> # Calculate its duration (in seconds)
            >>> len(sig)/rate
            0.00075
            >>>
            >>> # Pad the signal
            >>> padded_signal = pad_signal(signal=sig, rate=rate, length=0.5)
            >>> # Now the duration is equal to the 0.5 seconds specified by the 'length' argument
            >>> len(padded_signal)/rate
            0.5
        
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
