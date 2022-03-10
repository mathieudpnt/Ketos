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
    """ Parse date-time data from string.
       
        Returns None if parsing fails.

        If the year is encoded with only two figures, it is parsed to 
        the most recent past year ending in those two figures. For 
        example, 45 would be parsed to 1945 (assuming that the program 
        is being executed in a year earlier than 2045).
        
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
            dt = match.as_datetime()
            if dt > datetime.datetime.now() and "%y" in fmt: dt = dt.replace(year=dt.year-100)

            return dt

    return None

def find_files(path, substr, return_path=True, search_subdirs=False, search_path=False):
    """ Find all files in the specified directory containing the specified substring in their file name

        Args:
            path: str
                Directory path
            substr: str
                Substring contained in file name
            return_path: bool
                If True, path to each file, relative to the top directory. 
                If false, only return the filenames 
            search_subdirs: bool
                If True, search all subdirectories
            search_path: bool
                Search for substring occurrence in relative path rather than just the filename

        Returns:
            files: list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> from ketos.data_handling.data_handling import find_files
            >>>
            >>> # Find files that contain 'super' in the name;
            >>> # Do not return the relative path
            >>> find_files(path="ketos/tests/assets", substr="super", return_path=False)
            ['super_short_1.wav', 'super_short_2.wav']
            >>>
            >>> # find all files with '.h5" in the name
            >>> # Return the relative path
            >>> find_files(path="ketos/tests/", substr="super", search_subdirs=True)
            ['assets/super_short_1.wav', 'assets/super_short_2.wav']
    """
    # find all files
    all_files = []
    if search_subdirs:
        for dirpath, _, files in os.walk(path):
            if return_path:
                all_files += [os.path.relpath(os.path.join(dirpath, f), path) for f in files]
            else:
                all_files += files
    else:
        all_files = os.listdir(path)

    # select those that contain specified substring
    if isinstance(substr, str): substr = [substr]
    files = []
    for f in all_files:
        for ss in substr:
            if search_path: s = f
            else: s = os.path.basename(f)
            if ss in s:
                files.append(f)
                break

    # sort alphabetically
    files.sort()
    return files


def find_wave_files(path, return_path=True, search_subdirs=False, search_path=False):
    """ Find all wave files in the specified directory

        Args:
            path: str
                Directory path
            return_path: bool
                If True, path to each file, relative to the top directory. 
                If false, only return the filenames 
            search_subdirs: bool
                If True, search all subdirectories
            search_path: bool
                Search for substring occurrence in relative path rather than just the filename

        Returns:
            : list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> from ketos.data_handling.data_handling import find_wave_files
            >>>
            >>> find_wave_files(path="ketos/tests/assets", return_path=False)
            ['2min.wav', 'empty.wav', 'grunt1.wav', 'super_short_1.wav', 'super_short_2.wav']

    """
    return find_files(path, substr=['.wav', '.WAV'], 
        return_path=return_path, search_subdirs=search_subdirs, search_path=search_path)

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
    data = np.asfortranarray(data)
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
