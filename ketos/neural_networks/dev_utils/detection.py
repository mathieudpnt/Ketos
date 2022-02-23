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

""" detection sub-module within the ketos.neural_networks.dev_utils module

    This module provides auxiliary functions to incorporate pre-trained ketos classifiers models into detection tools

    Contents:
        

"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from ketos.audio.audio_loader import AudioLoader
from ketos.data_handling.data_feeding import BatchGenerator

def compute_avg_score(score_vector, win_len):
    """ Compute a moving average of the score vector.

        Args:
            score_vector: numpy array
                1d numpy array containing the target class classification score for each input
            win_len:int
                The window length for the moving average. Must be an odd integer

        Returns:
            numpy arrays
                One numpy arrays with the average scores each time step
    """
    assert isinstance(win_len, int) and win_len%2 == 1, 'win_len must be an odd integer'

    num_pad = int((win_len - 1) / 2)
    x = score_vector.astype(float)

    num_frames = len(score_vector) - 2*num_pad
    indices = np.tile(np.arange(0, win_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames, 1), (win_len, 1)).T
    frames = x[indices.astype(np.int32, copy=False)]

    avg_score = np.nanmean(frames, axis=1)

    avg_score = np.pad(avg_score, (num_pad, num_pad), 'constant', constant_values=(0, 0)) 

    return avg_score


def map_detection_to_time(det_start, det_end, batch_start_timestamp, batch_end_timestamp, step, spec_dur, buffer):
    """ Converts the start and end of a detection from the position in the scores vecotr to time.

        Args:
            det_start: int
                The detection start expressed as an index in the scores vector.
            det_end: int
                The detection end (the last score that is part of the detection) expressed as an index in the scores vector. 
            batch_start_timestap:float
                The timestamp (in seconds from the beginning of the file) of the first score in the scores vector
                (i.e.: the score of the first input spectrogram in that batch)
            batch_end_timestap:float
                The timestamp (in seconds from the beginning of the file) of the last score in the scores vector
                (i.e.: the score of the last input spectrogram in that batch)
            step: float
                The time interval(in seconds) between the starts of each contiguous input spectrogram.
                For example, a step=0.5 indicates that the first spectrogram starts at time 0.0s (from the beginning of the audio file), the second at 0.5s, etc.
            spec_dur: float
                The duration of each spectrogram in seconds
            buffer: float
                Time (in seconds) to be added around the detection.

        Raises:
            ValueError: 
                If det_end is lower than det_start

        Returns:
            time_start, duration:float
                The corresponding start (in seconds from the beggining of the file) and duration
    """
    if det_end < det_start:
        raise ValueError("'det_end' cannot be lower than 'det_start'")

    time_start =  det_start * step - buffer 
    time_start += batch_start_timestamp
    time_start = max(batch_start_timestamp, time_start)
    
    time_end =  det_end * step + buffer 
    time_end += batch_start_timestamp
    time_end = min(batch_end_timestamp, time_end)

    duration = (time_end + step) - time_start
    
    return time_start, duration
    

def group_detections(scores_vector, batch_support_data, buffer=0.0, step=0.5, spec_dur=3.0, threshold=0.5):
    """ Groups time steps with a detection score above the specified threshold.

        Consecutive detections are grouped into one single detection represented by the 
        time interval (start-end, in seconds from beginning of the file).

        Args:
            scores_vector: numpy array
                1d numpy array containing the target class classification score for each input
            batch_support_data: numpy array
                An array of shape n x 2, where n is the batch size. The second dimension contains the filename and the start timestamp for each input in the batch
            buffer: float
                Time (in seconds) to be added around the detection
            step: float
                The time interval(in seconds) between the starts of each contiguous input spectrogram.
                For example, a step=0.5 indicates that the first spectrogram starts at time 0.0s (from the beginning of the audio file), the second at 0.5s, etc.
            spec_dur: float
                The duration of each spectrogram in seconds
            threshold: float or list of floats
                Minimum score value for a time step to be considered as a detection.
                
        Returns:
            det_timestamps: list of tuples
                The detections time stamp. Each item in the list is a tuple with the filename, start time, duration and score for that detection.
                The filename corresponds to the file where the detection started. 
                If a list of threshold values is specified, the returned object will be a list of lists with len(det_timestamps) = len(thresholds)         
    """
    thresholds = threshold if isinstance(threshold, list) else [threshold]

    det_timestamps = [[] for _ in range(len(thresholds))]

    for i in range(len(thresholds)):
        thres = thresholds[i]

        det_vector = np.where(scores_vector >= thres, 1.0, 0.0)
        within_det = False
        filename_vector = batch_support_data[:,0]

        prev_det_index = 0
        for det_index,det_value in enumerate(det_vector):

            is_new_file = (filename_vector[det_index] != filename_vector[max(0,prev_det_index)])
            is_end_of_vector = (det_index == len(det_vector) - 1)

            if det_value == 1.0 and not within_det: #start new detection
                start = det_index
                within_det = True
            
            if (det_value == 0 or is_new_file or is_end_of_vector) and within_det: #end current detection
                if is_end_of_vector:
                    end = det_index
                else:
                    end = prev_det_index
                
                within_det = False
                filename = filename_vector[start]

                # From all timestamps within the batch, select only the timestamps for the file containing the detection start
                file_timestamps = batch_support_data[filename_vector==filename]

                batch_start_timestamp = float(file_timestamps[0,1])
                batch_end_timestamp = float(file_timestamps[-1,1])
                time_start, duration = map_detection_to_time(start, end, step=step, spec_dur=spec_dur, batch_start_timestamp=batch_start_timestamp, batch_end_timestamp=batch_end_timestamp, buffer=buffer)
                score = np.average(scores_vector[start:(end + 1)])
                if np.isnan(score) and end==start:
                    score = scores_vector[start]
                
                det_timestamps[i].append((filename, time_start, duration, score))

            prev_det_index = det_index #update index of previous instance

    if not isinstance(threshold, list): 
        det_timestamps = det_timestamps[0]

    return det_timestamps

def process_batch(batch_data, batch_support_data, model, buffer=0, step=0.5, spec_dur=3.0, threshold=0.5, win_len=1, group=False):
    """ Runs one batch of (overlapping) spectrogram throught the classifier.

        Args:
            batch_data: numpy array
                An array with shape n,f,t,  where n is the number of spectrograms in the batch, t is the number of time bins and f the number of frequency bins.
            batch_support_data: numpy array
                An array of shape n x 2, where n is the batch size. The second dimension contains the filename and the start timestamp for each input in the batch
            model: ketos model
                The ketos trained classifier
            buffer: float
                Time (in seconds) to be added around the detection
            step: float
                The time interval(in seconds) between the starts of each contiguous input spectrogram.
                For example, a step=0.5 indicates that the first spectrogram starts at time 0.0s (from the beginning of the audio file), the second at 0.5s, etc.
            spec_dur: float
                The duration of each input spectrogram in seconds
            threshold: float or list of floats
                Minimum score value for a time step to be considered as a detection.
            win_len:int
                The windown length for the moving average. Must be an odd integer. The default value is 5.
            group:bool
                If False, return the filename, start, duration and scores for each spectrogram with score above the threshold. In this case, the duration will always be the duration of a single spectrogram.
                If True (default), average scores over(overlapping) spectrograms and group detections that are immediatelly next to each other. In this case, the score given for that detection will be the
                average score of all spectrograms comprising the detection event.

        Returns:
            batch_detections: list
                An array with all the detections in the batch. Each detection (first dimension) consists of the filename, start, duration and score.
                The start is given in seconds from the beginning of the file and the duration in seconds. 
                If a list of threshold values is specified, the returned object will be a list of arrays with len(batch_detections) = len(thresholds)         

    """
    thresholds = threshold if isinstance(threshold, list) else [threshold]

    scores = model.run_on_batch(batch_data, return_raw_output=True)

    if win_len == 1:
        scores = scores[:,1]
    else:
        scores = compute_avg_score(scores[:,1], win_len=win_len)
        
    if group == True:
        batch_detections = group_detections(scores, batch_support_data, buffer=buffer, step=step, spec_dur=spec_dur, threshold=thresholds) 

    else:
        batch_detections = []

        for thres in thresholds:
            threshold_indices = scores >= thres
            batch_det = np.vstack([batch_support_data[threshold_indices,0], batch_support_data[threshold_indices,1], np.repeat(spec_dur, sum(threshold_indices)), scores[threshold_indices]])
            if batch_det.shape[1] == 0:
                batch_det = []
            else:
                batch_det = [(d[0], float(d[1]), float(d[2]), float(d[3])) for d in batch_det.T]

            batch_detections.append(batch_det)


    if not isinstance(threshold, list): 
        batch_detections = batch_detections[0]

    return batch_detections


def process(provider, **kwargs):
    """ Use classifier to process audio clips (waveform or spectrogram).

        Delegates to :func:`detection.process_audio_loader` or :func:`detection.process_batch_generator`
        depending on the type of the `provider` argument.

        See these functions for a description of the required input arguments.

        Args:
            provider: instance of ketos.audio.audio_loader.AudioFrameLoader or ketos.data_handling.data_feeding.BatchGenerator
    """
    if isinstance(provider, AudioLoader):
        return process_audio_loader(provider, **kwargs)

    elif isinstance(provider, BatchGenerator):
        return process_batch_generator(provider, **kwargs)


def process_audio_loader(audio_loader, model, batch_size=128, threshold=0.5, buffer=0, win_len=1, group=False, progress_bar=False, merge=False):
    """ Use an audio_loader object to compute spectrogram from the audio files and process them with the trained classifier.

        TODO: Improve the implementation so that it is not limited to audio_loader.batch_size = 1

        Args:
            audio_loader: a ketos.audio.audio_loader.AudioFrameLoader object
                An audio loader that computes spectrograms from the audio audio files as requested
            model: ketos model
                The ketos trained classifier
            batch_size:int
                The number of spectrogram to process at a time.
            threshold: float or list of floats
                Minimum score value for a time step to be considered as a detection.
            buffer: float
                Time (in seconds) to be added around the detection
            win_len:int
                The windown length for the moving average. Must be an odd integer. The default value is 5.   
            group:bool
                If False, return the filename, start, duration and scores for each spectrogram with score above the threshold. In this case, the duration will always be the duration of a single spectrogram.
                If True (default), average scores over(overlapping) spectrograms and group detections that are immediatelly next to each other. In this case, the score given for that detection will be the
                average score of all spectrograms comprising the detection event.
            progress_bar: bool
                Show progress bar.  
            merge: bool
                Apply :func:`merge_overlapping_detections` to the detections before they are returned. Default is False.

        Returns:
            detections: list
                List of detections.
                If a list of threshold values is specified, the returned object will be a list of lists with len(detections) = len(thresholds)         
    """
    assert isinstance(win_len, int) and win_len%2 == 1, 'win_len must be an odd integer'

    assert audio_loader.batch_size == 1, "audio_loader must have batch size of 1"

    thresholds = threshold if isinstance(threshold, list) else [threshold]
        
    n_extend = int((win_len - 1) / 2)

    n_batches = audio_loader.num() // batch_size
    last_batch_size = batch_size + (audio_loader.num() % batch_size)

    if n_batches == 0: 
        batch_sizes = [audio_loader.num()]
    elif n_batches == 1:
        batch_sizes = [last_batch_size]
    else:
        batch_sizes = [batch_size + n_extend] + [batch_size + 2 * n_extend for _ in range(n_batches - 2)] + [last_batch_size + n_extend]

    detections = [[] for _ in range(len(thresholds))]
    specs_prev_batch = []
    duration = None
    step = 0

    for siz in tqdm(batch_sizes, disable = not progress_bar): 
        batch_data, batch_support_data = [], []

        # first, collect data from the last specs from previous batch, if any            
        for spec in specs_prev_batch: 
            batch_data.append(spec.data)
            support_data = (spec.filename, spec.offset)
            batch_support_data.append(support_data)
            duration = spec.duration()

        # then, load specs from present batch
        specs_prev_batch = []
        while len(batch_data) < siz:
            spec = next(audio_loader)
            batch_data.append(spec.data)
            support_data = (spec.filename, spec.offset)
            batch_support_data.append(support_data)
            if siz - len(batch_data) < 2 * n_extend: specs_prev_batch.append(spec) # store last few specs
            duration = spec.duration()

        if len(batch_support_data) >= 2:
            step = batch_support_data[1][1] - batch_support_data[0][1]

        if step <= 0: step = duration

        batch_support_data = np.array(batch_support_data)
        batch_data = np.array(batch_data)

        batch_detections = process_batch(batch_data=batch_data, batch_support_data=batch_support_data, model=model, threshold=thresholds, 
                                        buffer=buffer, step=step, spec_dur=duration, win_len=win_len, group=group)

        for i in range(len(thresholds)):
            if len(batch_detections[i]) > 0: detections[i] += batch_detections[i]


    if merge:
        for i in range(len(thresholds)):
            detections[i] = merge_overlapping_detections(detections[i])

    if not isinstance(threshold, list): 
        detections = detections[0]

    return detections


def process_batch_generator(batch_generator, model, duration=3.0, step=0.5, threshold=0.5, buffer=0, win_len=1, group=False, merge=False):
    """ Use a batch_generator object to process pre-computed spectrograms stored in an HDF5 database with the trained classifier.

        Args:
            batch_generator: a ketos.data_handling.data_feeding.BatchGenerator object
                A batch_generator that loads pre-computed spectrograms from the a HDF5 database files as requested
            model: ketos model
                The ketos trained classifier
            duration: float
                The duration of each input spectrogram in seconds
            step: float
                The time interval(in seconds) between the starts of each contiguous input spectrogram.
                For example, a step=0.5 indicates that the first spectrogram starts at time 0.0s (from the beginning of the audio file), the second at 0.5s, etc.
            threshold: float or list of floats
                Minimum score value for a time step to be considered as a detection.
            buffer: float
                Time (in seconds) to be added around the detection
            win_len:int
                The windown length for the moving average. Must be an odd integer. The default value is 5.      
            group:bool
                If False, return the filename, start, duration and scores for each spectrogram with score above the threshold. In this case, the duration will always be the duration of a single spectrogram.
                If True (default), average scores over(overlapping) spectrograms and group detections that are immediatelly next to each other. In this case, the score given for that detection will be the
                average score of all spectrograms comprising the detection event.
            merge: bool
                Apply :func:`merge_overlapping_detections` to the detections before they are returned. Default is False.

        Returns:
            detections: list
                List of detections.
                If a list of threshold values is specified, the returned object will be a list of lists with len(detections) = len(thresholds)         
    """ 
    thresholds = threshold if isinstance(threshold, list) else [threshold]

    detections = [[] for _ in range(len(thresholds))]

    for b in range(batch_generator.n_batches):
        batch_data, batch_support_data = next(batch_generator)

        batch_detections = process_batch(batch_data=batch_data, batch_support_data=batch_support_data, model=model, threshold=thresholds, 
                                        buffer=buffer, spec_dur=duration, step=step, win_len=win_len, group=group)
        
        for i in range(len(thresholds)):
            if len(batch_detections[i]) > 0: detections[i] += batch_detections[i]


    if merge:
        for i in range(len(thresholds)):
            detections[i] = merge_overlapping_detections(detections[i])

    if not isinstance(threshold, list): 
        detections = detections[0]

    return detections


def transform_batch(x,y):
    """ Transform the data loaded from the database to the format expected by process_batch

        Args:
            x: numpy array
                A batch of spectrograms of shape batch_size, time bins, frequency bins
            y: numpy array
                Suppporting information for the batch (the filename and offset fields from the hdf5 dataset)
        
        Returns:
            tuple:
                transformed_x:numpy array
                    x, unmodified
                transformed_y:numpy array
                    y reshaped to shape batch_size, 2. type converted to str (from bytes string).
         """
  
    transformed_x = x
    filenames = y['filename']
    timestamps = y['offset']
    transformed_y = np.column_stack((filenames, timestamps))
    transformed_y = transformed_y.astype(str)
        
    return transformed_x, transformed_y


def merge_overlapping_detections(detections):
    """ Merge overlapping or adjacent detections.

        The score of the merged detection is computed as the average of the individual detection scores.

        Note: The detections are assumed to be sorted by start time in chronological order.

        Args:
            detections: numpy.array
                List of detections
        
        Returns:
            merged: numpy.array
                List of merged detections
    """
    num_det = len(detections)
    if num_det <= 1: return detections

    merged = [detections[0]]
    for i in range(1, num_det):
        filename = detections[i][0]
        start = detections[i][1]
        duration = detections[i][2]
        end = start + duration
        score = detections[i][3]
        filename_prev = merged[-1][0]
        start_prev = merged[-1][1]
        duration_prev = merged[-1][2]
        end_prev = start_prev + duration_prev        
        score_prev = merged[-1][3]
        
        # detections do not overlap, nor are they adjacent
        if start > end_prev or filename != filename_prev:
            merged.append(detections[i])

        # detections overlap, or adjacent to one another
        else:
            merged_dur = max(end, end_prev) - min(start,start_prev)
            avg_score = 0.5 * (score + score_prev) 
            merged_det = (filename, start_prev, merged_dur, avg_score)
            merged[-1] = merged_det #replace 

    return merged
    
def save_detections(detections, save_to):
    """ Save the detections to a csv file

        The resulting .csv is separated by commas, with each row representing one detection and has the following columns:

            filename: The name of the audio file where the detection was registered. In case the detection starts in one file and ends in the next, \
                      the name of the first file is registered.

            start:    Start of the detection (in seconds from the beginning of the file)

            duration: Length of the detection (in seconds)

            score:    The sore given to the detection by the trained neural network ([0-1])

        Args:
            detections: numpy.array
                List of detections
            save_to:string
                The path to the .csv file where the detections will be saved.
                Example: "/home/user/detections.csv"
    """
    if len(detections) == 0: return

    a = np.array(detections)
    df = pd.DataFrame({'filename':a[:,0], 'start':a[:,1], 'duration':a[:,2], 'score':a[:,3]})
    include_header = not os.path.exists(save_to)
    df.to_csv(save_to, mode='a', index=False, header=include_header)

