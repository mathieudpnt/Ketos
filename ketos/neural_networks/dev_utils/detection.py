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
                The detection end expressed as an index in the scores vector.
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

        Retuirns:
            time_start, duration:float
                The corresponding start (in seconds from the beggining of the file) and duration

    """
    time_start =  det_start * step - buffer + 0.5 * spec_dur
    duration = (det_end - det_start + 1) * step + 2 * buffer
    time_start += batch_start_timestamp
    time_start = max(batch_start_timestamp, time_start)
    return time_start, duration
    

def group_detections(scores_vector, batch_support_data,  buffer=0.0, step=0.5, spec_dur=3.0, threshold=0.5):
    """ Groups time steps with a detection score above the specified threshold.

        Consecutive detections are grouped into one single detection represented by the time interval (start-end, in seconds from beginning of the file).

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
            threshold: float
                Minimum score value for a time step to be considered as a detection.

        Returns:
            det_timestamps:list of tuples
            The detections time stamp. Each item in the list is a tuple with the start time, duration, score and filename for that detection.
            The filename corresponds to the file where the detection started.          
    """
    det_vector = np.where(scores_vector >= threshold, 1.0, 0.0)
    det_timestamps = []
    within_det = False
    filename_vector = batch_support_data[:,0]

    for det_index,det_value in enumerate(det_vector):
        if det_value == 1.0 and not within_det:
            start = det_index
            within_det = True
        if det_value == 0 and within_det:
            end = det_index - 1
            within_det = False
            filename = filename_vector[start]
            # From all timestamps within the batch, select only the timestamps for the file containing the detection start
            file_timestamps = batch_support_data[filename_vector==filename]
            batch_start_timestamp = float(file_timestamps[0,1])
            batch_end_timestamp = float(file_timestamps[-1,1])
            time_start, duration = map_detection_to_time(start, end, step=step, spec_dur=spec_dur, batch_start_timestamp=batch_start_timestamp, batch_end_timestamp=batch_end_timestamp, buffer=buffer)
            score = np.average(scores_vector[start:end+1])
            if np.isnan(score) and end==start:
                score = scores_vector[start]
            
            det_timestamps.append((filename, time_start, duration, score))

    return det_timestamps

def process_batch(batch_data, batch_support_data, model, buffer=1.0, step=0.5, spec_dur=3.0, threshold=0.5, win_len=5):
    """ Runs one batch of (overlapping) spectrograms through the classifier.

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
            threshold: float
                Minimum score value for a time step to be considered as a detection.
            win_len:int
                The windown length for the moving average. Must be an odd integer. The default value is 5.            

        Returns:
            batch_detections: list
                An array with all the detections in the batch. Each detection (first dimension) consists of the filename, start, duration and score.
                The start is given in seconds from the beginning of the file and the duration in seconds. 

    """
    scores = model.run_on_batch(batch_data, return_raw_output=True)

    avg_scores = compute_avg_score(scores[:,1], win_len=win_len)

    batch_detections = group_detections(avg_scores, batch_support_data, buffer=buffer, step=step, spec_dur=spec_dur, threshold=threshold) 

    return batch_detections
    
