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