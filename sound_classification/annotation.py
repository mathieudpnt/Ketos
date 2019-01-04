""" Annotation module within the sound_classification package

This module includes classes and functions useful for handling annotations
of audio data and spectrograms. 

Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: To package code useful for handling data, deriving features and 
             creating Deep Neural Networks for sound classification projects.
     
    License:

"""
import numpy as np
import math

def tostring(box):
    """ Convert an array, tuple or list into a string.
        
        Args:
            box: array, tuple or list
                Array, tuple or list that will be converted into a string.

        Returns:
            s: str
                String representation of array/tuple/list.
    """
    if box is None:
        return ''

    box = np.array(box).tolist()

    s = str(box)
    s = s.replace(' ', '')
    s = s.replace('(', '[')
    s = s.replace(')', ']')

    return s


class AnnotationHandler():
    """ Parent class for the AudioSignal and Spectrogram classes

        Args:
            labels: tuple(int)
                List of annotation labels
            boxes: 2D tuple(int)
                List of 2D or 4D tuples, specifying a time interval (in seconds),
                and optionally also a frequency interval (in Hz), for each annotation. 
                The format is (t_min, t_max, f_min, f_max)
    """
    def __init__(self, labels=None, boxes=None):

        self.labels = labels
        self.boxes = boxes

        if self.labels is None:
            self.labels = []
        if self.boxes is None:
            self.boxes = []

    def annotate(self, labels, boxes):
        """ Add a set of annotations.
            
            Args:
                labels: list(int)
                    Annotation labels.
                boxes: list(tuple)
                    Annotation boxes, specifying the start and stop time of the annotation 
                    and, optionally, the minimum and maximum frequency.
                    For example, box=(10.0, 12.2) specifies the start and stop times as being
                    10.0 and 12.2 seconds, respectively, while no constraints are placed on the 
                    frequency; box=(10.0, 12.2, 400., 6000.) specifies the same start and stop 
                    times, but now also specifies a frequency range of 400-6000 Hz.
        """
        if np.ndim(labels) == 0:
            self.labels.append(labels)
            boxes = self._ensure4D(boxes)
            self.boxes.append(boxes)
        else:
            assert len(labels) == len(boxes), 'number of boxes must be equal to number of labels'

            for l,b in zip(labels, boxes):
                self.labels.append(l)
                b = self._ensure4D(b)
                self.boxes.append(b)

        for b in self.boxes:
            b = np.array(b).tolist()

    def _ensure4D(self, b):
        """ Ensure that the annotation box has four entries.
            Set the minimum frequency to 0 and the maximum 
            frequency to infinity, if these are missing.
            
            Args:
                b: list
                   Annotation box 
        """
        if len(b) == 2:
            b = [b[0], b[1], 0, math.inf]
        
        return b

    def delete_annotations(self, id=None):
        """ Delete the annotation with the specified id.
            
            Args:
                id: int
                    ID of the annotation to be deleted.
        """
        if id is None:
            self.labels = []
            self.boxes = []
        else:
            # sort id's in ascending order 
            id = sorted(id, reverse=True)
            # loop over id's
            for i in id:
                if i < len(self.labels):
                    # delete specified annotations
                    del self.labels[i]
                    del self.boxes[i]

    def cut_annotations(self, t1=0, t2=math.inf, f1=0, f2=math.inf):
        """ Crop all annotations in time and/or frequency.
            
            Args:
                t1: float
                    New start time in seconds
                t2: float
                    New stop time in seconds
                f1: float
                    New minimum frequency in Hz
                f2: float
                    New maximum frequency in Hz
        """
        if t1 is None: t1 = 0
        if t2 is None: t2 = math.inf
        if f1 is None: f1 = 0
        if f2 is None: f2 = math.inf

        labels, boxes = [], []

        # loop over annotations
        for l, b in zip(self.labels, self.boxes):

            # check if box overlaps with cut
            if b[0] >= t1 and b[0] < t2 and not (b[3] < f1 or b[2] > f2):

                # update box boundaries
                b[0] -= t1
                b[1] -= t1
                b[1] = min(b[1], t2-t1)
                b[2] = max(b[2], f1)
                b[3] = min(b[3], f2)
                labels.append(l)
                boxes.append(b)

        return labels, boxes

    def _shift_annotations(self, delay=0):
        """ Shift all annotations by a fixed time interval.
            
            Args:
                delay: float
                    Size of time shift in seconds.
        """
        for b in self.boxes:
            b[0] += delay
            b[1] += delay
        