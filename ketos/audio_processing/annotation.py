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
from ketos.util import ndim
import numpy as np
import math


class AnnotationHandler():
    """ Parent class for the AudioSignal and Spectrogram classes.

        An annotation consists of an integer label (0,1,2,...) and 
        a bounding box that delimits the audio segment of interest 
        in time and optionally frequency. 
        
        The bounding box can be a list of two or four floats. A list 
        of two floats is interpreted as the start and end time of the 
        segment (in seconds); a list of four floats is interpreted as 
        start time, end time, minimum frequency, and maximum frequency 
        (with frequency expressed in Hz).

        Attributes:
            labels: list(int)
                List of annotation labels
            boxes: list(tuple)
                List of bounding boxes, each specifying a time interval (in seconds)
                and optionally also a frequency interval (in Hz).
                The format is [t_min, t_max, f_min, f_max]
    """
    def __init__(self, labels=None, boxes=None):

        self.labels = []
        self.boxes = []

        if labels is not None:
            self.annotate(labels, boxes)

    def annotate(self, labels, boxes):
        """ Add a set of annotations.
            
            Args:
                labels: list(int)
                    Annotation labels.
                boxes: list(tuple)
                    Annotation boxes, specifying the start and stop time of the annotation 
                    and, optionally, the minimum and maximum frequency.

            Example:
                >>> from ketos.audio_processing.annotation import AnnotationHandler
                >>> 
                >>> handler = AnnotationHandler()
                >>> labels = [0, 1]
                >>> boxes = [[10.0, 12.2, 110., 700.],[30., 34.]]
                >>> handler.annotate(labels, boxes)
                >>> print(handler.labels)
                [0, 1]
                >>> print(handler.boxes)
                [[10.0, 12.2, 110.0, 700.0], [30.0, 34.0, 0, inf]]
        """
        if ndim(labels) == 0:
            labels = [labels]

        if ndim(boxes) == 1:
            boxes = [boxes]

        assert len(labels) == len(boxes), 'number of labels must equal number of boxes'
        assert ndim(labels) == 1, 'labels list has invalid dimension'
        assert ndim(boxes) == 2, 'boxes list has invalid dimension'

        for l,b in zip(labels, boxes):
            b = self._ensure4D(b)
            self.labels.append(l)
            self.boxes.append(b)

        for b in self.boxes:
            b = np.array(b).tolist()

    def _ensure4D(self, b):
        """ Ensure that the annotation box has four entries.

            Set the minimum frequency to 0 and the maximum 
            frequency to infinity, if these are missing.
            
            Args:
                b: list or tuple
                   Bounding box with two or four entries 
        """
        if len(b) == 2:
            b = [b[0], b[1], 0, math.inf]
        
        assert len(b) == 4, 'Found box with {0} entries; all boxes must have either 2 or 4 entries'.format(len(b))
        
        return b

    def delete_annotations(self, id=None):
        """ Delete the annotation with the specified ID(s).

            If no ID is spefied, all annotations are deleted.
            
            Args:
                id: list(int)
                    ID of the annotation to be deleted.

            Example:
                >>> from ketos.audio_processing.annotation import AnnotationHandler
                >>> 
                >>> labels = [0, 1]
                >>> boxes = [[10.0, 12.2, 110., 700.],[30., 34.]]
                >>> handler = AnnotationHandler(labels, boxes)
                >>> handler.delete_annotations(0)
                >>> print(handler.labels)
                [1]
                >>> print(handler.boxes)
                [[30.0, 34.0, 0, inf]]
        """
        if id is None:
            self.labels = []
            self.boxes = []
        else:
            if ndim(id) == 0:
                id = [id]
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
                b0 = b[0] - t1
                b1 = b[1] - t1
                b1 = min(b1, t2-t1)
                b2 = max(b[2], f1)
                b3 = min(b[3], f2)
                box = [b0, b1, b2, b3]

                labels.append(l)
                boxes.append(box)

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
        
    def _scale_annotations(self, scale=0):
        """ Scale the time axis by a constant factor.
            
            Args:
                scale: float
                    Scaling factor.
        """
        for b in self.boxes:
            b[0] *= scale
            b[1] *= scale
