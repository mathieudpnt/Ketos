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

        if np.ndim(labels) == 0:
            self.labels.append(labels)
            self.boxes.append(boxes)
        else:
            assert len(labels) == len(boxes), 'number of boxes must be equal to number of labels'

            for l,b in zip(labels, boxes):
                self.labels.append(l)
                self.boxes.append(b)

    def delete_annotations(self, id=None):

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

    def _cut_annotations(self, t1=0, t2=math.inf, f1=0, f2=math.inf):

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
        for b in self.boxes:
            b[0] += delay
            b[1] += delay