import numpy as np
from ketos.audio_processing.annotation import AnnotationHandler

def main():

    print(' this is bla ...')

    an = AnnotationHandler(labels=[1,2], boxes=[[1,2],[3,4]])

    print(an.labels)


if __name__ == '__main__':
   main()




