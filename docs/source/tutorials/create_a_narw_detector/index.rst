Tutorial: Creating a detector
=============================

In the `Training a ResNet classifier <https://docs.meridian.cs.dal.ca/ketos/tutorials/train_a_narw_classifier/index.html>`_  tutorial, we trained a deep neural network to differentiate between the North Atlantic right whale upcalls and background noises.

We can use the trained network as the basis for a tool that is more convenient to our workflow. What this tool will look like largely depends on your needs.
Here we provide a command-line interface (CLI) to a program that will take long .wav files as inputs and output a list of upcall detections in a .csv file.
You can use the CLI tool with the previously trained model. It's also a good starting point for you to quickly put your own models to use. 

We also provide a follow-up tutorial to the *Training a ResNet classifier* tutorial. This will be helpful if you need to wrap your own classifier in a different tool to better match your workflow.
But even if the example CLI we provide here serves all your needs, following the tutorial will give you a better understanding of how it works. 
The same principles can also be used to integrate your ketos model into different interfaces. If instead of a CLI you want to use a trained model in a web-app for example, the functions used in the tutorial will also be helpful.


**Tutorial**


You can access the tutorial here: :doc:`tutorial <tutorial>`. You can either follow the tutorial online (read-only) or you can download a jupyter 
notebook version of the tutorial and sample data which will allow you to run the code yourself (interactive).
  
.. toctree::

    tutorial

**CLI**


You can download the program from here: :download:`detector.py <https://gitlab.meridian.cs.dal.ca/public_projects/ketos_tutorials/-/blob/master/tutorials/create_a_narw_detector/detector.py>`
The pre-trained narw model from here : :download:`narw.kt <https://gitlab.meridian.cs.dal.ca/public_projects/ketos_tutorials/-/blob/master/tutorials/create_a_narw_detector/narw.kt>`
And sample data can be download from here :download:`data <https://gitlab.meridian.cs.dal.ca/public_projects/ketos_tutorials/-/blob/master/tutorials/create_a_narw_detector/data.zip>`


The detector program has a few parameters that can be used to adjust its bahavior.

.. code-block::

  detector -h

  usage: detector.py [-h] [--model MODEL] [--audio_folder AUDIO_FOLDER]
                   [--input_list INPUT_LIST] [--output OUTPUT]
                   [--num_segs NUM_SEGS] [--step_size STEP_SIZE]
                   [--buffer BUFFER] [--win_len WIN_LEN]
                   [--threshold THRESHOLD] [--show_progress | --hide_progress]
                   [--with_group | --without_group]
                   [--with_merge | --without_merge]

  Ketos acoustic signal detection script

  optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         path to the trained ketos classifier model
  --audio_folder AUDIO_FOLDER
                        path to the folder containing the .wav files
  --input_list INPUT_LIST
                        a .txt file listing all the .wav files to be
                        processed. If not specified, all the files in the
                        folder will be processed.
  --output OUTPUT       the .csv file where the detections will be saved. An
                        existing file will be overwritten.
  --num_segs NUM_SEGS   the number of segments to hold in memory at one time
  --step_size STEP_SIZE
                        step size (in seconds) used for the sliding window
  --buffer BUFFER       Time (in seconds) to be added on either side of every
                        detected signal
  --win_len WIN_LEN     Length of score averaging window (no. time steps).
                        Must be an odd integer.
  --threshold THRESHOLD
                        minimum score for a detection to be accepted (ranging
                        from 0 to 1)
  --show_progress       Shows a progress bar with an estimated completion time.
  --hide_progress       Does not shows the progress bar
  --with_group          Group ovelrapping segments and computes averages the detection score detections 
  --without_group       Does not group overlapping segments, treating considering the detections score for each inout segment individually instead of using an average
  --with_merge          Merge consecutive detections
  --without_merge       Does not merge consecutive detections


If you unzip data.zip, you will find the audio folder, with three .wav files with 30 minutes each.

The following line runs the detector on all the files within the audio folder with a threshold of 0.7.

.. code-block::

  python detector.py --model=narw.kt --audio_folder=audio --threshold=0.7 --output=detections_no_overlap.csv 

Each audio file is divided into 3 seconds long segments, which are then passed to the trained model as spectrograms.
By default, there's no overlapping, so segments gor from 0 to 3 seconds, 3 to 6 seconds, and so on.

The results are saved in the 'detections_no_overlap.csv'

.. code-block::

  filename,     start,    duration,   score  
  sample_1.wav, 225.0,    3.008,    0.8094866  
  sample_1.wav, 951.0,    3.008,    0.7172976  
  sample_1.wav, 1128.0,   3.008,    0.9932656  
  sample_1.wav, 1152.0,   3.008,    0.910479  
  sample_1.wav, 1194.0,   3.008,    0.9637392  
  sample_1.wav, 1209.0,   3.008,    0.80935943  
  sample_1.wav, 1356.0,   3.008,    0.9869077  
  sample_1.wav, 1437.0,   3.008,    0.8602179  
  sample_1.wav, 1488.0,   3.008,    0.9796428  
  sample_1.wav, 1509.0,   3.008,    0.93203163  
  sample_1.wav, 1530.0,   3.008,    0.88415325  
  sample_1.wav, 1551.0,   3.008,    0.92117333  
  sample_1.wav, 1713.0,   3.008,    0.997532  
  sample_1.wav, 1767.0,   3.008,    0.9873333   
  sample_1.wav, 1776.0,   3.008,    0.9882101  
  sample_1.wav, 1797.0,   3.008,    0.81773585  
  sample_1.wav, 1800.0,   3.008,    1.0  
  sample_2.wav, 66.0,     3.008,    0.98680866  
  sample_2.wav, 687.0,    3.008,    0.9871126  
  sample_2.wav, 756.0,    3.008,    0.832537  
  sample_2.wav, 768.0,    3.008,    0.97378933  
  sample_2.wav, 1347.0,   3.008,    0.7106569  
  sample_2.wav, 1800.0,   3.008,    1.0  
  sample_3.wav, 1056.0,   3.008,    0.8226909  
  sample_3.wav, 1290.0,   3.008,    0.77391714  
  sample_3.wav, 1377.0,   3.008,    0.877185  
  sample_3.wav, 1428.0,   3.008,    0.80043906  
  sample_3.wav, 1674.0,   3.008,    0.7289632  
  sample_3.wav, 1800.0,   3.008,    1.0  

The output reports which 3 seconds segments received a score higher than the chosen threshold. 


In the next example, we will use some of the extra options available in the detector program.
If we don't want to process all the files in the audio folder, we can specify which files to run in a .txt file:


.. code-block::

  filename
  sample_1.wav
  sample_3.wav
 
In this example, only sample_1.wav and sample_3.wav will be processed.
Different from the first example, let's use some overlapping. the --step_size argument dets the interval with which the sliding window moves.
With a value of 0.5, each frame will start 0.5 seconds after the previous, so segments go from 0 to 3 seconds, 0.5 to 3.5 seconds, 1.0 to 4.0 seconds, and so on.

This, of course, will result in a lot more spectrograms that will be classified by the network, but it will increase the chances that any upcall will be contained in at least one frame.
If we simply output the score for each frame as we did before, we will probably get many duplicates, as the same upcall now has a high chance of being capture by multiple frames.
By passing the --with_group  and --merge flags when calling the detector, we will group detections in subsequent frames into detection events.
Since each detection event is comprised by one or more detections, the score reported is the moving average (the average's window size is defined by the --win_len argument)
Note that the threshold is applied after the moving average, which will likely lower the score values for the regions with NARW presence, but will also help to reduce false positives.

Check the tutorial and documentation for more details. 

.. code-block::

  python detector.py --model=narw.kt --audio_folder=audio --input_list=input_list.txt --threshold=0.5 --output=detections_with_overlap.csv --win_len=5 --buffer=1 --with_group --step_size=0.5  --with_merge

The output looks like this:

.. code-block::

  filename,     start,    duration, score
  sample_1.wav, 1037.5,   3.5,      0.7238730311393738  
  sample_1.wav, 1124.0,   6.0,      0.8661251336336135  
  sample_1.wav, 1150.5,   4.5,      0.8058994941413402  
  sample_1.wav, 1192.0,   5.5,      0.8895677924156188  
  sample_1.wav, 1200.0,   2.5,      0.727120554447174  
  sample_1.wav, 1208.0,   3.0,      0.7356999576091766  
  sample_1.wav, 1224.0,   2.5,      0.7138631701469421  
  sample_1.wav, 1354.5,   5.5,      0.8540670709950583  
  sample_1.wav, 1433.5,   5.0,      0.8502906531095503  
  sample_1.wav, 1485.5,   5.0,      0.8429508785406749  
  sample_1.wav, 1508.0,   5.0,      0.8350321878989537  
  sample_1.wav, 1527.0,   5.0,      0.7932462533315023  
  sample_1.wav, 1549.5,   3.5,      0.7231567819913228  
  sample_1.wav, 1710.0,   5.5,      0.8813733347824642  
  sample_1.wav, 1764.5,   5.0,      0.839470941821734  
  sample_1.wav, 1774.0,   5.0,      0.8613918056090673  
  sample_3.wav, 1054.5,   2.5,      0.7224384665489196  
  sample_3.wav, 1375.5,   4.0,      0.7978437319397926  
  sample_3.wav, 1425.5,   3.5,      0.7652663509051004  
  sample_3.wav, 1674.0,   2.5,      0.7496926188468933  

