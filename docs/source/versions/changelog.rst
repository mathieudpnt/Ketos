Change log
==========

**Version 2.0.3** (July 12, 2020)

 *  tensorflow version requirement changed to >=2.1, <=2.2.1

**Version 2.0.2** (July 9, 2020)

 *  tensorflow version requirement changed from ==2.1 to >=2.1, <=2.2

**Version 2.0.1** (July 8, 2020)

 * Removes tensorflow-addons dependency. As a consequence, the FScore metric is no longer available to be reported during training by the NNInterface, but Precision and Recall are. The FScoreLoss can still be used. 

 * This merge also fixes a small bug in the run_on_test_generator method.

**Version 2.0.0** (June 26, 2020)

 *  Added convenience method to the NNInterface class for model testing.

**Version 2.0.0 (beta)** (May 7, 2020)

 * Extensive upgrades to all modules!


**Version 1.1.5** (November 20, 2019)

 * Specify tensorflow version 1.12.0 in setup file.


**Version 1.1.4** (November 16, 2019)

 * Added option to specify padding mode for SpecProvider. 
 * Bug fix in SpecProvider: Loop over all segments.


**Version 1.1.3** (November 15, 2019)

 * Added option to specify resampling type in MagSpectrogram.from_wav method 
 * Bug fix in SpecProvider: jump to next file if time exceeds file duration.


**Version 1.1.2** (November 12, 2019)

 * Added option for creating overlapping spectrograms in the create_spec_database method
 * Added option for specifying batch size as an integer number of wav files in AudioSequenceReader
 * Added option for generating spectrograms from a SpectrogramConfiguration object
 * New SpecProvider class facilitates loading and computation of spectrograms from wave files


**Version 1.1.1** (October 2, 2019)

 * Fixed minor bug in spectrogram.get_label_vector method, occuring when annotation box goes beyond spectrogram time range.
 * When annotations are added to a spectrogram with the spectrogram.annotate mehod, any annotation that is fully outside the spectrogram time range is ignored.
 * When spectrograms are saved to a HDF5 database file using the database_interface.write_spec method, the time offset tmin is subtracted from all annotations, since this offset is lost when the spectrogram is saved.
 * from_wav methods in spectrogram module do not merge stereo recordings into mono


**Version 1.1.0** (August 13, 2019)

 * New Jupyter Notebook tutorial demonstrating how to implement a simple boat detection program.
 * AverageFilter added to spectrogram_filters module.


**Version 1.0.9** (August 7, 2019)

 * Fixed minor bug in spectrogram crop method.
 * Updated to latest numpy version (1.17.0), which includes an enhanced Fast-Fourier-Transform (FFT) implementation.


**Version 1.0.8** (July 24, 2019)

 * New method for generating CQT spectrograms directly from audio file (*.wav) input.
 * Spectrogram plot method provides correct labels for CQT spectrogram.
 * If necessary, maximum frequency of CQT spectrogram is automatically reduced to ensure that it is below the Nyquist frequency. 
 * Minor bug fix in _crop_image method in Spectrogram class


**Version 1.0.7** (July 23, 2019)

 * from_wav method in MagSpectrogram class raises an exception if the duration 
   does not equal an integer number of steps.


**Version 1.0.6** (July 23, 2019)

 * New method for generating magnitude spectrograms directly from audio file (*.wav) input.


**Version 1.0.5** (July 19, 2019)

 * BasicCNN accepts multi-channel images as input.


**Version 1.0.4** (June 26, 2019)

 * Option to add batch normalization layers to BasicCNN.
 * BasicCNN can save training and validation accuracy to ascii file during training.
 * BasicCNN class method _check_accuracy splits data into smaller chunks to avoid memory allocation error.
 * make_frames method in audio_processing module issues a warning when the estimated size of the output frames exceeds 10% of system memory.
 * New class method in AudioSignal class splits the audio signal into equal length segments, while also handling annotations
 * check of memory usage added to the create_spec_database method; if too much memory is used, the audio signal is segmented before the spectrogram is computed
 * parsing of file names in the audio_signal module improved to ensure correct behaviour also on Windows
 * An option has been added to enforce same length when extracting annotated segments from a spectrogram. If an annotation is shorter than the specified length, the annotation box is stretched; if it is shorter, the box is divided into several segments.
 * New CQTSpectrogram class in the spectrogram module.
 * data_handling.data_handling.find_wave_files looks not only for files with extension .wav, but also .WAV
 * conversion from byte literal to str in external.wavfile to avoid TypeError
 * Spectrogram class enforces window size to be an even number of bins. If the window size (specified in seconds) corresponds to an odd number of bins, +1 bin is added to the window size.
 * Implementation of new method for estimating audio signal from magnitude spectrogram based on the Griffin-Lim algorithm
 * Option to save output spectrograms from interbreed method to an hdf5 database file. This is useful for generating large synthetic training data sets.
 * Option to reduce tonal noise in connection with interbreed method.
 * Option to select write/append mode in SpecWriter.
 * Minor bug fix in append method in Spectrogram class.
 * Improved implementation of ActiveLearningBatchGenerator; train_active method in BasicCNN modified accordingly.
 * Both BatchGenerator and ActiveLearningBatchGenerator can read either from memory or database.
 * New tutorial showing how to compute spectrograms and save them to a database.


**Version 1.0.3** (June 21, 2019)

* New filters FAVFilter and FAVThresholdFilter added to spectrogram_filters module


**Version 1.0.2** (May 14, 2019)

* create_spec_database method in database_interface module correctly handles parsing of Windows paths


**Version 1.0.1** (April 12, 2019)

* First release

