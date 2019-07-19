Change log
==========


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

