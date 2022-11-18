Change log
==========

**Version 2.6.3** ()

 * Updated supported Python version up to 3.10.8

**Version 2.6.2** (Sept 21, 2022)

 * :meth:`MagSpectrogram:plot <ketos.audio.spectrogram.MagSpectrogram.plot>` and :meth:`PowerSpectrogram:plot <ketos.audio.spectrogram.PowerSpectrogram.plot>` will no longer display a label dB when `decibel`` is set to False.
 * `pad_mode="reflect"` is now correctly set as the default instead of `zero`.
 * The `pad_mode` argument is now correctly propagated from the spectrograms methods.
 * The code will now give a better more descriptive warning when this pad process occurs. 
 * pad argument in :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` now works as expected when set to False.
 * New method :meth:`ceil <ketos.utils.ceil>`. Allows to ceil to a given decimal instead of to the next integer
 * New method :meth:`floor <ketos.utils.floor>`. Allows to floor to a given decimal instead of to the previous integer
 * New method :meth:`ceil_round_down <ketos.utils.ceil_round_down>`. This method will floor instead of ceil when the number is very close to the previous integer. Useful when dealing with floating point arithmetics.
 * New method :meth:`floor_round_up <ketos.utils.floor_round_up>`. This method will ceil instead of floor when the number is very close to the next integer. Useful when dealing with floating point arithmetics.
 * :meth:`select <ketos.data_handling.selection_table.select>` will now check for inconsistent start and end times and give appropriate warnings. This fixes a previous infinite loop when the start time was greater than the end time.
 * Fixed bug in :meth:`standardize <ketos.data_handling.selection_table.standardize>` where setting `trim_table=True` and `mapper=None` would give an error.
 * Removed class :class:`AudioSelectionLoader <ketos.audio.audio_loader.AudioSelectionLoader>` in favor of using more primitive classes. See :class:`AudioLoader <ketos.audio.audio_loader.AudioLoader>` for examples on how to achieve the same functionality.
 * Added option for user to configure formatting of annotation boxes in :meth:`MagSpectrogram:plot <ketos.audio.spectrogram.MagSpectrogram.plot>` method
 * Fixed a minor bug in spectrogram crop method (the frequency axis of the original instance was being modified when calling crop with make_copy=True)
 * Added :class:`ArchiveManager <ketos.audio.audio_loader.ArchiveManager>` allowing for audio files to be loaded from within tar archive files using the :class:`AudioLoader <ketos.audio.audio_loader.AudioLoader>` class. 
 * Class :class:`AudioLoader <ketos.audio.audio_loader.AudioLoader>` replaces `repres` arguments with two new arguments `representation` and `representation_params`. This allows custom audio representation to be passed. Reference,
   the documentation of :class:`AudioLoader <ketos.audio.audio_loader.AudioLoader>` and :meth:`load_audio_representation <ketos.data_handling.parsing.load_audio_representation>` on ways of using a custom audio representation.
 * Classes :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` and :class:`AudioEfficientFrameLoader <ketos.audio.audio_loader.AudioEfficientFrameLoader>` followed the same changes made to the AudioLoader class.
 * Method :meth:`create_database <ketos.data_feeding.database_interface.create_database>` now requires a dict to be passed with an audio_repres containing a class rather than simply a string with the name of the class.
 * Changes made to :meth:`encode_parameter <ketos.data_handling.parsing.encode_parameter>` and :meth:`parse_audio_representation <ketos.data_handling.parsing.parse_audio_representation>`. These functions will now properly
   parse a json string or unparsed dict with strings into a python dict with classes and correct types and vice-versa.


**Version 2.6.1** (June 3, 2022)

 * In :meth:`AudioWriter:close <ketos.data_handling.database_interface.AudioWriter.close>` remove any existing column indices before attempting to reindex them. (Only relevant if appending to an existing database.)
 * Internally replaced pandas append method with pandas concat due to deprecation.
 * Several optimizations for the :meth:`create_rndm_selections <ketos.data_handling.selection_table.create_rndm_selections>` function.


**Version 2.6.0** (May 6, 2022)

Note: This version includes some non-backward compatible changes, as detailed below.

Non-backward compatible changes:

 * :class:`AudioSelectionLoader <ketos.audio.audio_loader.AudioSelectionLoader>` ignores the duration parameter (if present) in the audio representation dictionary. Furthermore, 
   the duration argument has been removed from the constructor, which previously allowed standardization of selection window sizes across the selection table. This must now be done by the user before passing the selection table to the audio loader.
 * Changes to the interface and default behaviour of the :meth:`standardize <ketos.data_handling.selection_table.standardize>` method.
 * Changed default behaviour of :meth:`num_samples <ketos.audio.utils.misc.pad_reflect>` to *not* invert the signal
 * :meth:`append <ketos.audio.waveform.Waveform.append>` preserves duration
 * AudioLoader stores annotation start time and end time without truncation.

Backward compatible changes:

 * Fixed bug in :class:`BatchGenerator <ketos.data_handling.data_feeding.BatchGenerator>` related to loading of labels from separate annotation table.
 * New :class:`MultiModalBatchGen <ketos.data_handling.data_feeding.MultiModalBatchGen>` class.
 * :meth:`create_database <ketos.data_handling.database_interface.create_database>` allows multiple audio representations to be specified as a nested dictionary, using keys as names for the data fields in the output table.
 * :class:`AudioSelectionLoader <ketos.audio.audio_loader.AudioSelectionLoader>` generalized to be able to handle new format for selection tables where a single selection may span several files.
 * Extended Python compatibility to <=3.9.12
 * Modified (and simplified) :meth:`load_audio_for_spec <ketos.audio.spectrogram.load_audio_for_spec>` so that it re-uses :meth:`from_wav <ketos.audio.waveform.Waveform.from_wav>`.
 * Added `pad_mode` argument to :meth:`from_wav <ketos.audio.waveform.Waveform.from_wav>` class method.
 * Minor bug fix in :meth:`num_samples <ketos.audio.utils.misc.num_samples>` for rounding time intervals to integer number of samples.
 * New method :meth:`append <ketos.audio.waveform.merge>` for stitching together multiple waveforms
 * `from_wav`` methods can load from multiple audio files
 * New option in :meth:`export_to_ketos <ketos.neural_networks.dev_utils.export.export_to_ketos>` to save extra files to the output .kt archive.
 * Added option to specify output transform function in :class:`JointBatchGen <ketos.data_handling.data_feeding.JointBatchGen>`.
 * New Class :class:`NNArch <ketos.neural_networks.nn_interface.NNArch>` that serves as the parent for specific architecture classes.
 * Methods for NNArch
 * :meth:`add_frontend <ketos.neural_networks.nn_interface.NNArch.add_frontend>` allows users to add a frontend block that will always be executed first during training
 * :meth:`call_frontend <ketos.neural_networks.nn_interface.NNArch.call_frontend>` to be used in the call method of a new NN architecture. Will call the frontend blocks if they exist 

**Version 2.5.1** (March 10, 2022)

 * Minor bug fix in :meth:`encode_parameter <ketos.data_handling.parsing.encode_parameter>`.
 * Generalized :class:`AudioSelectionLoader <ketos.audio.audio_loader.AudioSelectionLoader>` to accept audio representations with different durations.
 * :meth:`select <ketos.data_handling.selection_table.select>` allows user to specify unwanted labels and to discard selections that extend outside the file.
 * :meth:`create_database <ketos.data_handling.database_interface.create_database>` converts the annotations table to use best possible dtypes to avoid mixed types
 * :meth:`file_duration_table <ketos.data_handling.selection_table.file_duration_table>` can parse datetime information from filename.
 * datetime attributes can be saved to databases  
 * :meth:`standardize <ketos.data_handling.selection_table.standardize>` can parse datetime field.
 * :meth:`standardize <ketos.data_handling.selection_table.standardize>` converts paths to the format appropriate for the operating system that is being used.
 * New pad argument in :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` and minor bug fix.
 * Label mapping now also implemented in :class:`JointBatchGen <ketos.data_handling.data_feeding.JointBatchGen>`. Enabled by Default.
 * :meth:`create_database <ketos.data_handling.database_interface.create_database>` stores list of unique labels in table attribute 
 * Option to save metadata added to :meth:`export_to_ketos <ketos.neural_networks.dev_utils.export.export_to_ketos>` interface
 * Bug fix in :meth:`write <ketos.data_handling.database_interface.write>` (check that annotation object is not None before attempting to save)


**Version 2.5.0** (March 1, 2022)

 * New method :meth:`plot_phase_angle <ketos.audio.spectrogram.MagSpectrogram.plot_phase_angle>` for plotting complex phase angle.
 * New method :meth:`plot <ketos.audio.waveform.plot>` for plotting multiple waveforms superimposed on one another.
 * Modified :meth:`resample <ketos.audio.waveform.Waveform.resample>` to use librosa's resampling function.
 * Bug fix in :meth:`recover_waveform <ketos.audio.spectrogram.MagSpectrogram.recover_waveform>` (ensure that output waveform as correct length).
 * Bug fix in parsing module (ensure that audio representations are encoded correctly when passed as a nested dict).
 * Bug fix in :meth:`train_loop <ketos.neural_networks.dev_utils.nn_interface.train_loop>` (ensure `last_epoch_with_improvement` attr is always initialized)
 * Bug fix in :meth:`export_to_ketos <ketos.neural_networks.dev_utils.export.export_to_ketos>`.
 * Overall, we are changing the way we handle backgr_labels. Previously, we reserved label 0 for backgr. This will no longer be the case going forward.
   backgr labels will be treated as any other:
 * Added new argument to the BatchGenerator class called map_labels with a default value of True. When true, will map labels to incremental integers
   0,1,2,3...
 * selection_table.create_label_dict is now an internal method
 * selection_table.standardize function deprecated signal_labels and backgr_labels in favor of the new labels argument which combine the functionality
   of both
 * selection_table.select new argument background_label. Default value of 0. The value assigned to the background labels.
 * selection_table.create_rndm_backgr_selections is now deprecated in favor of create_rndm_selections.
 * new method selection_table.create_rndm_selections. Creates ramdnom selection with a label that must be defined by the user. Similar functionality 
   to create_rndm_backgr_selections function
 * Added new argument to selection_table.select_by_segmenting called label_empty with default value of 0. Only relevant if keep_only_empty=True.
   Value is assigned to selections without annotations
 * Bug fix in :meth:`segment_files <ketos.data_handling.selection_table.segment_files>`.
 * New method :meth:`skip <ketos.audio.audio_loader.AudioLoader.skip>`.
 * New argument `discard_outside` added to :meth:`create_database <ketos.data_handling.database_interface.create_database>`
 * Corrected error in stepping algorithm for creating selections with :meth:`select <ketos.data_handling.selection_table.select>`.
 * Created short-hand versions of the methods `save_model` (`save`), `load_model_file` (`load`), and `build_from_recipe_file` (`build`) in the `nn_interface` module. The long-hand versions are retained for backward compatibility.
 * New method `is_encoded` in parsing module
 * New method `export_to_ketos` in export module
 * New method `infer_shape` in the audio module


**Version 2.4.3** (February 11, 2022)

 * Fixed bug in :func:`ensure_dir <ketos.utils.ensure_dir>`


**Version 2.4.2** (February 11, 2022)

 * Extended Tensorflow compatibility to <=2.8
 * Extended Python compatibility to <=3.9.10
 * Disabled automatic import of data_handling, audio, and neural_networks submodules.
 * Output directories will be automatically created if they do not already exist when calling 
   :meth:`create_database <ketos.data_handling.database_interface.create_database>` and
   :meth:`save_model <ketos.neural_networks.dev_utils.nn_interface.save_model>`. 
 * For each processed selection the :meth:`create_database <ketos.data_handling.database_interface.create_database>` function checks 
   if the selection start and end times are within the audio file duration and issues a warning if this is not the case. (Requires 
   verbose to be set to True, which is the default setting.)   
 * Extended Tensorflow compatibility to include versions 2.6 and 2.7. (Note: If you are using Tensorflow 2.6, make sure that your Keras 
   version is also 2.6. Normally, when you install Tensorflow with pip, the correct Keras version will automatically be installed, but 
   specifically for Tensorflow 2.6, pip will wrongly install Keras 2.7 causing a mis-match between the two packages.) 
 * In :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` and :class:`FrameStepper <ketos.audio.audio_loader.FrameStepper>` the 
   `frame` argument has been renamed to `duration` for consistency with the rest of ketos. The `frame` argument is still there for backward 
   compatibility. 

**Version 2.4.1** (December 18, 2021)

 * Improved structure and rewamp style of html docs
 * Generalized detection module to handle multiple threshold values

**Version 2.4.0** (November 24, 2021)

 * :meth:`get <ketos.audio.annotation.AnnotationHandler.get>` method in :class:`AnnotationHandler <ketos.audio.annotation.AnnotationHandler>` class returns auxiliary columns
 * Reduced the size of large files in the tests/assets folder
 * Fixed broken search functionality in docs page
 * Added option to specify write mode (append/overwrite) in :meth:`create_database <ketos.data_handling.database_interface.create_database>` function
 * New methods for getting file paths and file durations in the :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` and 
   :class:`FrameStepper <ketos.audio.audio_loader.FrameStepper>` classes
 * Fixed bug in the :meth:`group_detections <ketos.neural_networks.dev_utils.detection.group_detections>` function that was causing single-sample 
   detections to be dropped if they occurred at the end of a batch.
 * Added `merge` argument to the :meth:`process <ketos.neural_networks.dev_utils.detection.process>` functions. With merge=True, the 
   :meth:`merge_overlapping_detections <ketos.neural_networks.dev_utils.detection.merge_overlapping_detections>` function is applied to the detections 
   before they are returned. The default value is `merge=False` to ensure backward-compatability.
 * New :meth:`aggregate_duration <ketos.data_handling.selection_table.aggregate_duration>` function for computing the aggregate duration of annotations
 * Improved the implementation of :class:`AudioFrameLoader <ketos.audio.audio_loader.AudioFrameLoader>` to ensure that transforms are applied to frames 
   on a individual basis when frames are loaded in batches.
 * New export module for exporting Ketos models to various formats such as protobuf

**Version 2.3.0** (October 13, 2021)

 * Added exception handling to the :meth:`create_database <ketos.data_handling.database_interface.create_database>` function
 * Added :meth:`get_selection <ketos.audio.data_loader.SelectionTableIterator.get_selection>` function
 * Fixed bug in computation of Mel spectrogram
 * Added :class:`MelAxis <ketos.audio.utils.axis.Axis>` class to handle frequency axis of Mel spectrograms
 * Improved implementation and interface of the ticks_and_labels :meth:`select <ketos.audio.utils.axis.Axis.ticks_and_labels>` method
 * Added :meth:`resize <ketos.audio.spectrogram.Spectrogram.resize>` function in Spectrogram class
 * Added option to select between linear and log (decibel) scale for MagSpectrogram and PowerSpectrogram at creation time

**Version 2.2.0** (June 24, 2021)

 * sort_by_filename_start argument added to :meth:`standardize <ketos.data_handling.selection_table.standardize>` method.
 * The create_database function can now include extra columns specified in the selection tables into the database. 
 * The reduce_tonal_noise function had a bug  that caused the desired method (median or running mean) not to be recognized sometimes. This has now been fixed.
 * The group_detections function had a bug that caused detections at the end of a batch to be dropped. This has been fixed.
 * bandbass_filter method in Waveform class.
 * Option in selection_table::create_rndm_backgr_selections to specify the minimum separation between the background selections and the annotated segments.
 * New module `gammatone` which contains the classes :class:`GammatoneFilterBank <ketos.audio.gammatone.GammatoneFilterBank>` and :class:`AuralFeatures <ketos.audio.gammatone.AuralFeatures>`
 * When creating a MagSpectrogram object, the user can now use the compute_phase argument to specify that the complex phase angle should be computed and stored along with the magnitude spectrogram.
 * Cleaning the duplicate run_on_test_generator method in the NNInterface class
 * Option to return a dictionary with metrics when calling the run_on_test_generator method
 * Assertion to verify that the checkpoint_freq does not exceed the number of epochs.
 * Assertion in the NNInterface.save_model() method, which raise and error if no checkpoints are found
 * Method set_batch_norm_momentum in ResNetArch for modifying the momentum parameter of the batch normalization layers in the network.
 * Method set_dropout_rate in ResNetArch for modifying the dropout rate parameter of the dropout layers in the network. Equivalent methods in ResNetBlock. Possibility to specify the above parameters at initialization
 * Added training=training in all calls to the dropout layers
 * Option to build indices for user-specified columns in the AudioWriter class and the create_database method.
 * Option to have JointBatchGen return indices, in addition to X and Y.
 * In the :meth:`select <ketos.data_handling.selection_table.select>` method, the user can now specify which labels to generate selections for.
 * In the :meth:`select_by_segmenting <ketos.data_handling.selection_table.select_by_segmenting>` method, I have added an extra boolean argument called keep_only_empty, which is useful for generating background samples.
 * A new method called random_choice() that selects a random subset of an annotation/selection table.
 * strides and kernel_size exposed in the ResNet and ResNet1D interfaces
 * Option to include extra attributes present in selection tables in the HDF5 database produced by the AudioWriter and create_database method.
 * Minor bug fix in reduce_tonal_noise method in the Spectrogram class.

**Version 2.1.3** (february 17, 2021)

 * Add features that allow database_interface and audio_loader modules to handle multiple audio representations (i.e. for the same audio clip, multiple representations are generated).
 * Add features to reproduce audio transforms (e.g. normalization, tonal noise removal, etc) from configurations recorded in 'audio representations' (as dictionaries or .json files).

**Version 2.1.2** (february 01, 2021)

 * Fix bug in the detection.py module. When transitioning from a file to another and a detection occurred at the very beginning of the next file, the group_detections function was not working properly, resulting in an error.

**Version 2.1.1** (january 05, 2021)

 * Fix bug in the inception.py module. It had a tensorflow-addons import, but since that's no longer installed with ketos a dependency error could be thrown when importing inception.py.
 * Update the 'train a narw classifier' tutorial to save the audio specifications with the model, as this is expected in the following tutorial

**Version 2.1.0** (November 3, 2020)

 *  New neural network architectures: densenet, inception, resnet-1D, cnn-1D
 *  Early stopping: All neural network interfaces can now use an early stopping monitor, to halt training if a condition is met.
 *  Learning rate scheduler: All neural network interfaces can now use a scheduler through the 'add_learning_rate_scheduler' method.
    Availeble schedulers include 'PiecewiseConstantDecay', 'ExponentialDecay', 'InverseTimeDecay' and 'PolynomialDecay'
 *  General load model function: a load_model_file function was added to the ketos.neural_networks namespace, which can rebuild a 
    model from a .kt file without the user having to know which architecture the model has. Before, you had to know which interface 
    to use (i.e.: which kind of network that was). In order for this to work, all model architectures add a field 'interface' to the 
    recipes. If a recipe does not have this field (e.g.: from a model created with an older ketos version), an exception will be raised. 
    All models can still be loaded as before, through their interface classes.
 *  Detection module: A new module ketos.neural_networks.dev_utils.detection was created to aid developers who want to use snapshot 
    classifiers as detectors in longer files. A tutorial was also added to the docs.
 *  tensorflow version requirement changed to >=2.2

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

 * New method for generating CQT spectrograms directly from audio file (.wav) input.
 * Spectrogram plot method provides correct labels for CQT spectrogram.
 * If necessary, maximum frequency of CQT spectrogram is automatically reduced to ensure that it is below the Nyquist frequency. 
 * Minor bug fix in _crop_image method in Spectrogram class


**Version 1.0.7** (July 23, 2019)

 * from_wav method in MagSpectrogram class raises an exception if the duration 
   does not equal an integer number of steps.


**Version 1.0.6** (July 23, 2019)

 * New method for generating magnitude spectrograms directly from audio file (.wav) input.


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

