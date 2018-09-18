""" Unit tests for the the 'data_handling' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Institute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""

import pytest
import numpy as np
import pandas as pd
import sound_classification.data_handling as dh
import sound_classification.pre_processing as pp
from sound_classification.spectrogram import MagSpectrogram
import shutil
import os
from glob import glob

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.mark.test_encode_database
def test_encode_database_with_one_image_and_one_label(datebase_with_one_image_col_and_one_label_col):
    db = datebase_with_one_image_col_and_one_label_col
    dh.encode_database(db, "image", "label")
    
@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_names_do_not_match(datebase_with_one_image_col_and_one_label_col):
    db = datebase_with_one_image_col_and_one_label_col
    with pytest.raises(AssertionError):
        dh.encode_database(db, "kangaroo", "label")

@pytest.mark.test_encode_database
def test_encode_database_throws_exception_if_database_does_not_have_a_label_column(datebase_with_one_image_col_and_no_label_col):
    db = datebase_with_one_image_col_and_no_label_col
    with pytest.raises(AssertionError):
        dh.encode_database(db, "image", "label")

@pytest.mark.test_encode_database
def test_encode_database_can_handle_inputs_with_multiple_columns(datebase_with_two_image_cols_and_one_label_col):
    db = datebase_with_two_image_cols_and_one_label_col
    dh.encode_database(db, "image1", "label")

@pytest.mark.test_split_database
def test_split_database_throws_exception_unless_all_three_keys_are_given(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    encoded, img_size = dh.encode_database(raw, "image", "label") 
    divisions = {"train":(0,100),"validation":(0,100)}
    with pytest.raises(AssertionError):
        split = dh.split_database(encoded, divisions)
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    split = dh.split_database(encoded, divisions)

@pytest.mark.test_stack_dataset
def test_stack_dataset_throws_exception_if_column_names_do_not_match(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    with pytest.raises(AssertionError):
        dh.stack_dataset(raw,(128,128))

@pytest.mark.test_stack_dataset
def test_stack_dataset_automatically_determines_image_size(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    encoded, img_size = dh.encode_database(raw, "image", "label")
    stacked = dh.stack_dataset(encoded, img_size)   

@pytest.mark.test_prepare_database
def test_prepare_database_executes(datebase_with_one_image_col_and_one_label_col):
    raw = datebase_with_one_image_col_and_one_label_col
    divisions = {"train":(0,100),"validation":(0,100),"test":(0,100)}
    dh.prepare_database(raw, "image", "label", divisions) 


@pytest.mark.test_def_slice_ffmpeg
def test_sliced_audio_file_has_correct_properties(sine_wave_file):
    prefix="halifax123456789"
    out_name = path_to_assets + "/" + prefix + ".wav"
    dh.slice_ffmpeg(sine_wave_file, 0.0, 1.7, out_name)
    rate_orig, sig_orig = pp.wave.read(sine_wave_file)
    rate, sig = pp.wave.read(out_name)
    duration = len(sig) / rate
    assert rate == rate_orig
    assert duration == 1.7
#    for i in range(len(sig)):
#        assert sig[i] == sig_orig[i]
    # clean
    for f in glob(path_to_assets + "/*" + prefix + "*"):
        os.remove(f)

@pytest.mark.parametrize("input,depth,expected",[
    (1,2,np.array([0,1])),
    (0,2,np.array([1,0])),
    (1.0,2,np.array([0,1])),
    (0.0,2,np.array([1,0])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_with_floats_and_ints(input, depth, expected):
    one_hot = dh.to1hot(input, depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (1,4,np.array([0,1,0,0])),
    (1,4,np.array([0,1,0,0])),
    (1,2,np.array([0,1])),
    (1,10,np.array([0,1,0,0,0,0,0,0,0,0])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_output_has_correct_depth(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert len(one_hot) == depth


@pytest.mark.parametrize("input,depth,expected",[
    (3,4,np.array([0,0,0,1])),
    (0,4,np.array([1,0,0,0])),
    (1.0,2,np.array([0,1])),
    (5.0,10,np.array([0,0,0,0,0,1,0,0,0,0])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_with_multiple_categories(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (np.array([3,0,1,5]),6,
     np.array([[0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1.]])),
    (np.array([0,1]),3,
     np.array([[1., 0., 0.],
               [0., 1., 0.]])),
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_with_multiple_input_values_at_once(input,depth, expected):
    one_hot = dh.to1hot(input,depth)
    assert (one_hot == expected).all()


@pytest.mark.parametrize("input,depth,expected",[
    (pd.DataFrame({"label":[0,0,1,0,1,0]}),2,
     np.array([[1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0]]),)
    ])
@pytest.mark.test_to1hot
def test_to1hot_works_when_when_applying_to_DataFrame(input,depth, expected):
     
    one_hot = input["label"].apply(dh.to1hot,depth=depth)
    for i in range(len(one_hot)):
        assert (one_hot[i] == expected[i]).all()

@pytest.mark.test_get_wave_files
def test_get_wave_files():
      
    dir = os.path.join(path_to_assets,'test_get_wave_files')
    dh.create_dir(dir)

    # create two wave files
    f1 = os.path.join(dir, "f1.wav")
    f2 = os.path.join(dir, "f2.wav")
    pp.wave.write(f2, rate=100, data=np.array([1.,0.]))
    pp.wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.get_wave_files(dir, fullpath=False)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    
    #delete directory and files within
    shutil.rmtree(dir)
    
################################
# from1hot() tests
################################


@pytest.mark.parametrize("input,expected",[
    (np.array([0,1]),1),
    (np.array([1,0]),0),
    (np.array([0.0,1.0]),1),
    (np.array([1.0,0.0]),0),
    ])
@pytest.mark.test_from1hot
def test_from1hot_works_with_floats_and_ints(input, expected):
    one_hot = dh.from1hot(input)
    assert one_hot == expected


@pytest.mark.parametrize("input,expected",[
    (np.array([0,0,0,1]),3),
    (np.array([1,0,0,0]),0),
    (np.array([0,1]),1),
    (np.array([0,0,0,0,0,1,0,0,0,0]),5),
    ])
@pytest.mark.test_from1hot
def test_from1hot_works_with_multiple_categories(input, expected):
    one_hot = dh.from1hot(input)
    assert one_hot == expected


@pytest.mark.parametrize("input,expected",[
    (np.array([[0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1.]]),np.array([3,0,1,5])),
    (np.array([[1., 0., 0.],
               [0., 1., 0.]]), np.array([0,1])),
    ])
@pytest.mark.test_from1hot
def test_from1hot_works_with_multiple_input_values_at_once(input, expected):
    one_hot = dh.from1hot(input)
    assert (one_hot == expected).all()

@pytest.mark.test_read_wave
def test_read_wave_file(sine_wave_file):
    rate, data = dh.read_wave(sine_wave_file)
    assert rate == 44100

@pytest.mark.test_parse_datetime
def test_parse_datetime_with_urban_sharks_format():
    fname = 'empty_HMS_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    pp.wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(fname=fname, fmt=fmt)
    os.remove(full_path)
    assert dt is not None
    assert dt.year == 2084
    assert dt.month == 2
    assert dt.day == 23
    assert dt.hour == 12
    assert dt.minute == 5
    assert dt.second == 28

@pytest.mark.test_parse_datetime
def test_parse_datetime_with_non_matching_format():
    fname = 'empty_HMQ_12_ 5_28__DMY_23_ 2_84.wav'
    full_path = os.path.join(path_to_assets, fname)
    pp.wave.write(full_path, rate=1000, data=np.array([0.]))
    fmt = '*HMS_%H_%M_%S__DMY_%d_%m_%y*'
    dt = dh.parse_datetime(fname=fname, fmt=fmt)
    os.remove(full_path)
    assert dt == None



@pytest.mark.create_raw_signal_table_description
def test_create_raw_signal_table_description():
    description = dh.create_raw_signal_table_description(signal_rate=2000, segment_length=2.5)
    description_columns = list(description.columns.keys())
    assert description_columns ==  ['id', 'labels', 'signal', 'boxes']
    assert description.columns['signal'].shape == (5000,)

@pytest.mark.create_image_table_description
def test_create_image_table_description():
    description = dh.create_image_table_description(dimensions=(20,64))
    description_columns = list(description.columns.keys())
    assert description_columns ==  ['id', 'labels', 'signal', 'boxes']
    assert description.columns['signal'].shape == (20, 64)

@pytest.mark.get_data_from_seg_name
def test_get_data_from_seg_name():
    id,labels = dh.get_data_from_seg_name('id_rb001_89_l_[0].wav')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.get_data_from_seg_name('id_rb001_89_l_[0]')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.get_data_from_seg_name('id_rb001_89_l_[1,2].wav')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 

    id,labels = dh.get_data_from_seg_name('id_rb001_89_l_[1,2]')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 

@pytest.mark.test_open_or_create_table
def test_open_or_create_tables():
    
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')

    raw_description = dh.create_raw_signal_table_description(signal_rate=2000, segment_length=2.0)
    spec_description = dh.create_image_table_description(dimensions=(20,60))

    table_1 = dh.open_or_create_table(h5, '/group_1', 'table_1',raw_description, sample_rate=2000)
    assert '/group_1' in h5
    assert '/group_1/table_1' in h5

    table_2 = dh.open_or_create_table(h5, '/group_2', 'table_1',spec_description, sample_rate=2000)
    assert '/group_2' in h5
    assert '/group_2/table_1' in h5

    table_3 = dh.open_or_create_table(h5, '/group_2/subgroup_1', 'table_1',spec_description, sample_rate=2000)
    assert '/group_2/subgroup_1' in h5
    assert '/group_2/subgroup_1/table_1' in h5

    table_4 = dh.open_or_create_table(h5, '/group_3/subgroup_1', 'table_1',spec_description, sample_rate=2000)
    assert '/group_3/subgroup_1' in h5
    assert '/group_3/subgroup_1/table_1' in h5


    #When information about an existing table is given, it should return the table and not create a new one
    existing_table = dh.open_or_create_table(h5, '/group_2', 'table_1',spec_description, sample_rate=2000 )

    assert existing_table == table_2
    
    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))


@pytest.mark.test_write_sig_to_h5_database
def test_write_sig_to_h5_database(sine_wave):
    
    rate, sig = sine_wave
    pp.wave.write(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"),rate, sig)    
    
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')

    raw_description = dh.create_raw_signal_table_description(signal_rate=44100, segment_length=3.0)
    spec_description = dh.create_image_table_description(dimensions=(20,60))

    table_1 = dh.open_or_create_table(h5, '/group_1', 'table_1',raw_description, sample_rate=44100)
    
    dh.write_sig_to_h5_database(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"), table_1)
    table_1.flush()

    pytest.approx(table_1[0]['signal'], sig)
    assert table_1[0]['id'].decode() == 'ex789_107'
    assert table_1[0]['labels'].decode() == '[1]'

    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))
    os.remove(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"))

@pytest.mark.test_write_spetrogram_to_h5_database
def test_write_spectrogram_to_h5_database(sine_audio):
    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "id_ex789_107_l_[1]"
        
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')
    spec_description = dh.create_image_table_description(dimensions=(26, 11026))
    table_1 = dh.open_or_create_table(h5, '/group_1', 'table_1',spec_description, sample_rate=44100)
    
    dh.write_spectrogram_to_h5_database(spec, table_1)
    table_1.flush()

    assert pytest.approx(table_1[0]['signal'],spec.image)
    assert table_1[0]['id'].decode() == 'ex789_107'
    assert table_1[0]['labels'].decode() == '[1]'


    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))
    

    
 

@pytest.mark.test_divide_audio_into_segments
def test_creates_correct_number_of_segments():
    audio_file = path_to_assets+ "/2min.wav"
    annotations = pd.DataFrame({'orig_file':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.34, 105.8],
                                 'end':[6.0,75.98,110.0]})

    try:
        shutil.rmtree(path_to_assets + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=annotations, save_to=path_to_assets + "/2s_segs")
    
    n_seg = len(glob(path_to_assets + "/2s_segs/id_2min*.wav"))
    assert n_seg == 60



    shutil.rmtree(path_to_assets + "/2s_segs")


@pytest.mark.test_divide_audio_into_segments
def test_seg_labels_are_correct():
    audio_file = path_to_assets+ "/2min.wav"
    annotations = pd.DataFrame({'orig_file':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.5, 105.0],
                                 'end':[6.0,73.0,108.0]})

    try:
        shutil.rmtree(path_to_assets + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=annotations, save_to=path_to_assets + "/2s_segs")
    
    label_0 = len(glob(path_to_assets + "/2s_segs/id_2min*l_[[]0].wav"))
    assert label_0 == 53

    label_1 = len(glob(path_to_assets + "/2s_segs/id_2min*l_[[]1].wav"))
    assert label_1 == 5

    label_2 = len(glob(path_to_assets + "/2s_segs/id_2min*l_[[]2].wav"))
    assert label_2 == 2

    shutil.rmtree(path_to_assets + "/2s_segs")





@pytest.mark.test_divide_audio_into_segments
def test_creates_segments_without_annotations():
    audio_file = path_to_assets+ "/2min.wav"
    
    try:
        shutil.rmtree(path_to_assets + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=None, save_to=path_to_assets + "/2s_segs")
    
    n_seg = len(glob(path_to_assets + "/2s_segs/id_2min*l_[[]NULL].wav"))

    assert n_seg == 60
    shutil.rmtree(path_to_assets + "/2s_segs")


@pytest.mark.test_seg_from_time_tag
def test_seg_from_time_tag():

    
    audio_file = os.path.join(path_to_assets, "2min.wav")
    
    try:
        shutil.rmtree(os.path.join(path_to_tmp, "from_tags"))
    except FileNotFoundError:
        pass

    dh.create_dir(os.path.join(path_to_tmp, "from_tags"))
    
    dh.seg_from_time_tag(audio_file=audio_file, start=0.5, end=2.5 , name="seg_1.wav", save_to=os.path.join(path_to_tmp, "from_tags") )

    
    sig, rate  = dh.librosa.load(os.path.join(path_to_tmp, "from_tags", "seg_1.wav"))
    duration = len(sig)/rate
    assert duration == 2.0
    shutil.rmtree(os.path.join(path_to_tmp, "from_tags"))


@pytest.mark.parametrize("start,end,expected_label",[
    (4.0,5.0,'[1]'),
    (4.0,5.5,'[1]'),
    (5.0,6.0,'[1]'),
    (5.1,6.0,'[1]'),
    (100.0,100.5,'[2]'),
    (100.5,101.0,'[2]'),
    (99.0,103.0,'[2]'),
    (90.0,110.0,'[2, 1]'),
     ])
@pytest.mark.test_get_label_from_annotations
def test_get_correct_labels(start,end,expected_label):
    audio_file="2min"
    annotations = pd.DataFrame({'orig_file':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 100.5, 105.0],
                                 'end':[6.0,103.0,108.0]})
    
    label = dh.get_label_from_annotations(file='2min',start=start, end=end,
                             annotations=annotations, not_in_annotations=0)
    print(label)
    assert label == expected_label
    
@pytest.mark.test_filter_annotations_by_orig_file
def test_filter_annotations_by_orig_file():
     annotations = pd.DataFrame({'orig_file':['2min_01.wav','2min_01.wav','2min_02.wav','2min_02.wav','2min_02.wav'],
                                 'label':[1,2,1,1,1], 'start':[5.0, 100.5, 105.0, 80.0, 90.0],
                                 'end':[6.0,103.0,108.0, 87.0, 94.0]})

     annot_01 = dh._filter_annotations_by_orig_file(annotations,'2min_01')
     assert annot_01.equals(pd.DataFrame({'orig_file':['2min_01.wav','2min_01.wav'],
                                 'label':[1,2], 'start':[5.0, 100.5],
                                 'end':[6.0,103.0]}))
                                 
     annot_02 = dh._filter_annotations_by_orig_file(annotations,'2min_02')
     assert annot_02.equals(pd.DataFrame({'orig_file':['2min_02.wav','2min_02.wav','2min_02.wav'],
                                 'label':[1,1,1], 'start':[105.0, 80.0, 90.0],
                                 'end':[108.0, 87.0, 94.0]}, index=[2,3,4]))
 
     annot_03 = dh._filter_annotations_by_orig_file(annotations,'2min_03')               
     assert annot_03.empty
 


