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
    #delete directory and files within
    if os.path.exists(dir):
        shutil.rmtree(dir)
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
    files = dh.get_wave_files(dir, fullpath=True)
    assert len(files) == 2
    assert files[0] == f1
    assert files[1] == f2
    #delete directory and files within
    shutil.rmtree(dir)

def test_get_wave_files_from_multiple_folders():
    folder = path_to_assets + "/sub"
    # create two wave files in separate subfolders
    sub1 = folder + "/sub1"
    sub2 = folder + "/sub2"
    if not os.path.exists(sub1):
        os.makedirs(sub1)
    if not os.path.exists(sub2):
        os.makedirs(sub2)
    # clean
    for f in glob(sub1 + "/*.wav"):
        os.remove(f)  #clean
    for f in glob(sub2 + "/*.wav"):
        os.remove(f)  #clean
    f1 = sub1 + "/f1.wav"
    f2 = sub2 + "/f2.wav"
    pp.wave.write(f2, rate=100, data=np.array([1.,0.]))
    pp.wave.write(f1, rate=100, data=np.array([0.,1.]))
    # get file names
    files = dh.get_wave_files(folder, fullpath=False, subdirs=True)
    assert len(files) == 2
    assert files[0] == "f1.wav"
    assert files[1] == "f2.wav"
    files = dh.get_wave_files(folder, fullpath=True, subdirs=True)
    assert len(files) == 2
    assert files[0] == f1
    assert files[1] == f2

    
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



@pytest.mark.audio_table_description
def test_audio_table_description():
    description = dh.audio_table_description(signal_rate=2000, segment_length=2.5)
    description_columns = list(description.columns.keys())
    description_columns.sort()
    assert description_columns ==  ['boxes','id', 'labels', 'signal']
    assert description.columns['signal'].shape == (5000,)

@pytest.mark.spec_table_description
def test_spec_table_description():
    description = dh.spec_table_description(dimensions=(20,64))
    description_columns = list(description.columns.keys())
    description_columns.sort()
    assert description_columns ==  ['boxes','id', 'labels', 'signal']
    assert description.columns['signal'].shape == (20, 64)

@pytest.mark.parse_seg_name
def test_parse_seg_name():
    id,labels = dh.parse_seg_name('id_rb001_89_l_[0].wav')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[0]')
    assert id == 'rb001_89'
    assert labels == '[0]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[1,2].wav')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 

    id,labels = dh.parse_seg_name('id_rb001_89_l_[1,2]')
    assert id == 'rb001_89'
    assert labels == '[1,2]' 

@pytest.mark.test_open_table
def test_open_tables():
    
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')

    raw_description = dh.audio_table_description(signal_rate=2000, segment_length=2.0)
    spec_description = dh.spec_table_description(dimensions=(20,60))

    table_1 = dh.open_table(h5, '/group_1', 'table_1',raw_description, sample_rate=2000)
    assert '/group_1' in h5
    assert '/group_1/table_1' in h5

    table_2 = dh.open_table(h5, '/group_2', 'table_1',spec_description, sample_rate=2000)
    assert '/group_2' in h5
    assert '/group_2/table_1' in h5

    table_3 = dh.open_table(h5, '/group_2/subgroup_1', 'table_1',spec_description, sample_rate=2000)
    assert '/group_2/subgroup_1' in h5
    assert '/group_2/subgroup_1/table_1' in h5

    table_4 = dh.open_table(h5, '/group_3/subgroup_1', 'table_1',spec_description, sample_rate=2000)
    assert '/group_3/subgroup_1' in h5
    assert '/group_3/subgroup_1/table_1' in h5


    #When information about an existing table is given, it should return the table and not create a new one
    existing_table = dh.open_table(h5, '/group_2', 'table_1',spec_description, sample_rate=2000 )

    assert existing_table == table_2
    
    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))


@pytest.mark.test_write_audio_to_table
def test_write_audio_to_table(sine_wave):
    
    rate, sig = sine_wave
    pp.wave.write(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"),rate, sig)    
    
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')

    raw_description = dh.audio_table_description(signal_rate=44100, segment_length=3.0)
    spec_description = dh.spec_table_description(dimensions=(20,60))

    table_1 = dh.open_table(h5, '/group_1', 'table_1',raw_description, sample_rate=44100)
    
    dh.write_audio_to_table(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"), table_1)
    table_1.flush()

    pytest.approx(table_1[0]['signal'], sig)
    assert table_1[0]['id'].decode() == 'ex789_107'
    assert table_1[0]['labels'].decode() == '[1]'

    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))
    os.remove(os.path.join(path_to_tmp,"id_ex789_107_l_[1].wav"))

@pytest.mark.test_write_spec_to_table
def test_write_spec_to_table(sine_audio):
    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "id_ex789_107_l_[1]"
        
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')
    spec_description = dh.spec_table_description(dimensions=(26, 11026))
    table_1 = dh.open_table(h5, '/group_1', 'table_1', spec_description, sample_rate=44100)
    
    dh.write_spec_to_table(table_1, spec)
    table_1.flush()

    assert pytest.approx(table_1[0]['signal'],spec.image)
    assert table_1[0]['id'].decode() == 'ex789_107'
    assert table_1[0]['labels'].decode() == '[1]'

    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))

@pytest.mark.test_write_spec_to_table
def test_write_spec_to_table_with_optional_args(sine_audio):
    
    spec = MagSpectrogram(sine_audio, 0.5, 0.1)
    spec.tag = "id_ex789_107_l_[1]"
        
    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp2_db.h5'), 'w')
    spec_description = dh.spec_table_description(dimensions=(26, 11026))
    table_1 = dh.open_table(h5, '/group_1', 'table_1',spec_description, sample_rate=44100)
    
    dh.write_spec_to_table(table_1, spec, id='id123?$', labels=(0,1,3), boxes=((0.1,0.8,40,400.5),(1.2,2.8,0.77,200.0),(7.7,77,40.0,500.5)))
    table_1.flush()

    assert pytest.approx(table_1[0]['signal'],spec.image)
    assert table_1[0]['id'].decode() == 'id123?$'
    assert table_1[0]['labels'].decode() == '[0,1,3]'
    assert table_1[0]['boxes'].decode() == '[[0.1,0.8,40.0,400.5],[1.2,2.8,0.77,200.0],[7.7,77.0,40.0,500.5]]'

    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp2_db.h5'))

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
def test_start_end_args():
    audio_file = path_to_assets+ "/2min.wav"
    _= pd.DataFrame({'orig_file':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 70.34, 105.8],
                                 'end':[6.0,75.98,110.0]})

    try:
        shutil.rmtree(path_to_assets + "/2s_segs")
    except FileNotFoundError:
        pass

    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, start_seg=10, end_seg=19, save_to=path_to_assets + "/2s_segs")
    
    n_seg = len(glob(path_to_assets + "/2s_segs/id_2min*.wav"))
    assert n_seg == 10



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

    
    rate, sig  = pp.wave.read(os.path.join(path_to_tmp, "from_tags", "seg_1.wav"))
    duration = len(sig)/rate
    assert duration == 2.0
    shutil.rmtree(os.path.join(path_to_tmp, "from_tags"))

@pytest.mark.test_seg_from_annotations
def test_segs_from_annotations():
    audio_file_path = os.path.join(path_to_assets,'2min.wav')
    annotations = pd.DataFrame({'orig_file':[audio_file_path,audio_file_path,audio_file_path],
                                 'label':[1,2,1], 'start':[5.0, 70.5, 105.0],
                                 'end':[6.0,73.0,108.0]})

    try:
        shutil.rmtree(path_to_assets + "/from_annot")
    except FileNotFoundError:
        pass
    dh.segs_from_annotations(annotations,path_to_assets + "/from_annot")
    
    # label_0 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]0].wav"))
    # assert label_0 == 53

    label_1 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]1].wav"))
    assert label_1 == 2

    label_2 = len(glob(path_to_assets + "/from_annot/id_2min*l_[[]2].wav"))
    assert label_2 == 1

    shutil.rmtree(path_to_assets + "/from_annot")
    


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
@pytest.mark.test_get_labels
def test_get_correct_labels(start,end,expected_label):
    audio_file="2min"
    annotations = pd.DataFrame({'orig_file':['2min.wav','2min.wav','2min.wav'],
                                 'label':[1,2,1], 'start':[5.0, 100.5, 105.0],
                                 'end':[6.0,103.0,108.0]})
    
    label = dh.get_labels(file='2min',start=start, end=end,
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
 
@pytest.mark.test_pad_signal
def test_pad_signal():

    sig=np.ones((100))
    rate = 50 
    desired_length = 3.0
    sig_length = len(sig)/rate #sig length in seconds

    padded = dh.pad_signal(signal=sig, rate=rate, length=desired_length)
    

    assert len(padded) == desired_length * rate
    pad_1_limit = int((desired_length - sig_length) * rate / 2)
    pad_2_limit = int(desired_length * rate - pad_1_limit)
    assert sum(padded[:pad_1_limit]) == 0
    assert sum(padded[pad_2_limit:]) == 0
    assert pytest.approx(padded[pad_1_limit:pad_2_limit], sig)

    
@pytest.mark.test_create_spec_table_from_audio_table
def test_create_spec_table_from_audio_table():
    audio_file = path_to_assets+ "/2min.wav"
    
    try:
        shutil.rmtree(path_to_assets + "/2s_segs")
    except FileNotFoundError:
        pass

    dir = path_to_assets + "/2s_segs"
    dh.divide_audio_into_segs(audio_file=audio_file,
        seg_duration=2.0, annotations=None, save_to=dir)

    h5 = dh.tables.open_file(os.path.join(path_to_tmp, 'tmp_db.h5'), 'w')

    raw_description = dh.audio_table_description(signal_rate=2000, segment_length=2.0)
    spec_description = dh.spec_table_description(dimensions=(20,60))

    table_raw = dh.open_table(h5, '/raw', 'seq_2s',raw_description, sample_rate=2000)
    
    segs = os.listdir(dir)
    for seg in segs:
        dh.write_audio_to_table(os.path.join(dir,seg),table_raw, pad=True, duration=2.0)

    table_raw.flush()
    
    dh.create_spec_table_from_audio_table(h5, table_raw, "/features/mag_spectrograms/", "seq_2s", MagSpectrogram, winlen=0.25, winstep=0.05)
    spec_table = h5.root.features.mag_spectrograms.seq_2s

    assert len(spec_table) == len(segs)
    assert pytest.approx(spec_table[:]['id'], table_raw[:]['id'])
    assert pytest.approx(spec_table[:]['labels'], table_raw[:]['labels'])
    
    h5.close()
    os.remove(os.path.join(path_to_tmp, 'tmp_db.h5'))

    shutil.rmtree(path_to_assets + "/2s_segs")