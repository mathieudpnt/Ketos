# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" Unit tests for the selection_table module within the ketos library
"""
import pytest
import os
import ketos.data_handling.selection_table as st
import pandas as pd
import numpy as np


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_trim():
    standard = ['filename','label','time_start','time_stop','freq_min','freq_max']
    extra = ['A','B','C']
    df = pd.DataFrame(columns=extra)
    df = st.trim(df)
    assert len(df.columns) == 0
    df = pd.DataFrame(columns=standard+extra)
    df = st.trim(df)
    assert sorted(df.columns.values) == sorted(standard)

def test_missing_columns():
    standard = ['filename','label','time_start','time_stop','freq_min','freq_max']
    df = pd.DataFrame(columns=standard)
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[:-1])
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[1:])
    assert sorted(st.missing_columns(df)) == sorted(['filename'])

def test_create_label_dict():
    l1 = [0, 'gg', -17, 'whale']
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    expected = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 4}
    assert d == expected

def test_create_label_dict_can_handle_nested_list():
    l1 = [0, 'gg', [-17, 'whale']]
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    expected = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 3}
    assert d == expected

def test_unfold(annot_table_mult_labels):
    df = st.unfold(annot_table_mult_labels)
    df_expected = pd.DataFrame({'filename':['f0.wav','f0.wav','f1.wav'], 'label':['1','2','3'], 'time_start':[0,0,1], 'time_stop':[1,1,2]})
    for name in df.columns.values:
        assert np.all(df[name].values == df_expected[name].values)

def test_standardize(annot_table_std):
    df = annot_table_std
    print(df)
    df = df.set_index(['filename', df.index])
    df = df.sort_index()
    print('levels: ', df.index.nlevels)
    print(df)
    df['orig_index'] = df.index.get_level_values(1)
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), df.groupby(level=0).cumcount()],
        names=['filename', 'id'])
    print(df)

def test_standardize_from_file(annot_table_file):
    df, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'time_stop'}, signal_labels=[1,'k'], backgr_labels=[-99, 'whale'])
    d_expected = {-99: 0, 'whale':0, 2: -1, 'zebra': -1, 1: 1, 'k':2}
    assert d == d_expected
    assert sorted(df.columns.values) == sorted(['filename', 'time_start', 'time_stop', 'label'])
    assert sorted(df['label'].values) == sorted([1, -1, 2, 0, 0, -1])

def test_standardize_with_nested_list(annot_table_file):
    df, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'time_stop'}, signal_labels=[[1,'whale'],'k'], backgr_labels=[-99])
    d_expected = {-99: 0, 2: -1, 'zebra': -1, 1: 1, 'whale':1, 'k':2}
    assert d == d_expected
    assert sorted(df.columns.values) == sorted(['filename', 'time_start', 'time_stop', 'label'])
    assert sorted(df['label'].values) == sorted([1, -1, 2, 0, 1, -1])

def test_label_occurrence(annot_table_std):
    df = annot_table_std
    oc = st.label_occurrence(df)
    oc_expected = {-1: 1, 0: 2, 1: 1, 2: 1, 3: 1}
    assert oc == oc_expected

def test_create_selections_center(annot_table_std):
    df = annot_table_std
    # request length shorter than annotations
    df_new = st.create_selections(df, select_len=1, center=True)
    assert len(df_new[df_new.label==-1]) == 0
    for i,r in df_new.iterrows():
        t1 = r['time_start']
        t2 = r['time_stop']
        assert pytest.approx(t1, i + 0.5 * 3.3 - 0.5, abs=0.00001)
        assert pytest.approx(t2, i + 0.5 * 3.3 + 0.5, abs=0.00001)
    # request length longer than annotations
    df_new = st.create_selections(df, select_len=5, center=True)
    for i,r in df_new.iterrows():
        t1 = r['time_start']
        t2 = r['time_stop']
        assert pytest.approx(t1, i + 0.5 * 3.3 - 2.5, abs=0.00001)
        assert pytest.approx(t2, i + 0.5 * 3.3 + 2.5, abs=0.00001)

def test_create_selections_removes_discarded_annotations(annot_table_std):
    df = annot_table_std
    df_new = st.create_selections(df, select_len=1, center=True)
    assert len(df_new[df_new.label==-1]) == 0

def test_create_selections_enforces_overlap(annot_table_std):
    np.random.seed(3)
    df = annot_table_std
    # requested length: 5.0 sec
    # all annotations have length: 3.3 sec  (3.3/5.0=0.66)
    select_len = 5.0
    overlap = 0.5
    df_new = st.create_selections(df, select_len=select_len, min_overlap=overlap, keep_index=True)
    for i,r in df_new.iterrows():
        t1 = r['time_start']
        t2 = r['time_stop']
        idx = r['orig_index']
        t1_orig = df.loc[idx]['time_start']
        t2_orig = df.loc[idx]['time_stop']
        assert t2 >= t1_orig + overlap * select_len
        assert t1 <= t2_orig - overlap * select_len

def test_create_selections_step(annot_table_std):
    df = annot_table_std
    N = len(df[df['label']!=-1])
    K = len(df[df['label']==0])
    df_new = st.create_selections(df, select_len=1, center=True, min_overlap=0, step_size=0.5)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)
    df_new = st.create_selections(df, select_len=1, center=True, min_overlap=0.4, step_size=0.5)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5-0.4)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)

def test_create_rndm_backgr_selections(annot_table_std, file_duration_table):
    np.random.seed(1)
    df = annot_table_std
    dur = file_duration_table 
    num = 5
    df_bgr = st.create_rndm_backgr_selections(table=df, file_duration=dur, select_len=2.0, num=num)
    assert len(df_bgr) == num
    df_c = st.complement(df, dur)
    num_ok = 0
    for i,ri in df_bgr.iterrows():
        dt = ri['time_stop'] - ri['time_start']
        assert pytest.approx(dt, 2.0, abs=0.001)
        for j,rj in df_c.iterrows():
            if ri['filename'] == rj['filename'] and ri['time_start'] >= rj['time_start'] \
                and ri['time_stop'] <= rj['time_stop']:
                num_ok += 1

    assert num_ok == num

def test_complement(annot_table_std, file_duration_table):
    df = annot_table_std
    dur = file_duration_table
    df_new = st.complement(df, dur)
    df_expected = pd.DataFrame()
    df_expected['filename'] = ['f0.wav','f1.wav','f1.wav','f2.wav','f2.wav','f3.wav','f4.wav','f5.wav']
    df_expected['time_start'] = [6.3, 0., 7.3, 0., 8.3, 0., 0., 0.]
    df_expected['time_stop']  = [30.0, 1., 31., 2., 32., 33., 34., 35.]
    assert df_expected.values.tolist() == df_new.values.tolist()
