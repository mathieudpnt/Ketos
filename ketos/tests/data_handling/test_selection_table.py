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
from io import StringIO


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_trim():
    standard = ['filename','label','start','end','freq_min','freq_max']
    extra = ['A','B','C']
    df = pd.DataFrame(columns=extra)
    df = st.trim(df)
    assert len(df.columns) == 0
    df = pd.DataFrame(columns=standard+extra)
    df = st.trim(df)
    assert sorted(df.columns.values) == sorted(standard)

def test_missing_columns():
    standard = ['filename','label','start','end','freq_min','freq_max']
    df = pd.DataFrame(columns=standard)
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[:-1])
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[1:])
    assert sorted(st.missing_columns(df)) == ['filename']

def test_is_standardized():
    df = pd.DataFrame({'filename':'test.wav','label':[1],'start':[0],'end':[2],'freq_min':[None],'freq_max':[None]})
    df, d = st.standardize(df)
    assert st.is_standardized(df) == True
    df = pd.DataFrame({'filename':'test.wav','label':[1]})
    df, d = st.standardize(df)
    assert st.is_standardized(df) == True
    df = pd.DataFrame({'filename':'test.wav','label':[1]})
    assert st.is_standardized(df) == False

def test_create_label_dict():
    l1 = [0, 'gg', -17, 'whale']
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    ans = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 4}
    assert d == ans

def test_create_label_dict_can_handle_nested_list():
    l1 = [0, 'gg', [-17, 'whale']]
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    ans = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 3}
    assert d == ans

def test_unfold(annot_table_mult_labels):
    res = st.unfold(annot_table_mult_labels)
    ans = pd.DataFrame({'filename':['f0.wav','f0.wav','f1.wav'], 'label':['1','2','3'], 'start':[0,0,1], 'end':[1,1,2]})
    res = res.reset_index(drop=True)[ans.columns]
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize(annot_table_std):
    res, d = st.standardize(annot_table_std)
    d = '''filename annot_id label  start  end                   
f0.wav   0             3    0.0  3.3
f0.wav   1             2    3.0  6.3
f1.wav   0             4    1.0  4.3
f1.wav   1             2    4.0  7.3
f2.wav   0             5    2.0  5.3
f2.wav   1             1    5.0  8.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize_from_file(annot_table_file):
    res, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'end'}, signal_labels=[1,'k'], backgr_labels=[-99, 'whale'])
    ans = {-99: 0, 'whale':0, 2: -1, 'zebra': -1, 1: 1, 'k':2}
    assert d == ans
    d = '''filename annot_id label  start  end                   
f0.wav   0             1      0    1
f1.wav   0            -1      1    2
f2.wav   0             2      2    3
f3.wav   0             0      3    4
f4.wav   0             0      4    5
f5.wav   0            -1      5    6'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize_with_nested_list(annot_table_file):
    res, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'end'}, signal_labels=[[1,'whale'],'k'], backgr_labels=[-99])
    ans = {-99: 0, 2: -1, 'zebra': -1, 1: 1, 'whale':1, 'k':2}
    assert d == ans
    d = '''filename annot_id label  start  end                   
f0.wav   0             1      0    1
f1.wav   0            -1      1    2
f2.wav   0             2      2    3
f3.wav   0             0      3    4
f4.wav   0             1      4    5
f5.wav   0            -1      5    6'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])
    
def test_label_occurrence(annot_table_std):
    df = annot_table_std
    oc = st.label_occurrence(df)
    ans = {-1: 1, 0: 2, 1: 1, 2: 1, 3: 1}
    assert oc == ans

def test_select_center(annot_table_std):
    df, d = st.standardize(annot_table_std)
    # request length shorter than annotations
    res = st.select(df, length=1, center=True)
    d = '''filename sel_id label  start   end
f0.wav   0           3   1.15  2.15
f0.wav   1           2   4.15  5.15
f1.wav   0           4   2.15  3.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15
f2.wav   1           1   6.15  7.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])
    # request length longer than annotations
    res = st.select(df, length=5, center=True)
    d = '''filename sel_id  label  start   end
f0.wav   0           3  -0.85  4.15
f0.wav   1           2   2.15  7.15
f1.wav   0           4   0.15  5.15
f1.wav   1           2   3.15  8.15
f2.wav   0           5   1.15  6.15
f2.wav   1           1   4.15  9.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_select_removes_discarded_annotations(annot_table_std):
    df = annot_table_std
    df, d = st.standardize(df)
    res = st.select(df, length=1, center=True)
    assert len(res[res.label==-1]) == 0

def test_select_enforces_overlap(annot_table_std):
    np.random.seed(3)
    df = annot_table_std
    df, d = st.standardize(df)
    # requested length: 5.0 sec
    # all annotations have length: 3.3 sec  (3.3/5.0=0.66)
    length = 5.0
    overlap = 0.5
    df_new = st.select(df, length=length, min_overlap=overlap, keep_id=True)
    t1 = df_new.start.values
    t2 = df_new.end.values
    idx = zip(df_new.index.get_level_values(0), df_new.annot_id)
    df = df.loc[idx]
    t2_orig = df.end.values
    t1_orig = df.start.values
    assert np.all(t2 >= t1_orig + overlap * length)
    assert np.all(t1 <= t2_orig - overlap * length)

def test_select_step(annot_table_std):
    df = annot_table_std
    df, d = st.standardize(df)
    N = len(df[df['label']!=-1])
    K = len(df[df['label']==0])
    df_new = st.select(df, length=1, center=True, min_overlap=0, step=0.5, keep_id=True)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)
    df_new = st.select(df, length=1, center=True, min_overlap=0.4, step=0.5)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5-0.4)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)

def test_time_shift(annot_table_std):
    row = pd.Series({'label':3.00,'start':0.00,'end':3.30,'annot_id':0.00,'length':3.30,'start_new':-0.35})
    res = st.time_shift(annot=row, time_ref=row['start_new'], length=4.0, min_overlap=0.8, step=0.5)
    d = '''label  start  end  annot_id  length  start_new
0    3.0    0.0  3.3       0.0     3.3      -0.35'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_select_with_varying_overlap(annot_table_std):
    """ Test that the number of selections increases as the 
        minimum required overlap is reduced"""
    df, d = st.standardize(annot_table_std)
    # request length shorter than annotations
    num_sel = []
    for min_overlap in np.linspace(1.0, 0.0, 11):
        res = st.select(df, length=1.0, step=0.5, center=True, min_overlap=min_overlap)
        num_sel.append(len(res))
    
    assert np.all(np.diff(num_sel) >= 0)
    # request length longer than annotations
    num_sel = []
    for min_overlap in np.linspace(1.0, 0.0, 11):
        res = st.select(df, length=4.0, step=0.5, center=True, min_overlap=min_overlap)
        num_sel.append(len(res))

    assert np.all(np.diff(num_sel) >= 0)

def test_create_rndm_backgr_selections(annot_table_std, file_duration_table):
    np.random.seed(1)
    df, _ = st.standardize(annot_table_std)
    dur = file_duration_table 
    num = 5
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=num)
    print(df_bgr)
    assert len(df_bgr) == num
    df_c = st.complement(df, dur)
    # assert selections have uniform length
    assert np.all(df_bgr.end.values - df_bgr.start.values == 2.0)
    # assert selections are within complement
    for bgr_idx, bgr_sel in df_bgr.iterrows():
        start_bgr = bgr_sel.start
        end_bgr = bgr_sel.end
        fname = bgr_idx[0]
        df = df_c.loc[fname,:]
        start_c = df.start.values
        end_c = df.end.values
        assert np.any(np.logical_and(start_bgr >= start_c, end_bgr <= end_c))

def test_create_rndm_backgr_keeps_misc_cols(annot_table_std, file_duration_table):
    """ Check that the random background selection creation method keeps 
        any miscellaneous columns"""
    np.random.seed(1)
    df, _ = st.standardize(annot_table_std)
    df['extra'] = 'testing'
    dur = file_duration_table 
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=5)
#    assert np.all(df_bgr['extra'].values == 'testing')

def test_complement(annot_table_std, file_duration_table):
    df, _ = st.standardize(annot_table_std)
    dur = file_duration_table
    res = st.complement(df, dur)
    d = '''filename annot_id  start   end
f0.wav   0           6.3  30.0
f1.wav   0           0.0   1.0
f1.wav   1           7.3  31.0
f2.wav   0           0.0   2.0
f2.wav   1           8.3  32.0
f3.wav   0           0.0  33.0
f4.wav   0           0.0  34.0
f5.wav   0           0.0  35.0'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_select_by_segmenting(annot_table_std, file_duration_table):
    a, _ = st.standardize(annot_table_std)
    f = file_duration_table
    sel = st.select_by_segmenting(a, f, length=5.1, step=4.0, discard_empty=True, pad=True)
    # check selection table
    d = '''filename sel_id start  end
f0.wav   0         0.0  5.1
f0.wav   1         4.0  9.1
f1.wav   0         0.0  5.1
f1.wav   1         4.0  9.1
f2.wav   0         0.0  5.1
f2.wav   1         4.0  9.1
f2.wav   2         8.0 13.1'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, sel[0][ans.columns.values])
    # check annotation table
    d = '''filename sel_id annot_id label  start  end
f0.wav   0      0             3         0.0        3.3
f0.wav   0      1             2         3.0        5.1
f0.wav   1      1             2         0.0        2.3
f1.wav   0      0             4         1.0        4.3
f1.wav   0      1             2         4.0        5.1
f1.wav   1      0             4         0.0        0.3
f1.wav   1      1             2         0.0        3.3
f2.wav   0      0             5         2.0        5.1
f2.wav   0      1             1         5.0        5.1
f2.wav   1      0             5         0.0        1.3
f2.wav   1      1             1         1.0        4.3
f2.wav   2      1             1         0.0        0.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1,2])
    pd.testing.assert_frame_equal(ans, sel[1][ans.columns.values])

def test_query_labeled(annot_table_std):
    df, d = st.standardize(annot_table_std)
    df = st.select(df, length=1, center=True)
    # query for 1 file
    q = st.query_labeled(df, filename='f1.wav')
    d = '''sel_id label  start   end                   
0           4   2.15  3.15
1           2   5.15  6.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for 2 files
    q = st.query_labeled(df, filename=['f1.wav','f2.wav'])
    d = '''filename sel_id label  start   end                            
f1.wav   0           4   2.15  3.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15
f2.wav   1           1   6.15  7.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for labels
    q = st.query_labeled(df, label=[2,5])
    d = '''filename sel_id label  start   end                   
f0.wav   1           2   4.15  5.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])

def test_query_annotated(annot_table_std, file_duration_table):
    a, _ = st.standardize(annot_table_std)
    f = file_duration_table
    sel = st.select_by_segmenting(a, f, length=5.1, step=4.0, discard_empty=True, pad=True)
    # query for 1 file
    q1, q2 = st.query_annotated(sel[0], sel[1], label=[2,4])
    d = '''filename sel_id start  end
f0.wav   0         0.0  5.1
f0.wav   1         4.0  9.1
f1.wav   0         0.0  5.1
f1.wav   1         4.0  9.1'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q1, ans[q1.columns.values])
    d = '''filename sel_id annot_id label  start  end  
f0.wav   0      1             2    3.0  5.1
f0.wav   1      1             2    0.0  2.3
f1.wav   0      0             4    1.0  4.3
f1.wav   0      1             2    4.0  5.1
f1.wav   1      0             4    0.0  0.3
f1.wav   1      1             2    0.0  3.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1,2])
    pd.testing.assert_frame_equal(q2, ans[q2.columns.values])
