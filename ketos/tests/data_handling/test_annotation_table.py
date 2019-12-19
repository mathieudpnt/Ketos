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

""" Unit tests for the annotation_table module within the ketos library
"""
import pytest
import os
import ketos.data_handling.annotation_table as at
import pandas as pd
import numpy as np


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_trim():
    standard = ['filename','label','time_start','time_stop','freq_min','freq_max']
    extra = ['A','B','C']
    df = pd.DataFrame(columns=extra)
    df = at.trim(df)
    assert len(df.columns) == 0
    df = pd.DataFrame(columns=standard+extra)
    df = at.trim(df)
    assert sorted(df.columns.values) == sorted(standard)

def test_missing_columns():
    standard = ['filename','label','time_start','time_stop','freq_min','freq_max']
    df = pd.DataFrame(columns=standard)
    assert len(at.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[:-1])
    assert len(at.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[1:])
    assert sorted(at.missing_columns(df)) == sorted(['filename'])

def test_create_label_dict():
    l1 = [0, 'gg', -17, 'whale']
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = at.create_label_dict(l1, l2, l3)
    expected = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 4}
    assert d == expected

def test_unfold(annot_table_mult_labels):
    df = at.unfold(annot_table_mult_labels)
    df_expected = pd.DataFrame({'filename':['f0.wav','f0.wav','f1.wav'], 'label':['1','2','3'], 'time_start':[0,0,1], 'time_stop':[1,1,2]})
    for name in df.columns.values:
        assert np.all(df[name].values == df_expected[name].values)

def test_standardize_from_file(annot_table_file):
    df, d = at.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'time_stop'}, signal_labels=[1,'k'], backgr_labels=[-99, 'whale'])
    d_expected = {-99: 0, 'whale':0, 2: -1, 'zebra': -1, 1: 1, 'k':2}
    assert d == d_expected
    assert sorted(df.columns.values) == sorted(['filename', 'time_start', 'time_stop', 'label'])
    assert sorted(df['label'].values) == sorted([1, -1, 2, 0, 0, -1])

