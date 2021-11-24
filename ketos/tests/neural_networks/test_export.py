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

""" Unit tests for the 'neural_networks.dev_utils.export' module within the ketos library
"""
import pytest
import os
import numpy as np
from ketos.neural_networks import load_model_file
from ketos.audio.audio_loader import audio_repres_dict
from ketos.neural_networks.dev_utils.export import export_to_protobuf, export_to_ketos_protobuf


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_export_to_ketos_protobuf():
    """Test export resnet to pamguard format"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    input_spec = np.ones(shape=(94,129))
    model.run_on_instance(input_spec)
    output_path = os.path.join(path_to_tmp, 'narw1.ktpb')
    export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr=audio_repr[0], overwrite=True, input_shape=(1,94,129,1))
    assert os.path.isfile(output_path)

def test_export_to_ketos_protobuf_audio_repr_file():
    """Test export resnet to pamguard format using audio representation file path"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    audio_repr_path = os.path.join(path_to_assets, 'audio_repr.json')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    input_spec = np.ones(shape=(94,129))
    model.run_on_instance(input_spec)
    output_path = os.path.join(path_to_tmp, 'narw2.ktpb')
    export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr_file=audio_repr_path, overwrite=True, input_shape=(1,94,129,1))
    assert os.path.isfile(output_path)

def test_export_to_protobuf():
    """Test export resnet to protobuf format"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model = load_model_file(model_path, tmp_path, load_audio_repr=False)
    input_spec = np.ones(shape=(94,129))
    model.run_on_instance(input_spec)
    output_path = os.path.join(path_to_tmp, 'pb_model')
    export_to_protobuf(model=model, output_folder=output_path)
    assert os.path.isdir(output_path)
