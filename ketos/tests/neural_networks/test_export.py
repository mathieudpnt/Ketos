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
import shutil
from zipfile import ZipFile
from ketos.neural_networks import load_model_file
from ketos.audio.audio_loader import audio_repres_dict
from ketos.data_handling.parsing import load_audio_representation
import ketos.neural_networks.dev_utils.export as exp


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_export_to_ketos_protobuf():
    """Test export resnet to ketos-protobuf format"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    
    output_path = os.path.join(path_to_tmp, 'narw1.ktpb')
    exp.export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr=audio_repr[0], 
        input_shape=(94,129), backward_compat=False)
    
    #check that file exists:
    assert os.path.isfile(output_path) 
    
    # check that file has correct content:
    with ZipFile(output_path, 'r') as zip:
        zip.extractall(path=tmp_path)
    
    assert os.path.isdir(os.path.join(tmp_path, 'model'))
    assert os.path.isdir(os.path.join(tmp_path, 'checkpoints'))
    assert os.path.isfile(os.path.join(tmp_path, 'recipe.json'))
    assert os.path.isfile(os.path.join(tmp_path, 'audio_repr.json'))

    # load audio representation
    audio_repr = load_audio_representation(os.path.join(tmp_path, 'audio_repr.json'))
    assert 'spectrogram' in audio_repr

    # check that duration was written to file
    a = audio_repr['spectrogram']
    assert 'duration' in a
    assert a['duration'] == "3.008 s"

    # check that input_shape was written to file
    assert 'input_shape' in a
    assert a['input_shape'] == [1,94,129,1]

    shutil.rmtree(tmp_path) #clean up


def test_export_to_ketos_protobuf_backward_compat():
    """Test export resnet to ketos-protobuf format backward compatible"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    output_path = os.path.join(path_to_tmp, 'narw1.ktpb')

    exp.export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr=audio_repr[0], 
        input_shape=(1,94,129,1))
    
    assert os.path.isfile(output_path)


def test_export_to_ketos_protobuf_infer_shape():
    """Test export resnet to ketos-protobuf format can infer shape"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    output_path = os.path.join(path_to_tmp, 'narw1.ktpb')

    with pytest.raises(AssertionError):
        exp.export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr=audio_repr[0])

    exp.export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr=audio_repr[0], duration=3.0)

    # check that file exists:
    assert os.path.isfile(output_path)

    # check that shape was inferred correctly
    with ZipFile(output_path, 'r') as zip:
        zip.extractall(path=tmp_path)
    audio_repr = load_audio_representation(os.path.join(tmp_path, 'audio_repr.json'))
    assert audio_repr['spectrogram']['input_shape'] == [1,94,129,1]
    shutil.rmtree(tmp_path) #clean up


def test_export_to_ketos_protobuf_audio_repr_file():
    """Test export resnet to ketos-protobuf format using audio representation file path"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    audio_repr_path = os.path.join(path_to_assets, 'audio_repr.json')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model = load_model_file(model_path, tmp_path)
    output_path = os.path.join(path_to_tmp, 'narw2.ktpb')

    exp.export_to_ketos_protobuf(model=model, output_name=output_path, audio_repr_file=audio_repr_path, 
        overwrite=True, duration=3.0)

    assert os.path.isfile(output_path)


def test_export_to_protobuf():
    """Test export resnet to protobuf format"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model = load_model_file(model_path, tmp_path)
    input_spec = np.ones(shape=(94,129))
    model.run_on_instance(input_spec)
    output_path = os.path.join(path_to_tmp, 'model.pb')

    exp.export_to_protobuf(model=model, output_name=output_path)

    assert os.path.isdir(output_path)


def test_export_to_ketos():
    """Test export resnet to ketos format"""
    model_path = os.path.join(path_to_assets, 'narw_resnet.kt')
    tmp_path = os.path.join(path_to_tmp, 'tmp_folder')
    model, audio_repr = load_model_file(model_path, tmp_path, load_audio_repr=True)
    audio_repr = audio_repr[0]

    # check that we can save model
    output_path = os.path.join(path_to_tmp, 'narw3.kt')
    exp.export_to_ketos(model=model, output_name=output_path)
    assert os.path.isfile(output_path)

    # check that we can save specific checkpoint
    output_path = os.path.join(path_to_tmp, 'narw4.kt')
    exp.export_to_ketos(model=model, output_name=output_path, checkpoint_name="cp-0030.ckpt")
    assert os.path.isfile(output_path)

    # check that audio representation is saved correctly
    output_path = os.path.join(path_to_tmp, 'narw5.kt')
    exp.export_to_ketos(model=model, output_name=output_path, audio_repr=audio_repr)
    assert os.path.isfile(output_path)

    with ZipFile(output_path, 'r') as zip:
        zip.extractall(path=tmp_path)
    audio_repr = load_audio_representation(os.path.join(tmp_path, 'audio_repr.json'))
    assert audio_repr['spectrogram']['window'] == audio_repr['spectrogram']['window']
    
    shutil.rmtree(tmp_path) #clean up


def test_get_export_function():
    """Test that we can detect the appropriate export function"""
    assert exp.get_export_function('/test/out.pb') == exp.export_to_protobuf
    assert exp.get_export_function('/test/out.kt') == exp.export_to_ketos
    assert exp.get_export_function('/test/out.ktpb') == exp.export_to_ketos_protobuf
    assert exp.get_export_function('/test/out.aaa') == exp.export_to_ketos
