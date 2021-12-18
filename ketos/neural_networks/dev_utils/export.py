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

""" 'neural_networks.dev_utils.export' module within the ketos library

    This module contains utilities for saving ketos models in various formats.
"""

from ketos.data_handling.parsing import parse_audio_representation, encode_audio_representation, parse_parameter, encode_parameter
from tensorflow.saved_model import save as save_pb
from ketos.audio.audio_loader import audio_repres_dict
from zipfile import ZipFile
import warnings
import json
import shutil
import os


def export_to_ketos_protobuf(model, output_name, input_shape, audio_repr=None, audio_repr_file=None, 
                            tmp_folder="tmp_export_folder", overwrite=False, input_duration=None):
    """ Export a ketos model to Ketos-Protobuf format. Protobuf is a free and open-source 
        cross-platform data format developed by Google. 

        Saving your ketos model in Ketos-Protobuf format makes it easier to share it with 
        collaborators and use it with other software applications.

        In particular, the output file generated by this function can be loaded directly into 
        PAMGuard, an open-source and widely adopted application for passive acoustic monitoring (PAM).

        The function generates a zipped archive containing,
        
         * the tensorflow model in protobuf format (model.pb)
         * the audiot representation (audio_repr.json)
         * the ketos model recipe (recipe.json)
       
        The user is free to specify the extension of the output file, but we recommend using \*.ktpb
        as this will allow the file to be recognized and loaded into PAMGuard.

        Args:
            model: 
                The ketos model to be exported. Usually created by one of the Interface classes found 
                in ketos.neural_networks (e.g.: ResNetInterface)
            output_name: str
                The name of the exported model. Must have the extension '.ktpb" to ensure that it can 
                be loaded into PAMGuard.
            input_shape: list or tuple.
                The input shape expected by the model. It can be represented by a tuple or list of four elements: 
                [number of intances, width, height, number of channels). The number of instances and number of channels 
                are commonly 1, and the width and height are usually the number of time and frequency bins ins a 
                spectrogram, respectively. This, however, can vary with the model in question.
            audio_repr: dict
                Audio representation. For example,
                    
                >>> audio_repr = {"spectrogram": {
                ...                   "type": "MagSpectrogram",
                ...                   "rate": "1000 Hz", 
                ...                   "window": "0.256 s",
                ...                   "step": "0.032 s",
                ...                   "freq_min": "0 Hz",
                ...                   "freq_max": "500 Hz",
                ...                   "window_func": "hamming",
                ...                   "transforms": [{"name":"normalize"}]}
                ...              }                

            audio_repr_file: str
                Path to json file containing audio representation. Overwrites audio_repr, if specified.
            tmp_folder: str
                The name for a temporary folder created during the model conversion. It will be deleted upon sucessful execution. 
                If the folder already exists, a 'FileExistsError will be thrown ( unless 'overwite' is set to True).
            overwrite: bool    
                If true and the folder specified in 'tmp_folder' exists, the folder will be overwritten.
            input_duration: None or float
                Duration in seconds of the input sample. If input_duration is None, the duration is taken from the 
                audio representation, or, if not available there, computed as step * input_shape[1]
    """
    assert audio_repr is not None or audio_repr_file is not None, 'either audio_repr or audio_repr_file must be specified'

    if audio_repr_file is not None:
        with open(audio_repr_file, 'r') as fil:
            audio_repr = json.load(fil)
        
    if os.path.exists(tmp_folder):
        if not overwrite:
            raise FileExistsError("{} already exists. If you want to overwrite it set the 'overwrite' argument to True.".format(tmp_folder))
        else:
            shutil.rmtree(tmp_folder)

    assert model.model.built, "The model must be built. Call model.run_on_instance() on a sample input"

    # loop over audio representations (usually, there will only be 1)
    for ar in audio_repr.values():

        assert 'step' in ar.keys(), 'step parameter is missing from audio representation'

        # encode audio representation if it isn't already encoded
        if isinstance(ar['step'], (int, float)): 
            ar = encode_audio_representation(ar)

        if input_duration is None:
            if 'duration' not in ar:
                duration = parse_parameter(name='step', value=ar['step']) * input_shape[1]
                ar['duration'] = encode_parameter(name='duration', value=duration)

        else:
            ar['duration'] = encode_parameter(name='duration', value=input_duration)

        ar['dtype'] = model.model.dtype
        ar['input_ndims'] = model.model.layers[0].input_spec.min_ndim
        ar['input_shape'] = input_shape

    os.makedirs(tmp_folder)
    recipe_path = os.path.join(tmp_folder, 'recipe.json')
    model.save_recipe_file(recipe_path)
    model_path = os.path.join(tmp_folder, 'model')
    save_pb(obj=model.model, export_dir=model_path)

    with ZipFile(output_name, 'w') as zip:
        zip.write(model_path, "model")

        for root, dirs, files in os.walk(model_path):
            renamed_root = root.replace(model_path, "model")
            for d in dirs:
                zip.write(os.path.join(root,d), os.path.join(renamed_root,d))
            for f in files:
                zip.write(os.path.join(root,f),os.path.join(renamed_root,f))            
                    
        zip.write(recipe_path, "recipe.json")

        audio_repr_path = os.path.join(tmp_folder, "audio_repr.json")
        with open(audio_repr_path, 'w') as json_repr:
            json.dump(audio_repr, json_repr)

        zip.write(audio_repr_path, "audio_repr.json")
        
    shutil.rmtree(tmp_folder)


def export_to_protobuf(model, output_folder):
    """ Export a ketos model to Protobuf format (\*.pb).

        See also the related fuction :func:`export_to_ketos_protobuf`.

        Args:
            model: 
                The ketos model to be exported. Usually created by one of the Interface classes found 
                in ketos.neural_networks (e.g.: ResNetInterface)
            output_folder: str
                Folder where the exported model will be saved
    """
    assert model.model.built, "The model must be built. Call model.run_on_instance() on a sample input"

    save_pb(obj=model.model, export_dir=output_folder)