from .cnn import CNNInterface, CNN1DInterface
from .resnet import ResNetInterface, ResNet1DInterface
from .densenet import DenseNetInterface
from .inception import InceptionInterface
from zipfile import ZipFile
from shutil import rmtree
import json
import os


interface_names_in_recipes = {'CNNInterface':CNNInterface,
                              'CNN1DInterface':CNN1DInterface,
                              'ResNetInterface':ResNetInterface,
                              'ResNet1DInterface':ResNet1DInterface,
                              'DenseNetInterface':DenseNetInterface,
                              'InceptionInterface':InceptionInterface}

def load_model_file(model_file, new_model_folder,  overwrite=True, load_audio_repr=False,  replace_top=False, diff_n_classes=None):
        """ Load a model from a ketos (.kt) model file.

            Args:
                model_file:str
                    Path to the ketos(.kt) file
                new_model_folder:str
                    Path to folder where files associated with the model will be stored.
                overwrite: bool
                    If True, the 'new_model_folder' will be overwritten.
                replace_top: bool
                    If True, the classification top of the model will be replaced by a new, untrained one.
                    What is actually replaced (i.e.: what exactly is the "top") is defined by the architecture.
                    It is usually a block of dense layers with the appropriate activations. Default is False.
                diff_n_classes: int
                    Only relevant when 'replace_top' is True.
                    If the new model should have a different number of classes it can be specified by this parameter.
                    If left to none, the new model will have the same number of classes as the original.
                load_audio_repr: bool
                    If True, look for an audio representation included with the model. 
                    
            Raises:
                FileExistsError: If the 'new_model_folder' already exists and 'overwite' is False.
                ValueError: If the model recipe does not contain a valid value for the 'interface' field.

            Returns:
                model_instance: The loaded model
                audio_repr: If load_audio_repr is True, also return a dictionary with the loaded audio representation.

        """

        try:
            os.makedirs(new_model_folder)
        except FileExistsError:
            if overwrite == True:
                rmtree(new_model_folder)
                os.makedirs(new_model_folder)
            else:
                raise FileExistsError("Ketos needs a new folder for this model. Choose a folder name that does not exist or set 'overwrite' to True to replace the existing folder")

        with ZipFile(model_file, 'r') as zip:
            zip.extractall(path=new_model_folder)
    
        with open(os.path.join(new_model_folder,"recipe.json"), 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        try:
            nn_interface = recipe_dict['interface']
            loaded = interface_names_in_recipes[nn_interface].load_model_file(model_file=model_file, new_model_folder=new_model_folder,  overwrite=True, load_audio_repr=load_audio_repr,  replace_top=replace_top, diff_n_classes=diff_n_classes)
        except KeyError:
            raise ValueError("The model recipe does not indicate a valid interface")

        return loaded
            