import os

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


print(__file__)


print("\ncurrent dir: ", current_dir)
print("\nassets: ", path_to_assets)
print("\ntmp: ", path_to_tmp)