import os

current_dir = os.getcwd()
path_to_assets = os.path.join(current_dir,"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')



print("current dir: ", current_dir)
print("assets: ", path_to_assets)
print("tmp: ", path_to_tmp)