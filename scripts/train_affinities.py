import gunpowder as gp
import sys
import json
import os

sys.path.append(r'C://Users/Crab_workstation/Documents/GitHub/MembraneSegmentation')
from src.io.dataloaders import dataloader_zarrmultiplesources3D
from src.pre.preprocess import create_affinity_preprocess_pipeline
from src.models.mknet import create_affinity_model, return_input_output_sizes
from src.post.train import load_affinity_model, gunpowder_train

print('loading config')
config_path = 'config/affinities_train_config.json'

with open(config_path, 'r') as config_file:
    config = json.load(config_file)

print('establishing parameters')
parent_dir = config['parent_dir'] 
data_dir_list = config["data_dir_list"]
data_dir_list = [i for i in data_dir_list.values()]

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
labels = gp.ArrayKey('LABELS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
pred_affs = gp.ArrayKey('PRED_AFFS')

# data parameters
z, x, y = config["zxy"]
input_shape = [z, x, y]
vx, vy, vz = config["voxel_dimensions"]
voxel_size = gp.Coordinate((vz, vx, vy)) 

# model parameters
in_channels= config["in_channels"]
num_fmaps = config["num_fmaps"]
fmap_inc_factor= config["fmap_inc_factor"]
ds1, ds2, ds3 = config["downsample_factors"]
downsample_factors=[(1,ds1,ds1),(1,ds2,ds2),(1,ds3,ds3)] # 1 in the z due to datasets being non isotropic in z

# model load + save locations
out_dir = config["out_directory"]
checkpoint_basename = out_dir + "/checkpoints/chkp"
log_dir = out_dir + "/log"

def check_folder_exists(directory):
    if not os.path.exists(directory):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{directory}': {e}")
    else:
        print(f"Folder '{directory}' already exists.")

check_folder_exists(out_dir)
check_folder_exists(out_dir + "/checkpoints")
check_folder_exists(log_dir)

batch_dict = {'RAW': raw, 'LABELS': labels, 'GT_AFFS': gt_affs, 'AFFS_WEIGHTS': affs_weights, 'PRED_AFFS': pred_affs}

print('creating data source')
sources  = dataloader_zarrmultiplesources3D(raw, labels, parent_dir, data_dir_list)

print('creating data pipeline')
pipeline = create_affinity_preprocess_pipeline(sources, raw, labels, gt_affs, affs_weights)

print('creating model')
model_aff  = create_affinity_model(num_fmaps, fmap_inc_factor, downsample_factors)
input_size, output_size = return_input_output_sizes(input_shape, voxel_size, model_aff)

print('load model into pipeline')
pipeline = load_affinity_model(pipeline, model_aff, raw, pred_affs, gt_affs, affs_weights, checkpoint_basename, log_dir, save_every=1, log_every=1)

print('request batch')
request = gp.BatchRequest()
request.add(raw, input_size)
request.add(labels, output_size)
request.add(gt_affs, output_size)
request.add(affs_weights, output_size)
request.add(pred_affs, output_size)

print('train model')
gunpowder_train(request, pipeline, batch_dict, voxel_size, max_iteration=10, test_training=True, show_every=1)
