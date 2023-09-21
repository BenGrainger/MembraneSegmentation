import gunpowder as gp
import sys
import json
import os

sys.path.append(r'C://Users/Crab_workstation/Documents/GitHub/MembraneSegmentation')
from src.io.dataloaders import dataloader_zarr3Dpredict
from src.models.mknet import create_lsd_model, return_input_output_sizes
from src.post.predict import predict_pipeline, get_input_output_roi
from src.utils.utility_funcs import find_latest_checkpoint

print('loading config')
config_path = 'config/affinities_config.json'

with open(config_path, 'r') as config_file:
    config = json.load(config_file)

print('establishing parameters')
parent_dir = config['parent_dir'] 
validation_dir = config["validation_dir"]
validation_path = os.path.join(parent_dir, validation_dir)

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
pred_lsds = gp.ArrayKey('PRED_LSDS')

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

print('creating val data source')
source  = dataloader_zarr3Dpredict(raw, validation_path)

print('creating model')
model_aff  = create_lsd_model(num_fmaps, fmap_inc_factor, downsample_factors)
input_size, output_size = return_input_output_sizes(input_shape, voxel_size, model_aff)

print('creating predict pipeline')
pred_outs = {0: pred_lsds}
out_dir = config["out_directory"]
checkpoint = find_latest_checkpoint(out_dir + "/checkpoints")
pipeline = predict_pipeline(source, model_aff, raw, pred_outs, input_size, output_size, checkpoint)
total_input_roi, total_output_roi = get_input_output_roi(source, raw, input_size, output_size)

print('request batch')
predict_request = gp.BatchRequest()
predict_request.add(raw, total_input_roi.get_end())
predict_request.add(pred_lsds, total_output_roi.get_end())

with gp.build(pipeline):
    batch = pipeline.request_batch(predict_request)

