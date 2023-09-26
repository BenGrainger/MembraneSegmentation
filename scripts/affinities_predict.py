import gunpowder as gp
import json
import os

from MembraneSegmentation.io.dataloaders import dataloader_zarr3Dpredict
from MembraneSegmentation.models.mknet import mknet
from MembraneSegmentation.post.predict import predict_pipeline, get_input_output_roi
from MembraneSegmentation.utils.utility_funcs import find_latest_checkpoint
from MembraneSegmentation.utils.script_setup import ScriptSetup

print('loading config')
config_path = r'config/affinities/affinities_config.json'

script = ScriptSetup(config_path)
script.load_script()
config = script.return_config()
out_dir = script.return_out_dir()
root = script.return_root()

print('establishing parameters')
data_dir = os.path.join(root, config['data_dir'])
validation_dir = config["validation_dir"]
validation_path = os.path.join(data_dir, validation_dir)

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
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

print('creating val data source')
source  = dataloader_zarr3Dpredict(raw, validation_path)

print('creating model')
aff_model = mknet(num_fmaps, fmap_inc_factor, downsample_factors, model=None)
aff_model.create_affinity_model()
input_size, output_size = aff_model.return_input_output_sizes(input_shape, voxel_size)
aff_model = aff_model.get_model()

print('creating predict pipeline')
pred_outs = {0: pred_affs}
checkpoint = find_latest_checkpoint(out_dir + "/checkpoints")
pipeline = predict_pipeline(source, aff_model, raw, pred_outs, input_size, output_size, checkpoint).create_pipeline()
total_input_roi, total_output_roi = get_input_output_roi(source, raw, input_size, output_size)


print('request batch')
predict_request = gp.BatchRequest()
predict_request.add(raw, total_input_roi.get_end())
predict_request.add(pred_affs, total_output_roi.get_end())

with gp.build(pipeline):
    batch = pipeline.request_batch(predict_request)


