import gunpowder as gp
import os

from MembraneSegmentation.io.dataloaders import dataloader_zarrmultiplesources3D_autocontext
from MembraneSegmentation.pre.pipeline import preprocessing_pipeline
from MembraneSegmentation.models.mknet import mknet
from MembraneSegmentation.post.train import train
from MembraneSegmentation.utils.script_setup import ScriptSetup, check_folder_exists

print('loading config')
config_path = r'config/ACLSD/ACLSD_config.json'

script = ScriptSetup(config_path)
script.load_script()
config = script.return_config()
logging = script.return_logger()
out_dir = script.return_out_dir()
root = script.return_root()

logging.info('establishing parameters')
data_dir = os.path.join(root, config['data_dir'])
data_list = config["data_list"]
data_list = [i for i in data_list.values()]

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
labels = gp.ArrayKey('GT_LABELS')
pretrained_lsd = gp.ArrayKey('PRETRAINED_LSD')
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
checkpoint_basename = out_dir + "/checkpoints/chkp"
log_dir = out_dir + "/log"

check_folder_exists(out_dir)
check_folder_exists(out_dir + "/checkpoints")
check_folder_exists(log_dir)

batch_dict = {'RAW': raw, 'LABELS': labels, 'PRETRAINED_LSD': pretrained_lsd, 'GT_AFFS': gt_affs, 'AFFS_WEIGHTS': affs_weights, 'PRED_AFFS': pred_affs}

logging.info('creating data source')
sources = dataloader_zarrmultiplesources3D_autocontext(raw, labels, pretrained_lsd, data_dir, data_list)

logging.info('creating data pipeline')
pipeline = preprocessing_pipeline(sources, raw, labels, pipeline=None)
pipeline.create_pipeline()
pipeline.add_affinity_pipeline(gt_affs, affs_weights)
pipeline.add_final_prepprocess_pipeline()

logging.info('creating model')
aclsd_model = mknet(num_fmaps, fmap_inc_factor, downsample_factors, model=None)
aclsd_model.create_ACLSD_model()
input_size, output_size = aclsd_model.return_input_output_sizes(input_shape, voxel_size, 10)
aclsd_model = aclsd_model.get_model()

logging.info('request batch')
request = gp.BatchRequest()
request.add(raw, input_size)
request.add(labels, output_size)
request.add(pretrained_lsd, input_size)
request.add(gt_affs, output_size)
request.add(affs_weights, output_size)
request.add(pred_affs, output_size)

logging.info('load model into pipeline')
outputs = [pred_affs]
loss_inputs = [pred_affs, gt_affs, affs_weights]
pipeline.add_model(aclsd_model, [raw], outputs, loss_inputs, checkpoint_basename, log_dir, save_every=1000, log_every=10)

pipeline = pipeline.get_pipeline()

logging.info('train model')
train(request, pipeline, batch_dict, voxel_size).gunpowder_train(max_iteration=10000, test_training=False, show_every=1)

