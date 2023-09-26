import gunpowder as gp
import os

from MembraneSegmentation.io.dataloaders import dataloader_zarrmultiplesources3D
from MembraneSegmentation.pre.pipeline import preprocessing_pipeline
from MembraneSegmentation.models.mknet import MtlsdModel
from MembraneSegmentation.post.train import train
from MembraneSegmentation.utils.script_setup import ScriptSetup, check_folder_exists

print('loading config')
config_path = r'config/MTLSD/MTLSD_config.json'

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
labels = gp.ArrayKey('LABELS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
pred_affs = gp.ArrayKey('PRED_AFFS')
gt_lsds = gp.ArrayKey('GT_LSDS')
lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
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

# model load + save locations
checkpoint_basename = out_dir + "/checkpoints/chkp"
log_dir = out_dir + "/log"

check_folder_exists(out_dir)
check_folder_exists(out_dir + "/checkpoints")
check_folder_exists(log_dir)

batch_dict = {'RAW': raw, 'LABELS': labels, 'GT_AFFS': gt_affs, 'AFFS_WEIGHTS': affs_weights, 'PRED_AFFS': pred_affs, 'GT_LSDS': gt_lsds, 'LSDS_WEIGHTS': lsds_weights, 'PRED_LSDS': pred_lsds}

logging.info('creating data source')
sources = dataloader_zarrmultiplesources3D(raw, labels, data_dir, data_list)

logging.info('creating data pipeline')
pipeline = preprocessing_pipeline(sources, raw, labels, pipeline=None)
pipeline.create_pipeline()
pipeline.add_lsd_pipeline(gt_lsds, lsds_weights)
pipeline.add_affinity_pipeline(gt_affs, affs_weights)
pipeline.add_final_prepprocess_pipeline()

logging.info('creating model')
mtlsd_model = MtlsdModel( num_fmaps, fmap_inc_factor, downsample_factors)
input_size, output_size = mtlsd_model.return_input_output_sizes(input_shape, voxel_size)

logging.info('request batch')
request = gp.BatchRequest()
request.add(raw, input_size)
request.add(labels, output_size)
request.add(gt_affs, output_size)
request.add(affs_weights, output_size)
request.add(pred_affs, output_size)
request.add(gt_lsds, output_size)
request.add(lsds_weights, output_size)
request.add(pred_lsds, output_size)

logging.info('load model into pipeline')
outputs = [pred_lsds, pred_affs]
loss_inputs = [pred_lsds, gt_lsds, lsds_weights, pred_affs, gt_affs, affs_weights]
pipeline.add_model(mtlsd_model, [raw], outputs, loss_inputs, checkpoint_basename, log_dir, save_every=1000, log_every=10, MTLSD=True)

pipeline = pipeline.get_pipeline()

logging.info('train model')
train(request, pipeline, batch_dict, voxel_size).gunpowder_train(max_iteration=30000, test_training=False, show_every=1)
