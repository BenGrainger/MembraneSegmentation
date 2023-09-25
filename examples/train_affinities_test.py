#%%
import gunpowder as gp

from MembraneSegmentation.io.dataloaders import dataloader_zarrmultiplesources3D
from MembraneSegmentation.pre.pipeline import create_affinity_preprocess_pipeline
from MembraneSegmentation.models.mknet import create_affinity_model, return_input_output_sizes
from MembraneSegmentation.post.train import load_affinity_model, gunpowder_train

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
labels = gp.ArrayKey('LABELS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
pred_affs = gp.ArrayKey('PRED_AFFS')

batch_dict = {'RAW': raw, 'LABELS': labels, 'GT_AFFS': gt_affs, 'AFFS_WEIGHTS': affs_weights, 'PRED_AFFS': pred_affs}

parent_dir = r"U://users/beng/automatic_segmentation/LSD/data3D"
data_dir_list = ["trainA.zarr", "trainB.zarr"]

sources  = dataloader_zarrmultiplesources3D(raw, labels, parent_dir, data_dir_list)
pipeline = create_affinity_preprocess_pipeline(sources, raw, labels, gt_affs, affs_weights)

model_aff  = create_affinity_model(12, 6, [[2,2,3]])
input_shape = [84, 268, 268]
voxel_size = gp.Coordinate((40, 4, 4)) 
input_size, output_size = return_input_output_sizes(input_shape, voxel_size, model_aff)

pipeline = load_affinity_model(pipeline, model_aff, raw, pred_affs, gt_affs, affs_weights)

request = gp.BatchRequest()
request.add(raw, input_size)
request.add(labels, output_size)
request.add(gt_affs, output_size)
request.add(affs_weights, output_size)
request.add(pred_affs, output_size)

gunpowder_train(request, pipeline, batch_dict, voxel_size, max_iteration=10, test_training=True, show_every=1)
# %%
