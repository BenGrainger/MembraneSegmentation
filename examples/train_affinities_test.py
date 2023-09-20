#%%
import gunpowder as gp
import torch

import sys
sys.path.append(r'C://Users/Crab_workstation/Documents/GitHub/MembraneSegmentation')

from src.io.dataloaders import dataloader_zarrmultiplesources3D
from src.pre.preprocess import create_aff_pipeline
from src.models.mknet import create_affinity_model, return_input_output_sizes

# Array keys for gunpowder interface
raw = gp.ArrayKey('RAW')
labels = gp.ArrayKey('LABELS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
pred_affs = gp.ArrayKey('PRED_AFFS')

parent_dir = r"U://zoo/users/beng/automatic_segmentation/LSD/data3D"
data_dir_list = ["trainA.zarr", "trainB.zarr"]

sources  = dataloader_zarrmultiplesources3D(raw, labels, parent_dir, data_dir_list)
pipeline = create_aff_pipeline(sources, raw, labels, gt_affs, affs_weights)

model_aff  = create_affinity_model(12, 6, [[2,2,3]])
input_shape = [84, 268, 268]
voxel_size = gp.Coordinate((40, 4, 4)) 
input_size, output_size = return_input_output_sizes(input_shape, voxel_size, model_aff)
# %%
