import gunpowder as gp
import numpy as np
import os
from MembraneSegmentation.utils.script_setup import get_project_root
from MembraneSegmentation.io.dataloaders import dataloader_zarr3D, dataloader_zarr3Dpredict
from MembraneSegmentation.pre.pipeline import preprocessing_pipeline

raw = gp.ArrayKey('RAW')
labels = gp.ArrayKey('LABELS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')

voxel_size = gp.Coordinate((40, 4, 4)) 
input_size = gp.Coordinate((64, 256, 256)) * voxel_size

root = get_project_root()
dir = os.path.join(root, 'beng/Data/segmentations/data3D/val.zarr')

source = dataloader_zarr3D(raw, labels, dir)

scan_request = gp.BatchRequest()
for i in [raw, labels, gt_affs, affs_weights]:
    scan_request.add(i, input_size)
scan = gp.Scan(scan_request)

pipeline = preprocessing_pipeline(source, raw, labels, pipeline=None)
pipeline.create_pipeline(augment=False)
pipeline.add_affinity_pipeline(gt_affs, affs_weights)
pipeline.add_final_prepprocess_pipeline()
pipeline = pipeline.get_pipeline()
pipeline += scan

size_source  = dataloader_zarr3Dpredict(raw, dir) # only works with predict source
with gp.build(size_source):
    total_input_roi = size_source.spec[raw].roi

request = gp.BatchRequest()
request.add(raw, total_input_roi.get_end())
request.add(labels, total_input_roi.get_end())
request.add(gt_affs, total_input_roi.get_end())
request.add(affs_weights, total_input_roi.get_end())

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

test_affs = batch[gt_affs].data

np.save('testAffinties.npy', test_affs)