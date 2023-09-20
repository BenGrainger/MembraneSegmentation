import gunpowder as gp
from gunpowder.torch import Train
import torch
from tqdm import tqdm
from src.models.unet import WeightedMSELoss
from src.utils.utility_funcs import imshow

def load_affinity_model(pipeline, model, raw, pred_affs, gt_affs, affs_weights, checkpoint_basename, log_dir, save_every=1000, log_every=10):
    pipeline += Train(
            model,
            WeightedMSELoss(),
            optimizer = torch.optim.Adam(model.parameters(),lr=0.5e-4,betas=(0.95,0.999)),
            inputs={
                'input': raw
            },
            outputs={
                0: pred_affs
            },
            loss_inputs={
                0: pred_affs,
                1: gt_affs,
                2: affs_weights
            },
            checkpoint_basename=checkpoint_basename,
            save_every=save_every,
            log_every=log_every,
            log_dir=log_dir)
    
    return pipeline



def load_LSD_model(pipeline, model, raw, pred_lsds, gt_lsds, lsds_weights, checkpoint_basename, log_dir, save_every=1000, log_every=10):
    pipeline += Train(
            model,
            WeightedMSELoss(),
            optimizer = torch.optim.Adam(model.parameters(),lr=0.5e-4,betas=(0.95,0.999)),
            inputs={
                'input': raw
            },
            outputs={
                0: pred_lsds
            },
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: lsds_weights
            },
            checkpoint_basename=checkpoint_basename,
            save_every=save_every,
            log_every=log_every,
            log_dir=log_dir)

    return pipeline



def load_MTLSD_model(pipeline, model, raw, pred_lsds, gt_lsds, lsds_weights, pred_affs, gt_affs, affs_weights, checkpoint_basename, log_dir, save_every=1000, log_every=10):
    pipeline += Train(
            model,
            WeightedMSELoss(),
            optimizer = torch.optim.Adam(model.parameters(),lr=0.5e-4,betas=(0.95,0.999)),
            inputs={
                'input': raw
            },
            outputs={
                0: pred_lsds,
                1: pred_affs
            },
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: lsds_weights,
                3: pred_affs,
                4: gt_affs,
                5: affs_weights
            },
            checkpoint_basename=checkpoint_basename,
            save_every=save_every,
            log_every=log_every,
            log_dir=log_dir)



def gunpowder_train(request, pipeline, batch_keys, voxel_size, max_iteration=100, test_training=False, show_every=1):

    # affinity channels
    aff_channels = {'affs1': 0,
                    'affs2': 1,
                    'affs3': 2}

    # lsd channels dict
    lsd_channels = {
        'offset (y)': 0,
        'offset (x)': 1,
        'orient (y)': 2,
        'orient (x)': 3,
        'yx change': 4,
        'voxel count': 5
    }

    with gp.build(pipeline):
        progress = tqdm(range(max_iteration))
        for i in progress:
            batch = pipeline.request_batch(request)
            if test_training:
                if i % show_every == 0:
                    
                    start = request[batch_keys['LABELS']].roi.get_begin()/voxel_size
                    end = request[batch_keys['LABELS']].roi.get_end()/voxel_size

                    batch_raw = batch[batch_keys['RAW']].data[:,:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                    batch_raw_images = batch_raw[0][0][0:5]
                    imshow(
                        raw=batch_raw_images
                    )

                    batch_labels = batch[batch_keys['LABELS']].data
                    batch_labels_images = batch_labels[0][0:5]
                    imshow(
                        ground_truth=batch_labels_images
                    )
                    if 'GT_LSDS' in batch_keys:
                        for n,c in lsd_channels.items():

                            batch_lsds = batch[batch_keys['GT_LSDS']].data
                            batch_lsds_images = batch_lsds[0][c][0:5]
                            imshow(
                                target=batch_lsds_images, target_name='gt'+n
                            )
                            
                            batch_pred = batch[batch_keys['PRED_LSDS']].data
                            batch_pred_images = batch_pred[0][c][0:5]
                            imshow(
                                prediction=batch_pred_images, prediction_name='pred'+n 
                            )
                    if 'GT_AFFS' in batch_keys:
                        for n,c in aff_channels.items():

                            batch_lsds = batch[batch_keys['GT_AFFS']].data
                            batch_lsds_images = batch_lsds[0][c][0:5]
                            imshow(
                                target=batch_lsds_images, target_name='gt'+n
                            )
                            
                            batch_pred = batch[batch_keys['PRED_AFFS']].data
                            batch_pred_images = batch_pred[0][c][0:5]
                            imshow(
                                prediction=batch_pred_images, prediction_name='pred'+n 
                            )