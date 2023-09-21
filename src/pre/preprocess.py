import gunpowder as gp
from src.pre.add_local_shape_descriptor import AddLocalShapeDescriptor
import math


def initialize_training_pipeline(sources, raw, labels):
    """ create and add source to pipeline as well as add ubiquitous augmentation steps
    Args:

        sources: source already established with dataloaders

        raw: (gp.Arraykey)

        labels: (gp.Arraykey)
    """
    pipeline = sources

    # randomly choose a sample from our tuple of samples - this is absolutely necessary to be able to get any data!
    pipeline += gp.RandomProvider()

    # add steps to the pipeline
    #randomly mirror and transpose a batch
    pipeline == gp.SimpleAugment()

    # elastcally deform the batch
    pipeline += gp.ElasticAugment(
        [4,40,40],
        [0,2,2],
        [0,math.pi/2.0],
        prob_slip=0.05,
        prob_shift=0.05,
        max_misalign=25)

    # randomly shift and scale the intensities
    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        z_section_wise=True)
    
    # dilate the boundaries between labels
    pipeline += gp.GrowBoundary(labels, 
                             steps=3,
                             only_xy=True)
    
    return pipeline


def add_affinity_pipeline(pipeline, labels, gt_affs, affs_weights):
    """ add steps required for affinity pipeline
    Args:

        pipeline: 

        labels: (gp.Arraykey)

        gt_affs: (gp.Arraykey)

        affs_weights: (gp.Arraykey)
    
    """
    pipeline += gp.AddAffinities(
        affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        labels=labels,
        affinities=gt_affs)

    # no longer need labels since we use the gt affs for training

    # create scale array to balance class losses (will then use the affs_weights array during training)
    pipeline += gp.BalanceLabels(
            gt_affs,
            affs_weights)
    
    return pipeline
    
def add_lsd_pipeline(pipeline, labels, gt_lsds, lsds_weights):
    """ add steps required for lsd pipeline
    Args:

        pipeline: 

        labels: (gp.Arraykey)

        gt_lsds: (gp.Arraykey)

        lsds_weights: (gp.Arraykey)
    """
    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=80,
        downsample=2)
    
    return pipeline

def create_lsd_preprocess_pipeline(sources, raw, labels, gt_lsds, lsds_weights):
    """ create the full lsd pipeline
    Args:

        sources: data source established with dataloaders

        raw: (gp.Arraykey)

        pred_lsds: (gp.Arraykey)

        gt_lsds: (gp.Arraykey)

        lsds_weights: (gp.Array)
    """
    pipeline = initialize_training_pipeline(sources, raw, labels)
    pipeline = add_lsd_pipeline(pipeline, labels, gt_lsds, lsds_weights)
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)
    return pipeline


def create_affinity_preprocess_pipeline(sources, raw, labels, gt_affs, affs_weights):
    """ create the full aff pipeline
    Args:

        sources: data source established with dataloaders
        
        raw: (gp.Arraykey)

        pred_affs: (gp.Arraykey)

        gt_affs: (gp.Arraykey)

        affs_weights: (gp.Arraykey)
    
    """
    pipeline = initialize_training_pipeline(sources, raw, labels)
    pipeline = add_affinity_pipeline(pipeline, labels, gt_affs, affs_weights)
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)
    return pipeline

def create_mtlsd_preprocess_pipeline(sources, raw, labels, gt_affs, affs_weights, gt_lsds, lsds_weights):
    """ create the full MTLSD pipeline
    Args:

    sources: data source established with dataloaders

    raw: (gp.Arraykey)

    pred_lsds: (gp.Arraykey)

    gt_lsds: (gp.Arraykey)

    lsds_weights: (gp.Array)

    pred_affs: (gp.Arraykey)

    gt_affs: (gp.Arraykey)

    affs_weights: (gp.Arraykey)
    
    """
    pipeline = initialize_training_pipeline(sources, raw, labels)
    pipeline = add_lsd_pipeline(pipeline, labels, gt_lsds, lsds_weights)
    pipeline = add_affinity_pipeline(pipeline, labels, gt_affs, affs_weights)
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)
    return pipeline


