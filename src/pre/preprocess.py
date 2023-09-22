import gunpowder as gp
from src.pre.add_local_shape_descriptor import AddLocalShapeDescriptor
import math

class preprocessing_pipeline(object):
    def __init__(self, source, raw, labels, pipeline):
        """
        Args:

            sources: source already established with dataloaders

            raw: (gp.Arraykey)

            labels: (gp.Arraykey)
        """
        self.source = source
        self.raw = raw
        self.labels = labels 

        self.pipeline = pipeline

    def get_pipeline(self):
        return self.pipeline

    def create_pipeline(self):
        # randomly choose a sample from our tuple of samples - this is absolutely necessary to be able to get any data!
        
        self.pipeline = self.source
        
        self.pipeline += gp.RandomProvider()

        # add steps to the pipeline
        #randomly mirror and transpose a batch
        self.pipeline == gp.SimpleAugment()

        # elastcally deform the batch
        self.pipeline += gp.ElasticAugment(
            [4,40,40],
            [0,2,2],
            [0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=25)

        # randomly shift and scale the intensities
        self.pipeline += gp.IntensityAugment(
            self.raw,
            scale_min=0.9,
            scale_max=1.1,
            shift_min=-0.1,
            shift_max=0.1,
            z_section_wise=True)
        
        # dilate the boundaries between labels
        self.pipeline += gp.GrowBoundary(self.labels, 
                                steps=3,
                                only_xy=True)
        
        

    def add_affinity_pipeline(self, gt_affs, affs_weights):
        """ add steps required for affinity pipeline
        Args:

            pipeline: 

            labels: (gp.Arraykey)

            gt_affs: (gp.Arraykey)

            affs_weights: (gp.Arraykey)
        
        """
        self.pipeline += gp.AddAffinities(
            affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            labels=self.labels,
            affinities=gt_affs)

        # no longer need labels since we use the gt affs for training

        # create scale array to balance class losses (will then use the affs_weights array during training)
        self.pipeline += gp.BalanceLabels(
                gt_affs,
                affs_weights)
        
        
        
    def add_lsd_pipeline(self, gt_lsds, lsds_weights):
        """ add steps required for lsd pipeline
        Args:

            pipeline: 

            labels: (gp.Arraykey)

            gt_lsds: (gp.Arraykey)

            lsds_weights: (gp.Arraykey)
        """
        self.pipeline += AddLocalShapeDescriptor(
            self.labels,
            gt_lsds,
            lsds_mask=lsds_weights,
            sigma=80,
            downsample=2)

        

    def add_final_pipeline(self):
        self.pipeline += gp.Unsqueeze([self.raw])
        self.pipeline += gp.Stack(1)













