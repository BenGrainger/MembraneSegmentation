import gunpowder as gp
from MembraneSegmentation.pre.add_local_shape_descriptor import AddLocalShapeDescriptor
import math
from gunpowder.torch import Train
import torch
from MembraneSegmentation.models.unet import WeightedMSELoss, MTLSDWeightedMSELoss

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



    def add_final_prepprocess_pipeline(self):
        self.pipeline += gp.Unsqueeze([self.raw])
        self.pipeline += gp.Stack(1)
        

    def add_model(self, model, inputs, outputs, loss_inputs, checkpoint_basename, log_dir, save_every=1000, log_every=10, MTLSD=False, ACLRSD=False):
        """ load model into the pipeline. Each iteration of pipeline will lead to a successive training step.
        Args:

            pipeline: pipeline for training step to be added to

            model: pytorch model

            inputs: (list), e.g [raw, pretrained_lsd]

            outputs: (list) converted to this format e.g. outputs={0: pred_lsds}

            loss_inputs: (list) e.g. converted to this fommat loss_inputs={
                                                0: pred,
                                                1: gt,
                                                2: weights,...
                                            }

            checkpoint_basename: (str)

            log_dir: (str)

        """
        def create_dict(to_be_dct):
            """ turn arrays into proper format
            to_be_dct: (list) of arrays
            """
            dictionary = {}
            for i, array in enumerate(to_be_dct):
                dictionary[i] = array
            return dictionary
    
        if MTLSD:
            loss = MTLSDWeightedMSELoss()
        else:
            loss = WeightedMSELoss()

        if ACRLSD:
            inputs={
                    0: inputs[0],
                    1: inputs[1]
                }
        else:
            inputs={
                    'input': inputs[0]
                }



        self.pipeline += Train(
                model,
                loss=loss,
                optimizer = torch.optim.Adam(model.parameters(),lr=0.5e-4,betas=(0.95,0.999)),
                inputs=inputs,
                outputs=create_dict(outputs),
                loss_inputs=create_dict(loss_inputs),
                checkpoint_basename=checkpoint_basename,
                save_every=save_every,
                log_every=log_every,
                log_dir=log_dir)













