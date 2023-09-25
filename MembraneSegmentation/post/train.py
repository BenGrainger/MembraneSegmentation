import gunpowder as gp
from tqdm import tqdm
from src.utils.utility_funcs import imshow

class train(object):
    def __init__(self, request, pipeline, batch_keys, voxel_size):
        """ train model with gunpowder api
        Args:

            request: request for batch(s) via gunpowder api

            pipeline: 

            batch_keys: (dict) dictionary containing the gp.Arraykeys for easier access

            voxel_size: (gp.Coordinate) e.g. gp.Coordinate((40, 4, 4)) 
        """
        self.request = request
        self.pipeline = pipeline
        self.batch_keys = batch_keys
        self.voxel_size = voxel_size

    def gunpowder_train(self, max_iteration=100, test_training=False, show_every=1):
        """ initiate training
        """
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
            'voxel count': 5}

        with gp.build(self.pipeline):
            progress = tqdm(range(max_iteration))
            for i in progress:
                batch = self.pipeline.request_batch(self.request)
                if test_training:
                    if i % show_every == 0:
                        
                        start = self.request[self.batch_keys['LABELS']].roi.get_begin()/self.voxel_size
                        end = self.request[self.batch_keys['LABELS']].roi.get_end()/self.voxel_size

                        batch_raw = batch[self.batch_keys['RAW']].data[:,:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                        batch_raw_images = batch_raw[0][0][0:5]
                        imshow(
                            raw=batch_raw_images
                        )

                        batch_labels = batch[self.batch_keys['LABELS']].data
                        batch_labels_images = batch_labels[0][0:5]
                        imshow(
                            ground_truth=batch_labels_images
                        )
                        if 'GT_LSDS' in self.batch_keys:
                            for n,c in lsd_channels.items():

                                batch_lsds = batch[self.batch_keys['GT_LSDS']].data
                                batch_lsds_images = batch_lsds[0][c][0:5]
                                imshow(
                                    target=batch_lsds_images, target_name='gt'+n
                                )
                                
                                batch_pred = batch[self.batch_keys['PRED_LSDS']].data
                                batch_pred_images = batch_pred[0][c][0:5]
                                imshow(
                                    prediction=batch_pred_images, prediction_name='pred'+n 
                                )
                        if 'GT_AFFS' in self.batch_keys:
                            for n,c in aff_channels.items():

                                batch_lsds = batch[self.batch_keys['GT_AFFS']].data
                                batch_lsds_images = batch_lsds[0][c][0:5]
                                imshow(
                                    target=batch_lsds_images, target_name='gt'+n
                                )
                                
                                batch_pred = batch[self.batch_keys['PRED_AFFS']].data
                                batch_pred_images = batch_pred[0][c][0:5]
                                imshow(
                                    prediction=batch_pred_images, prediction_name='pred'+n 
                                )