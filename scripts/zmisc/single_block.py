from MembraneSegmentation.post.segmentation import get_segmentation
import numpy as np

threshold = 0.5
affs = np.load('testAffinties.npy')
segmentation = get_segmentation(np.expand_dims(pred_affs, axis=1), threshold)
