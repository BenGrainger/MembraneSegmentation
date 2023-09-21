import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np
import os

def find_latest_checkpoint(checkpoint_path):
    """ returns the path for the highest checkpoint
    Args:

        checkpoint_path: (str)
    """
    checkpoint_list = os.listdir(checkpoint_path)
    def return_int_in_name(name):
        """
        name: (str)
        """
        parts = name.split('_')
        return int(parts[2])
    int_list = [return_int_in_name(i) for i in checkpoint_list]
    max_checkpoint= np.max(np.array(int_list))
    max_checkpoint_name = [i for i in checkpoint_list if str(max_checkpoint) in i][0]
    return os.path.join(checkpoint_path, max_checkpoint_name)

def create_lut(labels):

    max_label = np.max(labels)

    lut = np.random.randint(
            low=0,
            high=255,
            size=(int(max_label + 1), 3),
            dtype=np.uint8)

    lut = np.append(
            lut,
            np.zeros(
                (int(max_label + 1), 1),
                dtype=np.uint8) + 255,
            axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels


def imshow(
        raw=None,
        ground_truth=None,
        target=None,
        prediction=None,
        h=None,
        shader='jet',
        subplot=True,
        channel=0,
        target_name='target',
        prediction_name='prediction',
        save_name=None):

    rows = 0

    if raw is not None:
        rows += 1
        cols = raw.shape[0] if len(raw.shape) > 2 else 1
    if ground_truth is not None:
        rows += 1
        cols = ground_truth.shape[0] if len(ground_truth.shape) > 2 else 1
    if target is not None:
        rows += 1
        cols = target.shape[0] if len(target.shape) > 2 else 1
    if prediction is not None:
        rows += 1
        cols = prediction.shape[0] if len(prediction.shape) > 2 else 1

    if subplot:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            squeeze=False)

    if h is not None:
        fig.subplots_adjust(hspace=h)

    def wrapper(data,row,name="raw"):

        if subplot:
            if len(data.shape) == 2:
                if name == 'raw':
                    axes[0][0].imshow(data, cmap='gray')
                    axes[0][0].set_title(name)
                if name == 'labels':
                    axes[row][0].imshow(create_lut(data))
                    axes[row][0].set_title(name)
                else: 
                    axes[row][0].imshow(im, cmap=shader)
                    axes[row][0].set_title(name)

            elif len(data.shape) == 3:
                for i, im in enumerate(data):
                    if name == 'raw':
                        axes[0][i].imshow(im, cmap='gray')
                        axes[0][i].set_title(name)
                    elif name == 'labels':
                        axes[row][i].imshow(create_lut(im))
                        axes[row][i].set_title(name)
                    else:
                        axes[row][i].imshow(im, cmap=shader)
                        axes[row][i].set_title(name)

            else:
                for i, im in enumerate(data):
                    axes[row][i].imshow(im[channel], cmap=shader)
                    axes[row][i].set_title(name)


        else:
            if name == 'raw':
                plt.imshow(data, cmap='gray')
            if name == 'labels':
                plt.imshow(data, alpha=0.5)

    row=0 

    if raw is not None:
        wrapper(raw,row=row)
        row += 1
    if ground_truth is not None:
        wrapper(ground_truth,row=row,name='labels')
        row += 1
    if target is not None:
        wrapper(target,row=row,name=target_name)
        row += 1
    if prediction is not None:
        wrapper(prediction,row=row,name=prediction_name)
        row += 1

    plt.show()
    if save_name is not None:
        fig.savefig(save_name)