import torch
from src.models.unet import UNet, ConvPass

def create_affinity_model(num_fmaps, fmap_inc_factor, downsample_factors):
    """
    Args:

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps of the Unet before passing to the final convolutional layer. Stored in the ``channels``
                dimension. this is often 12

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``. this is often 6

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

    """

    # create unet
    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors)

    # add an extra convolution to get from 12 feature maps to 3 (affs in x,y,z)
    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=num_fmaps, out_channels=3, kernel_sizes=[[1, 1, 1]], activation='Sigmoid'))
    
    return model




def create_lsd_model(num_fmaps, fmap_inc_factor, downsample_factors):
    """
    Args:

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps of the Unet before passing to the final convolutional layer. Stored in the ``channels``
                dimension. this is often 12

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``. this is often 6

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

    """

    # create unet
    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors)

    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=num_fmaps, out_channels=10, kernel_sizes=[[1, 1, 1]], activation='Sigmoid'))
    
    return model




def create_aclsd_model(num_fmaps, fmap_inc_factor, downsample_factors):
    """
    Args:

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps of the Unet before passing to the final convolutional layer. Stored in the ``channels``
                dimension. this is often 12

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``. this is often 6

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

    """

    # create unet
    unet = UNet(
        in_channels=10,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors)

    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=num_fmaps, out_channels=3, kernel_sizes=[[1, 1, 1]], activation='Sigmoid'))

    return model