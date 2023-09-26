import torch
from MembraneSegmentation.models.unet import UNet, ConvPass
import gunpowder as gp


class mknet(object):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, model):
        """ returns a pytorch implementation of the unet based affinities model (untrained)
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
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.model = model

    def get_model(self):
        return self.model
    
    def create_model(self, in_channels, out_channels):
        """
        Args:

                in_channels:

                    The number of channels input to model

                out_channels:

                    the number of output channels of the model 

        """
        # create unet
        unet = UNet(
            in_channels=in_channels,
            num_fmaps=self.num_fmaps,
            fmap_inc_factor=self.fmap_inc_factor,
            downsample_factors=self.downsample_factors)

        # add an extra convolution to get from 12 feature maps to 3 (affs in x,y,z)
        self.model = torch.nn.Sequential(
            unet,
            ConvPass(in_channels=self.num_fmaps, out_channels=out_channels, kernel_sizes=[[1, 1, 1]], activation='Sigmoid'))
        
    def create_affinity_model(self):
        self.create_model(1, 3)

    def create_LSD_model(self):
        self.create_model(1, 10)

    def create_ACLSD_model(self):
        self.create_model(10, 3)

    def return_input_output_sizes(self, input_shape, voxel_size):
        """ returns the input and output size of the model in gp.Coordinates. Used to request/specify batches/arrays of certain sizes
        Args:

                input_shape: (list) e.g. [z, x, y] no batch or channel size

                voxel_size: (gp.Coordinate) e.g. gp.Coordinate((40, 4, 4)) 

                model: pytorch model 

        """
        model_input = torch.ones([1, 1, input_shape[0], input_shape[1], input_shape[2]])
        outputs = self.model(model_input)
        output_shape = gp.Coordinate((outputs.shape[2], outputs.shape[3], outputs.shape[4]))
        input_size = gp.Coordinate(input_shape) * voxel_size
        output_size = output_shape * voxel_size
        
        return input_size, output_size


class MtlsdModel(torch.nn.Module):

    def __init__(
        self,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors):
  
        super().__init__()

        # create unet
        self.unet = UNet(
          in_channels=1,
          num_fmaps=num_fmaps,
          fmap_inc_factor=fmap_inc_factor,
          downsample_factors=downsample_factors,
          constant_upsample=True)

        # create lsd and affs heads
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        # pass raw through unet
        z = self.unet(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs
    
    def return_input_output_sizes(self, input_shape, voxel_size):
        """ returns the input and output size of the model in gp.Coordinates. Used to request/specify batches/arrays of certain sizes
        Args:

                input_shape: (list) e.g. [z, x, y] no batch or channel size

                voxel_size: (gp.Coordinate) e.g. gp.Coordinate((40, 4, 4)) 

                model: pytorch model 

        """
        model_input = torch.ones([1, 1, input_shape[0], input_shape[1], input_shape[2]])
        outputs = self.forward(model_input)
        output_shape = gp.Coordinate((outputs[0].shape[2], outputs[0].shape[3], outputs[0].shape[4]))
        input_size = gp.Coordinate(input_shape) * voxel_size
        output_size = output_shape * voxel_size
        
        return input_size, output_size