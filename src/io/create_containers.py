import zarr

def create_zarr_container_2D(raw, segmentation, destination_folder, offset, resolution):
    """
    Args:

        raw: (numpy array)

        segementation: (numpy array)

        destination_folder: (str)

        offset: (list) e.g. [0,0] for cremi

        resolution: (list) e.g. [4,4] for cremi
    
    """
    # gunpowder relies on zarr
    # create output zarr container
    container = zarr.open(destination_folder, 'w') # do not pre make this folder

    # make zarr container partitions

    sections = range(raw.shape[0]-1)

    for index, section in enumerate(sections):

        raw_slice = raw[section]
        segmentation_slice = segmentation[section]

        # write data into zarr container slice by slice and set resolution and offset
        for ds_name, data in [
            ('raw', raw_slice),
            ('labels', segmentation_slice)]:
            
            container[f'{ds_name}/{index}'] = data
            container[f'{ds_name}/{index}'].attrs['offset'] = offset # set offset
            container[f'{ds_name}/{index}'].attrs['resolution'] = resolution # set resolution

def create_zarr_container_2D(raw, segmentation, destination_folder, offset, resolution):
    """
    Args:

        raw: (numpy array)

        segementation: (numpy array)

        destination_folder: (str)

        offset: (list) e.g. [0,0,0] for cremi

        resolution: (list) e.g. [40,4,4] for cremi
    
    """
    # gunpowder relies on zarr
    # create output zarr container
    container = zarr.open(destination_folder, 'w') # do not pre make this folder

    # make zarr container partitions

    # write data into zarr container and set resolution and offset     
    container['raw'] = raw
    container['raw'].attrs['offset'] = offset # set offset
    container['raw'].attrs['resolution'] = resolution # set resolution

    container['segmentation'] = segmentation
    container['segmentation'].attrs['offset'] = offset # set offset
    container['segmentation'].attrs['resolution'] = resolution # set resolution
