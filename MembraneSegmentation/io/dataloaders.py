import gunpowder as gp


def dataloader_zarr2D(raw, labels, dir, num_samples):
    """ provides 2D arrays from the zarr container, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        labels: (gp.Arraykey)

        dir: (str)

        num_samples: (int) number of samples per batch
    """
    sources = tuple(
    gp.ZarrSource(
        dir,  
        {
            raw: f'raw/{i}',
            labels: f'labels/{i}'
        },  
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False)
        }) + 

        # convert raw to float in [0, 1]
        gp.Normalize(raw) +

        # chose a random location for each requested batch
        gp.RandomLocation()

        for i in range(num_samples)
    )
    return sources



def dataloader_zarr3D(raw, labels, dir): 
    """ provides 3D arrays from the zarr container, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        labels: (gp.Arraykey)

        dir: (str)

    """
    source = gp.ZarrSource(
        dir,
        {
            raw: '/raw',
            labels: '/segmentation'
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False)
        }) 
    
    sources = (
        # read from zarr file
        source +

        # convert raw to float in [0, 1]
        gp.Normalize(raw) +

        # chose a random location for each requested batch
        gp.RandomLocation()
    ) 
    return sources



def dataloader_zarr3Dpredict(raw, dir):
    """ provides 3D arrays from the zarr container, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        dir: (str)
    """
    source = gp.ZarrSource(
        dir,
        {
            raw: '/raw'
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        }
        )
    return source

def dataloader_zarr3Dpredict_autocontext(lsd, dir):
    """ provides 3D arrays from the zarr container, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        dir: (str)
    """
    source = gp.ZarrSource(
        dir,
        {
            lsd: '/lsd'
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        }
        )
    return source



def dataloader_zarrmultiplesources3D(raw, labels, parent_dir, data_dir_list): 
    """ provides 3D arrays from multiple zarr containers, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        labels: (gp.Arraykey)

        parent_dir: (str)

        data_dir_list: (list) list of the zarr file names
    """
    sources = tuple(
            # read batches from the Zarr file
            gp.ZarrSource(
                parent_dir+'/'+s,
                datasets = {
                    raw: '/raw',
                    labels: '/segmentation'
                },
                array_specs = {
                    raw: gp.ArraySpec(interpolatable=True),
                    labels: gp.ArraySpec(interpolatable=False)
                }
            ) +

            # convert raw to float in [0, 1]
            gp.Normalize(raw) +

            # chose a random location for each requested batch
            gp.RandomLocation()

            for s in data_dir_list
        ) 
    return sources


def dataloader_zarrmultiplesources3D_autocontext(raw, labels, lsd, parent_dir, data_dir_list): 
    """ provides 3D arrays from multiple zarr containers, in other words creates the source for the pipeline
    Args:

        raw: (gp.Arraykey)

        labels: (gp.Arraykey)

        parent_dir: (str)

        data_dir_list: (list) list of the zarr file names
    """
    sources = tuple(
            # read batches from the Zarr file
            gp.ZarrSource(
                parent_dir+'/'+s,
                datasets = {
                    raw: '/raw',
                    labels: '/segmentation',
                    lsd: '/lsd'
                },
                array_specs = {
                    raw: gp.ArraySpec(interpolatable=True),
                    labels: gp.ArraySpec(interpolatable=False),
                    lsd: gp.ArraySpec(interpolatable=True)
                }
            ) +

            # convert raw to float in [0, 1]
            gp.Normalize(raw) +

            # chose a random location for each requested batch
            gp.RandomLocation()

            for s in data_dir_list
        ) 
    return sources
