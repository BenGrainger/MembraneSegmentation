import gunpowder as gp


def dataloader_zarr2D(raw, labels, dir, num_samples):
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




def dataloader_zarrmultiplesources3D(raw, labels, parent_dir, data_dir_list): 
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
