import torchio as tio

def intensity_transform():
    ## image transforms ##
    transforms_dict = {
        tio.RandomNoise(p=1.0),
    }
    transform = tio.Compose(transforms_dict)

    return transform
