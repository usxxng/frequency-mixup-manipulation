import torchio as tio

def intensity_transform():
    ## image transforms ##
    transforms_dict = {
        #tio.Resize([224,224,224]),
        #tio.RandomGhosting(p=0.5),
        #tio.RandomBiasField(p=1.0),
        #tio.RandomMotion(p=1.0),
        tio.RandomNoise(p=1.0),
        #tio.RandomAffine(p=1.0),
        #tio.RandomElasticDeformation(p=0.5),
        #tio.RandomFlip(p=1.0),
        #tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    }
    transform = tio.Compose(transforms_dict)

    return transform
