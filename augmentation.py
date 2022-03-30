import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

IMAGE_AUGMENTATION_SEQUENCE = None
IMAGE_AUGMENTATION_NUM_TRIES = 10

loaded_augmentation_name = ""

def _load_augmentation_aug_geometric():
    return iaa.OneOf([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2)
    ])

augmentation_functions = {
    "aug_geometric": _load_augmentation_aug_geometric
}


def _load_augmentation(augmentation_name="aug_geometric"):

    global IMAGE_AUGMENTATION_SEQUENCE

    if augmentation_name not in augmentation_functions:
        raise ValueError("Augmentation name not supported")

    IMAGE_AUGMENTATION_SEQUENCE = augmentation_functions[augmentation_name]()


def _augment_seg(img, img2, seg, augmentation_name="aug_geometric", other_imgs=None):

    global loaded_augmentation_name

    if (not IMAGE_AUGMENTATION_SEQUENCE) or\
       (augmentation_name != loaded_augmentation_name):
        _load_augmentation(augmentation_name)
        loaded_augmentation_name = augmentation_name

    # Create a deterministic augmentation from the random one
    aug_det = IMAGE_AUGMENTATION_SEQUENCE.to_deterministic()
    # Augment the input image
    image_aug = aug_det.augment_image(img)

    if other_imgs is not None:
        image_aug = [image_aug]

        for other_img in other_imgs:
            image_aug.append(aug_det.augment_image(other_img))
            
    # Augment the input image2
    image_aug2 = aug_det.augment_image(img2)
    
    if other_imgs is not None:
        image_aug2 = [image_aug2]

        for other_img in other_imgs:
            image_aug2.append(aug_det.augment_image(other_img))

    segmap = ia.SegmentationMapsOnImage(
        seg, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr()

    return image_aug, image_aug2, segmap_aug


def _try_n_times(fn, n, *args, **kargs):
    """ Try a function N times """
    attempts = 0
    while attempts < n:
        try:
            return fn(*args, **kargs)
        except Exception:
            attempts += 1

    return fn(*args, **kargs)


def augment_seg(img, img2, seg, augmentation_name="aug_geometric", other_imgs=None):
    return _try_n_times(_augment_seg, IMAGE_AUGMENTATION_NUM_TRIES,
                        img, img2, seg, augmentation_name=augmentation_name,
                        other_imgs=other_imgs)