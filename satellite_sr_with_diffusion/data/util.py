import os

import torch
import torchvision

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]


def is_image_file(filename):
    """Checks if a file is a common image file."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    """Returns a list of all image paths in a directory."""
    assert os.path.isdir(path), f"{path:s} is not a valid directory"
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, f"{path:s} has no valid image file"
    return sorted(images)


def transform_augment(img_list, split="val", min_max=(0, 1)):
    """
    Transforms and augments a list of PIL images using torchvision.
    Includes conversion to tensor and optional horizontal flipping for training.
    """
    totensor = torchvision.transforms.ToTensor()
    hflip = torchvision.transforms.RandomHorizontalFlip()
    imgs = [totensor(img) for img in img_list]
    if split == "train":
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    return [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
