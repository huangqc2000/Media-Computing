from utils import get_img, get_coordinates, compute_gradient_laplace, scale_img
from PIL import Image
import jittor as jt
import numpy as np


def get_img_fitting_dataset(img_path, sidelength, channels):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    coords = get_coordinates(sidelength, dim=2)
    pixels = img.reshape(-1, channels)  # HW, C
    return {
        'coords': coords,
        'img': pixels
    }


def get_poisson_equation_dataset(img_path, sidelength, channels):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    coords = get_coordinates(sidelength, dim=2)
    pixels = img.reshape(-1, channels)

    grads, laplaces = compute_gradient_laplace(img, channels)

    return {
        'coords': coords,
        'img': pixels,
        'gradient': grads,
        'laplace': laplaces
    }


def get_composite_dataset(img_path1, img_path2, sidelength, channels):
    img1 = get_img(sidelength=sidelength, img_path=img_path1, channels=channels)
    img2 = get_img(sidelength=sidelength, img_path=img_path2, channels=channels)

    coords = get_coordinates(sidelength, dim=2)
    pixels1 = img1.reshape(-1, channels)
    pixels2 = img2.reshape(-1, channels)
    grads1, _ = compute_gradient_laplace(img1, channels)
    grads2, _ = compute_gradient_laplace(img2, channels)

    grads = 0.5 * grads1 + 0.5 * grads2

    return {
        'coords': coords,
        'gradient': grads,
        'img1': pixels1,
        'img2': pixels2
    }


def get_img_inpainting_dataset(img_path, sidelength, channels, sample_rate=None, crop_half = None, points_choose=None):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    all_coords = get_coordinates(sidelength, dim=2)
    all_pixels = img.reshape(-1, channels)

    if sample_rate is not None:
        mask = np.random.choice(np.arange(0, all_coords.shape[0]), int(sample_rate * all_coords.shape[0]), replace=False)
    elif crop_half is not None:
        mask = np.arange(0, all_coords.shape[0] // 2)
    elif points_choose is not None:
        mask = np.random.choice(np.arange(0, all_coords.shape[0]), points_choose, replace=False)
    else:
        mask = np.arange(0, all_coords.shape[0])

    coords = all_coords[mask]
    pixels = all_pixels[mask]

    mask_img = jt.ones_like(all_pixels)
    mask_img[mask] = pixels

    mask_img = scale_img(mask_img, sidelength=sidelength, channels=channels, ref=None)

    return {
        'all_coords': all_coords,
        'all_pixels': all_pixels,
        'coords': coords,
        'img': pixels,
        'mask_img': mask_img
    }


if __name__ == '__main__':
    print(1)
