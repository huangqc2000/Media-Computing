from utils import get_img, get_coordinates, compute_gradient_laplace, scale_img
from PIL import Image
import torch
import numpy as np


def get_img_fitting_dataset(img_path, sidelength, channels, device):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    coords = get_coordinates(sidelength, dim=2)
    pixels = img.reshape(-1, channels)

    coords, pixels = coords.to(device), pixels.to(device)

    return {
        'coords': coords,
        'img': pixels
    }


def get_poisson_equation_dataset(img_path, sidelength, channels, device):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    coords = get_coordinates(sidelength, dim=2)
    pixels = img.reshape(-1, channels)

    grads, laplaces = compute_gradient_laplace(img, channels)

    coords, pixels, grads, laplaces = coords.to(device), pixels.to(device), grads.to(device), laplaces.to(device)
    return {
        'coords': coords,
        'img': pixels,
        'gradient': grads,
        'laplace': laplaces
    }


def get_composite_dataset(img_path1, img_path2, sidelength, channels, device):
    img1 = get_img(sidelength=sidelength, img_path=img_path1, channels=channels)
    img2 = get_img(sidelength=sidelength, img_path=img_path2, channels=channels)

    coords = get_coordinates(sidelength, dim=2)
    pixels1 = img1.reshape(-1, channels)
    pixels2 = img2.reshape(-1, channels)
    grads1, _ = compute_gradient_laplace(img1, channels)
    grads2, _ = compute_gradient_laplace(img2, channels)

    grads = 0.5 * grads1 + 0.5 * grads2

    coords, grads, pixels1, pixels2 = coords.to(device), grads.to(device), pixels1.to(device), pixels2.to(device)

    return {
        'coords': coords,
        'gradient': grads,
        'img1': pixels1,
        'img2': pixels2
    }


def get_img_inpainting_dataset(img_path, sidelength, channels, device, sample_rate=0.5):
    img = get_img(sidelength, img_path, channels)  # H, W, C
    all_coords = get_coordinates(sidelength, dim=2)
    all_pixels = img.reshape(-1, channels)

    mask = np.random.choice(np.arange(0, all_coords.shape[0]), int(sample_rate * all_coords.shape[0]), replace=False)
    coords = all_coords[mask]
    pixels = all_pixels[mask]

    mask_img = torch.ones_like(all_pixels)
    mask_img[mask] = pixels

    mask_img = scale_img(mask_img, sidelength=sidelength, channels=channels, ref=None)

    all_coords, all_pixels, coords, pixels = all_coords.to(device), all_pixels.to(device), coords.to(device), pixels.to(device)

    return {
        'all_coords':all_coords,
        'all_pixels':all_pixels,
        'coords':coords,
        'img':pixels,
        'mask_img':mask_img
    }


if __name__ == '__main__':
    dataset = get_img_inpainting_dataset(img_path='./data/knot.jpg', sidelength=512,
                                         channels=3,
                                         device=torch.device('cuda'))
    print(dataset)
