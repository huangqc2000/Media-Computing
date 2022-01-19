import jittor as jt
from PIL import Image
from jittor import transform
import matplotlib.pyplot as plt
import os
import scipy.ndimage
import numpy as np


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += jt.grad(y[..., i], x)[..., i:i + 1]
    return div


def gradient(y, x):
    return jt.grad(y, x)


def compute_gradient(img, channels):
    img_np = img.numpy()
    grads = []
    for c in range(channels):
        gradx = scipy.ndimage.sobel(img_np[:, :, c] * 1e1, axis=0)[..., None]
        grady = scipy.ndimage.sobel(img_np[:, :, c] * 1e1, axis=1)[..., None]
        grads += [jt.array(gradx), jt.array(grady)]

    grads = jt.concat(grads, dim=-1)  # H, W, 2C
    grads = grads.reshape(-1, 2 * channels)  # HW, 2C
    return grads


def compute_laplace(img, channels):
    img_np = img.numpy()
    lapls = []
    for c in range(channels):
        lapl = scipy.ndimage.laplace(img_np * 1e4)[..., None]
        lapls += [jt.array(lapl)]

    lapls = jt.concat(lapls, dim=-1)  # H, W, C
    lapls = lapls.reshape(-1, channels)  # HW, C
    return lapls


def compute_gradient_laplace(img, channels):
    return compute_gradient(img, channels), compute_laplace(img, channels)


def get_coordinates(len, dim):
    lens = dim * (len,)
    linespace = [jt.linspace(-1, 1, len) for l in lens]
    grids = jt.meshgrid(*linespace)
    coords = jt.stack(grids, dim=-1)
    coords = coords.view(-1, dim)
    return coords


def get_img(sidelength, img_path, channels):
    img = Image.open(img_path).resize((sidelength, sidelength))
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = jt.array(np.array(img))

    if channels == 1:
        img = img.unsqueeze(-1)

    trans = transform.ImageNormalize(mean=[0.5], std=[0.5])

    img = img / 255
    img = trans(img)

    return img


def scale_img(output, sidelength, channels, ref):
    img = output.reshape(sidelength, sidelength, channels).detach()
    if ref is not None:
        img = (img - img.min()) / (img.max() - img.min()) * (ref.max() - ref.min()) + ref.min()

    img = img * 0.5 + 0.5
    img = (img.safe_clip(0, 1) * 255).numpy().astype(np.uint8)
    if channels == 1:
        mode = 'L'
        img = img.squeeze(-1)
    else:
        mode = 'RGB'
    img = Image.fromarray(img, mode=mode)
    return img


def save_result(model_output, sidelength, channels, model_dir, ref=None, compute_grad=False):
    output, coords = model_output['model_out'], model_output['model_in']

    if channels == 1 and compute_grad:
        result = output.reshape(sidelength, sidelength).detach().numpy()
        grad = gradient(output, coords)
        grad = grad.norm(dim=-1).reshape(sidelength, sidelength).detach().numpy()
        laplacian = laplace(output, coords)
        laplacian = laplacian.reshape(sidelength, sidelength).detach().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(result)
        axes[0].set_xlabel("image")
        axes[1].imshow(grad)
        axes[1].set_xlabel("gradient")
        axes[2].imshow(laplacian)
        axes[2].set_xlabel("laplace")
        plt.savefig(os.path.join(model_dir, "image_gradient_laplace.png"))
        plt.show()

    img = scale_img(output, sidelength, channels, ref)
    img.save(os.path.join(model_dir, "result.png"))


if __name__ == '__main__':
    # get_img(sidelength=256, img_path="./data/knot.jpg", channels=3)
    get_coordinates(256, dim=2)
