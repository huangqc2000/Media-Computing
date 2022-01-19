import torch
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import os
import scipy.ndimage
import numpy as np

'''from https://github.com/vsitzmann/siren'''


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_gradient(img, channels):
    img_np = img.numpy()
    grads = []
    for c in range(channels):
        gradx = scipy.ndimage.sobel(img_np[:, :, c] * 1e1, axis=0)[..., None]
        grady = scipy.ndimage.sobel(img_np[:, :, c] * 1e1, axis=1)[..., None]
        grads += [torch.from_numpy(gradx), torch.from_numpy(grady)]

    grads = torch.cat(grads, dim=-1)  # H, W, 2C
    grads = grads.reshape(-1, 2 * channels)  # HW, 2C
    return grads


def compute_laplace(img, channels):
    img_np = img.numpy()

    lapls = []
    for c in range(channels):
        lapl = scipy.ndimage.laplace(img_np * 1e4)[..., None]
        lapls += [torch.from_numpy(lapl)]

    lapls = torch.cat(lapls, dim=-1)  # H, W, C
    lapls = lapls.reshape(-1, channels)  # HW, C
    return lapls


def compute_gradient_laplace(img, channels):
    return compute_gradient(img, channels), compute_laplace(img, channels)


def get_coordinates(len, dim):
    lens = dim * (len,)
    linespace = [torch.linspace(-1, 1, len) for l in lens]
    grids = torch.meshgrid(*linespace)
    coords = torch.stack(grids, dim=-1)
    coords = coords.view(-1, dim)
    return coords


def get_img(sidelength, img_path, channels):
    img = Image.open(img_path).resize((sidelength, sidelength))
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    img = img.permute(1, 2, 0)
    return img


def scale_img(output, sidelength, channels, ref):
    img = output.view(sidelength, sidelength, channels).detach().cpu()
    if ref is not None:
        rmax, rmin = ref.max(), ref.min()
        max, min = img.max(), img.min()
        img = (img - min) / (max - min) * (rmax - rmin) + rmin

    img = img * 0.5 + 0.5
    img = (img.clip(0, 1) * 255).numpy().astype(np.uint8)
    if channels == 1:
        mode = 'L'
        img = img.squeeze(-1)
    else:
        mode = 'RGB'
    img = Image.fromarray(img, mode=mode)
    return img


def save_result(model_output, sidelength, channels, model_dir, reference=None, compute_grad=False):
    output, coords = model_output['model_out'], model_output['model_in']

    if channels == 1 and compute_grad:
        result = output.reshape(sidelength, sidelength).detach().cpu().numpy()
        grad = gradient(output, coords)
        grad = grad.norm(dim=-1).reshape(sidelength, sidelength).detach().cpu().numpy()
        laplacian = laplace(output, coords)
        laplacian = laplacian.reshape(sidelength, sidelength).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(result)
        axes[0].set_xlabel("image")
        axes[1].imshow(grad)
        axes[1].set_xlabel("gradient")
        axes[2].imshow(laplacian)
        axes[2].set_xlabel("laplace")
        plt.savefig(os.path.join(model_dir, "image_gradient_laplace.png"))
        plt.show()

    img = scale_img(output, sidelength, channels, reference)
    img.save(os.path.join(model_dir, "result.png"))
