from torch.utils.data import Dataset, DataLoader
from utils import get_img, get_coordinates, compute_gradient
import scipy
import torch
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class ImageFitting(Dataset):
    def __init__(self, sidelength, img_path, channels):
        super().__init__()
        self.img = get_img(sidelength, img_path)  # C, H, W
        self.img = self.img.permute(1, 2, 0) # H, W, C
        self.coords = get_coordinates(sidelength, dim=2)
        self.pixels = self.img.view(-1, channels) # HW, C

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        in_dict = {'coords': self.coords}
        gt_dict = {'img': self.pixels}

        return in_dict, gt_dict


class PoissonEquation(Dataset):
    def __init__(self, sidelength, img_path):
        super().__init__()
        self.img = get_img(sidelength, img_path)
        self.coords = get_coordinates(sidelength, dim=2)

        self.grads = compute_gradient(self.img, is_color=False)

        self.laplace = scipy.ndimage.laplace(self.img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace).view(-1, 1)

        self.pixels = self.img.permute(1, 2, 0).view(-1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        in_dict = {'coords': self.coords}
        gt_dict = {'img': self.pixels, 'grads': self.grads, 'laplace': self.laplace}

        return in_dict, gt_dict


class CompositeGrads(Dataset):
    def __init__(self, img_path1, img_path2, is_color, sidelength):

        super().__init__()
        self.is_color = is_color
        if self.is_color:
            self.channels = 3
        else:
            self.channels = 1

        self.img1 = Image.open(img_path1)
        self.img2 = Image.open(img_path2)

        self.coords = get_coordinates(sidelength, dim=2)

        if self.is_color:
            self.img1 = self.img1.convert('RGB')
            self.img2 = self.img2.convert('RGB')
        else:
            self.img1 = self.img1.convert('L')
            self.img2 = self.img2.convert('L')

        paddedImg = .85 * torch.ones_like(self.img1)
        paddedImg[:, 512 - 340:512, :] = self.img2
        self.img2 = paddedImg

        self.grads1 = compute_gradient(self.img1, self.is_color)
        self.grads2 = compute_gradient(self.img2, self.is_color)

        self.composite_grads = (.5 * self.grads1 + .5 * self.grads2)

        self.img1 = self.img1.permute(1, 2, 0).view(-1, self.channels)
        self.img2 = self.img2.permute(1, 2, 0).view(-1, self.channels)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        in_dict = {'coords': self.coords}
        gt_dict = {'img1': self.img1, 'img2':self.img2, 'grads': self.composite_grads}

        return in_dict, gt_dict


if __name__ == "__main__":
    img_path = './data/knot.jpg'
    dataset = ImageFitting(256, img_path, channels=3)
    dataloader = DataLoader(dataset, batch_size=1)
    model_input, gt = next(iter(dataloader))
    img = gt['img'].view(256,256,3)
    img = img * 0.5 + 0.5
    img.clip(0, 1)
    img = (img * 255).numpy().astype(np.uint8)


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    plt.show()
