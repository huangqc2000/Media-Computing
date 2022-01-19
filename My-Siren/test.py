import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import skimage

if __name__ == "__main__":
    img = Image.open("./data/camera.jpg")
    img = img.resize((256, 256))
    img.save("./data/camera.jpg")