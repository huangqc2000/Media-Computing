from dataset import PoissonEquation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
dataset = PoissonEquation(256)
dataloader = DataLoader(dataset, batch_size=1)
model_input, gt = next(iter(dataloader))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(gt['img'].view(256, 256))
axes[1].imshow(gt['grads'].norm(dim=-1).view(256, 256))
axes[2].imshow(gt['laplace'].view(256, 256))
plt.savefig("picture.png")
plt.show()