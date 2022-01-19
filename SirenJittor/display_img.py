import matplotlib.pyplot as plt
from dataset import get_poisson_equation_dataset

if __name__ == '__main__':
    img_path = './data/camera.jpg'
    img_name = 'camera'
    sidelength,channels = 256, 1
    dataset = get_poisson_equation_dataset(img_path=img_path, sidelength=sidelength, channels=channels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(dataset['img'].reshape(sidelength, sidelength).detach().numpy())
    axes[0].set_xlabel("image")
    axes[1].imshow(dataset['gradient'].norm(dim=-1).reshape(sidelength, sidelength).detach().numpy())
    axes[1].set_xlabel("gradient")
    axes[2].imshow(dataset['laplace'].reshape(sidelength, sidelength).detach().numpy())
    axes[2].set_xlabel("laplace")
    plt.savefig( f"./data/{img_name}_image_gradient_laplace.png")
    plt.show()

