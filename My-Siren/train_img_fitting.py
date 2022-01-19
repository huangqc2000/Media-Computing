import torch
from dataset import get_img_fitting_dataset
from torch.utils.data import DataLoader
from model.Siren import Siren
from model.MLP import MLP
import matplotlib.pyplot as plt
from utils import get_coordinates, save_result
from training import train
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--nonlinearity', type=str, default='sine')
    parser.add_argument('--sidelength', type=int, default=256)
    parser.add_argument('--lr_schedule', type=int, default=None)
    parser.add_argument('--img_path', type=str, default='./data/camera.jpg')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--compute_grad', action='store_true', default=False)

    return parser.parse_args()


def out_of_range_test(model, sidelength, model_dir):
    coords = get_coordinates(sidelength, 2) * 2
    model_output, coords = model(coords)

    result = model_output.view(sidelength, sidelength).detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.savefig(os.path.join(model_dir, "out_of_range.png"))


def img_mse(model_output, gt):
    return ((model_output['model_out'] - gt['img']) ** 2).mean()


def main():
    args = parse()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Train on the {device}")

    dataset = get_img_fitting_dataset(args.img_path, args.sidelength, args.channels, device=device)

    if args.nonlinearity == 'sine':
        model = Siren(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3)
    else:
        model = MLP(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3,
                    nonlinearity=args.nonlinearity)

    print(f"Define model, use nonlinearity {args.nonlinearity}")
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = img_mse
    print("Define Optimizer and Loss Function.")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(os.path.join(args.result_dir, "image_fit"), args.nonlinearity)

    print("-------Start Training-------")
    train(model, optimizer, dataset=dataset, num_epochs=args.num_epochs, loss_fn=loss_fn,
          print_every=args.print_every, model_dir=model_dir, lr_schedule=args.lr_schedule)

    model.load_state_dict(torch.load(os.path.join(model_dir, "model_final.pth")))
    model_input = {'coords':dataset['coords']}
    model_output = model(model_input)

    save_result(model_output=model_output, sidelength=args.sidelength, channels=args.channels, model_dir=model_dir, compute_grad=args.compute_grad)

    # out_of_range_test(model=model, sidelength=512, model_dir=model_dir)


if __name__ == "__main__":
    main()
