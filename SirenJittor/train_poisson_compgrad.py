import jittor as jt
from jittor import nn
from dataset import get_composite_dataset
from model.Siren import Siren
from utils import gradient, save_result
from training import train
import os
import argparse

jt.flags.use_cuda = 1  # jt.flags.use_cuda 表示是否使用 gpu 训练。


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--nonlinearity', type=str, default='sine')
    parser.add_argument('--sidelength', type=int, default=512)
    parser.add_argument('--lr_schedule', type=int, default=None)
    parser.add_argument('--img_path1', type=str, default='./data/bear.jpg')
    parser.add_argument('--img_path2', type=str, default='./data/gizeh.jpg')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--compute_grad', action='store_true', default=False)

    return parser.parse_args()


def img_grads_mse(model_output, gt):
    channels = gt['img1'].shape[-1]
    if channels == 1:
        gradients = jt.grad(model_output['model_out'], model_output['model_in'])
    else:
        gradients = []
        for c in range(channels):
            grads_channel = jt.grad(model_output['model_out'][..., c], model_output['model_in'])
            gradients += [grads_channel]
        gradients = jt.concat(gradients, dim=-1)
    return ((gradients - gt['gradient']).pow(2).sum(-1)).mean()


def main():
    args = parse()

    dataset = get_composite_dataset(img_path1=args.img_path1, img_path2=args.img_path2, sidelength=args.sidelength,
                                    channels=args.channels)

    model = Siren(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3)

    print(f"Define model, use nonlinearity sine")
    print(model)

    optimizer = nn.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = img_grads_mse
    print("Define Optimizer and Loss Function.")

    model_dir = os.path.join(args.result_dir, "composite_grad")

    print("-------Start Training-------")
    train(model, optimizer, dataset=dataset, num_epochs=args.num_epochs, loss_fn=loss_fn,
          print_every=args.print_every, model_dir=model_dir, lr_schedule=args.lr_schedule)

    model.load_state_dict(jt.load(os.path.join(model_dir, "model_final.pkl")))
    model_input = {'coords': dataset['coords']}
    model_output = model(model_input)

    save_result(model_output=model_output, sidelength=args.sidelength, channels=args.channels, model_dir=model_dir,
                ref=dataset['img1'], compute_grad=args.compute_grad)


if __name__ == "__main__":
    main()
