import jittor as jt
from jittor import nn
from dataset import get_img_fitting_dataset
from model.Siren import Siren
from model.MLP import MLP
from utils import get_coordinates, save_result, scale_img
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
    parser.add_argument('--sidelength', type=int, default=256)
    parser.add_argument('--lr_schedule', type=int, default=None)
    parser.add_argument('--img_path', type=str, default='./data/camera.jpg')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--compute_grad', action='store_true', default=True)

    return parser.parse_args()


def out_of_range_test(model, sidelength, channels, model_dir):
    coords = get_coordinates(sidelength, 2) * 2
    model_input = {'coords': coords}
    model_output = model(model_input)
    output, coords = model_output['model_out'], model_output['model_in']

    img = scale_img(output, sidelength, channels, ref=None)
    img.save(os.path.join(model_dir, "out_of_range.png"))


def img_mse(model_output, gt):
    return ((model_output['model_out'] - gt['img']) ** 2).mean()


def main():
    args = parse()

    dataset = get_img_fitting_dataset(args.img_path, args.sidelength, args.channels)

    if args.nonlinearity == 'sine':
        model = Siren(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3)
    else:
        model = MLP(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3,
                    nonlinearity=args.nonlinearity)

    print(f"Define model, use nonlinearity {args.nonlinearity}")
    print(model)

    optimizer = nn.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = img_mse
    print("Define Optimizer and Loss Function.")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(os.path.join(args.result_dir, "image_fit"), args.nonlinearity)

    print("-------Start Training-------")
    train(model, optimizer, dataset=dataset, num_epochs=args.num_epochs, loss_fn=loss_fn,
          print_every=args.print_every, model_dir=model_dir, lr_schedule=args.lr_schedule)

    model.load_state_dict(jt.load(os.path.join(model_dir, "model_final.pkl")))
    model_input = {'coords': dataset['coords']}
    model_output = model(model_input)

    save_result(model_output=model_output, sidelength=args.sidelength, channels=args.channels, model_dir=model_dir,
                compute_grad=args.compute_grad)

    if args.nonlinearity == 'sine':
        out_of_range_test(model=model, sidelength=args.sidelength * 2, channels=args.channels, model_dir=model_dir)


if __name__ == "__main__":
    main()
