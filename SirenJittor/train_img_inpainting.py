import jittor as jt
from jittor import nn
from dataset import get_img_inpainting_dataset
from model.Siren import Siren
from utils import save_result
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
    parser.add_argument('--img_path', type=str, default='./data/human.jpg')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--sample_rate', type=float, default=None)
    parser.add_argument('--points_choose', type=int, default=None)
    parser.add_argument('--crop_half', action='store_true', default=None)
    parser.add_argument('--compute_grad', action='store_true', default=False)

    return parser.parse_args()


def img_mse(model_output, gt):
    return ((model_output['model_out'] - gt['img']) ** 2).mean()


def main():
    args = parse()

    dataset = get_img_inpainting_dataset(args.img_path, args.sidelength, args.channels, sample_rate=args.sample_rate,
                                         crop_half=args.crop_half, points_choose=args.points_choose)

    model = Siren(in_features=2, out_features=args.channels, hidden_features=256, hidden_layers=3)

    print(model)

    optimizer = nn.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = img_mse
    print("Define Optimizer and Loss Function.")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_name = args.img_path.split('/')[-1][:-4]
    model_dir = os.path.join(os.path.join(args.result_dir, "image_inpainting"), img_name)

    print("-------Start Training-------")
    train(model, optimizer, dataset=dataset, num_epochs=args.num_epochs, loss_fn=loss_fn,
          print_every=args.print_every, model_dir=model_dir, lr_schedule=args.lr_schedule)

    model.load_state_dict(jt.load(os.path.join(model_dir, "model_final.pkl")))
    model_input = {'coords': dataset['all_coords']}
    model_output = model(model_input)

    save_result(model_output=model_output, sidelength=args.sidelength, channels=args.channels, model_dir=model_dir,
                compute_grad=args.compute_grad)

    mask_img = dataset['mask_img']
    mask_img.save(os.path.join(model_dir, "mask_img.png"))

    # out_of_range_test(model=model, sidelength=512, model_dir=model_dir)


if __name__ == "__main__":
    main()
