from tensorboardX import SummaryWriter
import numpy as np

result_dir = './result/image_fit/'
writer = SummaryWriter('./result/loss_curve')

if __name__ == '__main__':
    end_epoch = 1000
    sine_loss = np.loadtxt(result_dir + 'sine/' + 'train_loss.txt')
    relu_loss = np.loadtxt(result_dir + 'relu/' + 'train_loss.txt')
    tanh_loss = np.loadtxt(result_dir + 'tanh/' + 'train_loss.txt')
    sigmoid_loss = np.loadtxt(result_dir + 'sigmoid/' + 'train_loss.txt')
    for epoch in range(end_epoch):
        writer.add_scalars("loss", {'sine': sine_loss[epoch], 'relu': relu_loss[epoch], 'tanh': tanh_loss[epoch],
                                    'sigmoid': sigmoid_loss[epoch]}, epoch)
