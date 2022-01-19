import argparse
import cv2
import numpy as np
import os

import scipy.sparse
import scipy.sparse.linalg

import hparams as hp
from solveFB import solveFB
import time


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./image/woman.png')
    parser.add_argument('--scribble_path', type=str, default='./scribble/woman.png')
    parser.add_argument('--result_dir', type=str, default='./result')

    return parser.parse_args()


def scribble_detection(img, scribble, channels):
    difference = np.sum((scribble - img).reshape((-1, channels)), axis=-1)

    prior_alpha = np.sign(difference) / 2 + 0.5
    alpha_known = prior_alpha != 0.5
    return prior_alpha, alpha_known


def get_gk(img, h, w):
    win_size = hp.win_size
    win_len = hp.win_len
    eps = hp.eps
    sqrt_eps = np.sqrt(eps)
    window = img[h:h + win_len, w:w + win_len, :]
    gk = np.zeros((win_size + 3, 3 + 1))
    gk[:win_size, :3] = window.reshape((-1, 3))
    gk[:win_size, 3] = 1
    gk[win_size:win_size + 3, :3] = np.eye(3) * sqrt_eps
    return gk


def get_gk_bar(gk):
    tmp = gk.T.dot(gk)
    tmp = np.linalg.inv(tmp)
    gk_bar = gk.dot(tmp).dot(gk.T)
    l = gk_bar.shape[0]
    gk_bar = gk_bar - np.eye(l)
    return gk_bar


def get_lk(img, h, w):
    win_size = hp.win_size
    gk = get_gk(img, h, w)
    gk_bar = get_gk_bar(gk)
    lk = gk_bar.T.dot(gk_bar)[:win_size, :win_size]
    return lk


def get_pos(img, h, w, i):
    H, W, C = img.shape
    win_len = hp.win_len
    pos = h * W + w
    pos += (i // win_len) * W
    pos += (i % win_len)
    return pos


def get_Lalpace(img, alpha_known):
    H, W, C = img.shape
    win_len = hp.win_len
    win_size = hp.win_size
    L_map = {}
    for h in range(H - win_len + 1):
        for w in range(W - win_len + 1):
            lk = get_lk(img, h, w)

            for i in range(win_size):
                pos_i = get_pos(img, h, w, i)
                for j in range(win_size):
                    pos_j = get_pos(img, h, w, j)
                    L_map[(pos_i, pos_j)] = L_map.get((pos_i, pos_j), 0) + lk[i, j]

    row = np.array([key[0] for key in L_map.keys()])
    col = np.array([key[1] for key in L_map.keys()])
    data = np.array(list(L_map.values()))

    L = scipy.sparse.coo_matrix((data, (row, col)), shape=(H * W, H * W))
    return L


def solve_Lagrange(L, prior_alpha, alpha_known, l=1000):
    dc = scipy.sparse.diags(alpha_known.astype(np.float64))
    lambda_dc = l * dc
    alpha = scipy.sparse.linalg.spsolve(L + lambda_dc, prior_alpha * lambda_dc)
    return alpha


def main():
    start = time.time()
    args = parser()

    img_name = args.image_path.split('/')[-1][:-4]
    img_dir = os.path.join(args.result_dir, img_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR).astype(np.float64)
    scribble = cv2.imread(args.scribble_path, cv2.IMREAD_COLOR).astype(np.float64)

    H, W, C = img.shape

    prior_alpha, alpha_known = scribble_detection(img=img, scribble=scribble, channels=C)

    cv2.imwrite(os.path.join(img_dir, f"{img_name}_prior_alpha.png"),
                (prior_alpha.reshape((H, W)) * 255).astype(np.uint8))
    print("getting laplace")
    L = get_Lalpace(img, alpha_known)
    print("solving alpha")
    alpha = solve_Lagrange(L, prior_alpha, alpha_known)

    alpha = alpha.reshape((H, W)).clip(0, 1)
    alpha_save = (alpha * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_alpha.png"), alpha_save)
    print("solving f and b")
    f, b = solveFB(img / 255, alpha)
    f_save, b_save = (f.clip(0, 1) * 255).astype(np.uint8), (b.clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_f.png"), f_save)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_b.png"), b_save)

    for c in range(C):
        f[:, :, c] *= alpha
        b[:, :, c] *= 1 - alpha

    f = f.clip(0, 1)
    b = b.clip(0, 1)
    f_save, b_save = (f * 255).astype(np.uint8), (b * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_foreground.png"), f_save)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_background.png"), b_save)

    print(f"done! using {time.time() - start} s")


if __name__ == '__main__':
    main()
