import numpy as np
import scipy.sparse as sp
import cv2
import scipy.sparse.linalg


def get_grad(mask):
    H, W = mask.shape
    h_row, h_col = [], []
    v_row, v_col = [], []
    dm_row, dm_col = [], []
    ds_row, ds_col = [], []

    for i in range(H):
        for j in range(W - 1):
            if mask[i][j] or mask[i][j + 1]:
                h_row.append(i)
                h_col.append(j)

    h_left = np.ravel_multi_index((h_row, h_col), mask.shape)
    h_right = h_left + 1

    for i in range(H - 1):
        for j in range(W):
            if mask[i][j] or mask[i + 1][j]:
                v_row.append(i)
                v_col.append(j)

    v_top = np.ravel_multi_index((v_row, v_col), mask.shape)
    v_bottom = v_top + W

    for i in range(H - 1):
        for j in range(W - 1):
            if mask[i][j] or mask[i + 1][j + 1]:
                dm_row.append(i)
                dm_col.append(j)

    dm1 = np.ravel_multi_index((dm_row, dm_col), mask.shape)
    dm2 = dm1 + W + 1

    for i in range(H - 1):
        for j in range(1, W):
            if mask[i][j] or mask[i + 1][j - 1]:
                ds_row.append(i)
                ds_col.append(j)

    ds1 = np.ravel_multi_index((ds_row, ds_col), mask.shape)
    ds2 = ds1 + W - 1

    indices = np.stack((
        np.concatenate((h_left, v_top, dm1, ds1)),
        np.concatenate((h_right, v_bottom, dm2, ds2))
    ), axis=-1)  # [4HW, 2]

    len = indices.shape[0]
    row_nz = np.arange(indices.size) // 2
    col_nz = indices.flatten()

    return sp.coo_matrix((np.tile([-1, 1], len), (row_nz, col_nz)),
                         shape=(len, mask.size)), sp.coo_matrix(
        (np.tile([0, 1], len), (row_nz, col_nz)), shape=(len, mask.size))


def get_conditions(alpha_f):
    return (alpha_f < 0.02) * 100.0 + 0.03 * (1.0 - alpha_f) * (alpha_f < 0.3) + 0.01 * (alpha_f > 1.0 - 0.02)


def solveFB(image, alpha):
    H, W, C = image.shape
    alpha_f = alpha.flatten()
    mask = np.logical_and(alpha > 0.02, alpha < 0.98)  # H, W

    grad, grad_positive = get_grad(mask)  # 4HW, HW

    ga = grad.dot(alpha_f)  # 4HW
    grad_f_weight = np.sqrt(np.abs(ga)) + 0.003 * grad_positive.dot((1. - alpha_f))  # 4HW
    grad_b_weight = np.sqrt(np.abs(ga)) + 0.003 * grad_positive.dot(alpha_f)  # 4HW

    condition_f = get_conditions(1.0 - alpha_f)  # HW
    condition_b = get_conditions(alpha_f)  # HW
    condition_f_sp = sp.diags(condition_f)  # HW HW
    condition_b_sp = sp.diags(condition_b)  # HW HW

    bi = image.reshape(H * W, C)  # HW, C
    bf = (condition_f * (alpha_f > 0.02)).reshape((-1, 1)) * bi  # HW, C
    bb = (condition_b * (alpha_f < 0.98)).reshape((-1, 1)) * bi  # HW, C

    b = np.concatenate((bi, bf, bb))  # 3HW, C
    a = sp.vstack((  # 3HW, 2HW
        sp.hstack((  # HW, 2HW
            sp.diags(alpha_f),
            sp.diags(1.0 - alpha_f)
        )),
        sp.hstack((  # HW, 2HW
            condition_f_sp,
            sp.coo_matrix(condition_f_sp.shape)
        )),
        sp.hstack((  # HW, 2HW
            sp.coo_matrix(condition_b_sp.shape),
            condition_b_sp
        ))
    ))
    rhs = a.transpose().dot(b)  # 2HW, C
    a = sp.vstack((  # 11HW, 2HW
        a,
        sp.vstack((  # 8HW, 2HW
            scipy.sparse.hstack((
                sp.diags(grad_f_weight).dot(grad),
                sp.coo_matrix(grad.shape)
            )),
            scipy.sparse.hstack((
                sp.coo_matrix(grad.shape),
                sp.diags(grad_b_weight).dot(grad)
            ))
        ))
    ))
    lhs = a.transpose().dot(a)

    solution = scipy.sparse.linalg.spsolve(lhs, rhs)  # 2HW, 3
    f, b = solution[:H * W, :].reshape((H, W, C)), solution[H * W:].reshape((H, W, C))

    # f = f.clip(0, 1)
    # b = b.clip(0, 1)

    return f, b


if __name__ == '__main__':
    image_path = './image/dabdelion.png'
    alpha_path = './result/dandelion/dandelion_alpha.png'

    image = cv2.imread(image_path) / 255.0
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE) / 255.0

    foreground, background = solveFB(image, alpha)

    cv2.imwrite('foreground.png', foreground * 255.0)
    cv2.imwrite('background.png', background * 255.0)
