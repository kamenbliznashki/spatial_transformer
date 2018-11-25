import math
import copy

import torch
import torch.nn.functional as F


def vec_to_perpective_matrix(vec):
    # vec rep of the perspective transform has 8 dof; so add 1 for the bottom right of the perspective matrix;
    # note network is initialized to transformer layer bias = [1, 0, 0, 0, 1, 0] so no need to add an identity matrix here
    out = torch.cat((vec, torch.ones((vec.shape[0],1), dtype=vec.dtype, device=vec.device)), dim=1).reshape(vec.shape[0], -1)
    return out.view(-1,3,3)


def gen_random_perspective_transform(params):
    """ generate a batch of 3x3 homography matrices by composing rotation, translation, shear, and projection matrices, 
    where each samples components from a uniform(-1,1) * multiplicative_factor
    """

    batch_size = params.batch_size

    # debugging
    if params.dict.get('identity_transform_only'):
        return torch.eye(3).repeat(batch_size, 1, 1).to(params.device)


    I = torch.eye(3).repeat(batch_size, 1, 1)
    uniform = torch.distributions.Uniform(-1,1)
    factor = 0.25
    c = copy.deepcopy

    # rotation component
    a = math.pi / 6 * uniform.sample((batch_size,))
    R = c(I)
    R[:, 0, 0] = torch.cos(a)
    R[:, 0, 1] = - torch.sin(a)
    R[:, 1, 0] = torch.sin(a)
    R[:, 1, 1] = torch.cos(a)
    R.to(params.device)

    # translation component
    tx = factor * uniform.sample((batch_size,))
    ty = factor * uniform.sample((batch_size,))
    T = c(I)
    T[:, 0, 2] = tx
    T[:, 1, 2] = ty
    T.to(params.device)

    # shear component
    sx = factor * uniform.sample((batch_size,))
    sy = factor * uniform.sample((batch_size,))
    A = c(I)
    A[:, 0, 1] = sx
    A[:, 1, 0] = sy
    A.to(params.device)

    # projective component
    px = uniform.sample((batch_size,))
    py = uniform.sample((batch_size,))
    P = c(I)
    P[:, 2, 0] = px
    P[:, 2, 1] = py
    P.to(params.device)

    # compose the homography
    H = R @ T @ P @ A

    return H


def apply_transform_to_batch(im_batch_tensor, transform_tensor):
    """ apply a geometric transform to a batch of image tensors
    args
        im_batch_tensor -- torch float tensor of shape (N, C, H, W)
        transform_tensor -- torch float tensor of shape (1, 3, 3)

    returns
        transformed_batch_tensor -- torch float tensor of shape (N, C, H, W)
    """
    N, C, H, W = im_batch_tensor.shape
    device = im_batch_tensor.device

    # torch.nn.functional.grid_sample takes a grid in [-1,1] and interpolates;
    # construct grid in homogeneous coordinates
    x, y = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
    x, y = x.flatten(), y.flatten()
    xy_hom = torch.stack([x, y, torch.ones(x.shape[0])], dim=0).unsqueeze(0).to(device)

    # tansform the [-1,1] homogeneous coords
    xy_transformed = transform_tensor.matmul(xy_hom)  # (N, 3, 3) matmul (N, 3, H*W) > (N, 3, H*W)
    # convert to inhomogeneous coords -- cf Szeliski eq. 2.21

    grid = xy_transformed[:,:2,:] / (xy_transformed[:,2,:].unsqueeze(1) + 1e-9)
    grid = grid.permute(0,2,1).reshape(-1, H, W, 2)  # (N, H, W, 2); cf torch.functional.grid_sample
    grid = grid.expand(N, *grid.shape[1:])  # expand to minibatch

    transformed_batch = F.grid_sample(im_batch_tensor, grid, mode='bilinear')
    transformed_batch.transpose_(3,2)

    return transformed_batch




# --------------------
# Test
# --------------------

def test_get_random_perspective_transform():
    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    import matplotlib.pyplot as plt
    from unittest.mock import Mock

    np.random.seed(6)

    im = np.zeros((30,30))
    im[10:20,10:20] = 1
    im[20,20] = 1

    imt = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  18,
              18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170, 253,
             253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253, 253,
             253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253, 253,
             198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253, 205,
              11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,  90,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253, 190,
               2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190, 253,
              70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35, 241,
             225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  81,
             240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39, 148,
             229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221, 253,
             253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253, 253,
             253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253, 195,
              80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,  11,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]])



    # get transform
    params = Mock()
    params.batch_size = 1
    params.dict = {'identity_transform_only': False}
    params.device = torch.device('cpu')
    H = gen_random_perspective_transform(params)

    im = im[np.newaxis, np.newaxis, ...]
    im = torch.FloatTensor(im)
    im_transformed = apply_transform_to_batch(im, H)

    imt = imt[np.newaxis, np.newaxis, ...]
    imt = torch.FloatTensor(imt)
    imt_transformed = apply_transform_to_batch(imt, H)

    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(im.squeeze().numpy(), cmap='gray')
    axs[0,1].imshow(im_transformed.squeeze().numpy(), cmap='gray')

    axs[1,0].imshow(imt.squeeze().numpy(), cmap='gray')
    axs[1,1].imshow(imt_transformed.squeeze().numpy(), cmap='gray')

    for ax in plt.gcf().axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/transform_test.png')
    plt.close()


if __name__ == '__main__':
    test_get_random_perspective_transform()


