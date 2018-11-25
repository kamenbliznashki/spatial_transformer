import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transforms import apply_transform_to_batch, vec_to_perpective_matrix


# --------------------
# Model helpers
# --------------------

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],-1)

def initialize(model, std=0.1):
    for p in model.parameters():
        p.data.normal_(0,std)

    # init last linear layer of the transformer at 0
    model.transformer.net[-1].weight.data.zero_()
    model.transformer.net[-1].bias.data.copy_(torch.eye(3).flatten()[:model.transformer.net[-1].out_features])
    # NOTE: this initialization the last layer of the transformer layer to identity here means the apply_tranform function should not
    #       add an identity matrix when converting coordinates


# --------------------
# Model components
# --------------------

class BasicSTNModule(nn.Module):
    """ pytorch builtin affine transform """
    def __init__(self, params, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 4, kernel_size=7),        # (N, 1, 28, 28) > (N, 4, 22, 22)
                                 nn.ReLU(True),
                                 nn.Conv2d(4, 8, kernel_size=7),        # (N, 4, 20, 20) > (N, 8, 16, 16)
                                 nn.MaxPool2d(2, stride=2),             # (N, 8, 18, 18) > (N, 8, 8, 8)
                                 nn.ReLU(True),
                                 Flatten(),
                                 nn.Linear(8**3, 48),
                                 nn.ReLU(True),
                                 nn.Linear(48, out_dim))

    def forward(self, x, P_init):
        x = apply_transform_to_batch(x, P_init)
        theta = self.net(x).view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)


class STNModule(BasicSTNModule):
    """ homography stn """
    def __init__(self, params, out_dim=8):
        super().__init__(params, out_dim)

    def forward(self, x, P_init):
        # apply the perturbation matrix to the minibatch of image tensors
        x = apply_transform_to_batch(x, P_init)
        # predict the transformation to approximate
        p = self.net(x)
        # convert to matrix
        P_net = vec_to_perpective_matrix(p)
        # apply to the original image
        return apply_transform_to_batch(x, P_net)


class ICSTNModule(STNModule):
    """ inverse compositional stn cf Lin, Lucey ICSTN paper """
    def __init__(self, params):
        super().__init__(params)
        self.icstn_steps = params.icstn_steps

    def forward(self, x, P_init):
        P = P_init
        # apply spatial transform recurrently for n_steps
        for i in range(self.icstn_steps):
            # apply the perturbation matrix to the minibatch of image tensors
            transformed_x = apply_transform_to_batch(x, P)
            # predict the trasnform
            p = self.net(transformed_x)
            # convert to matrix
            P_net = vec_to_perpective_matrix(p)
            # compose transform with previous
            P = P @ P_net  # compose on the left; apply_transform_to_batch takes the composite transform and right multiplies by xy_hom
        # apply the final composite transform to the original image
        return apply_transform_to_batch(x, P)


class ClassifierModule(nn.Module):
    def __init__(self, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 3, kernel_size=9),        # (N, 1, 28, 28) > (N, 3, 20, 20)
                                 nn.ReLU(True),
                                 Flatten(),
                                 nn.Linear(3*20*20, out_dim))

    def forward(self, x):
        return self.net(x)


# --------------------
# Model
# --------------------

class STN(nn.Module):
    def __init__(self, transformer_module, params):
        super().__init__()
        self.transformer = transformer_module(params)
        self.clf = ClassifierModule()

    def forward(self, x, P):
        # take minibatch of image tensors x and geometric transform P
        x = self.transformer(x, P)
        # return the output of the transformer and the output of the classifier
        return x, self.clf(x)



