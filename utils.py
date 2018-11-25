import os
import shutil
import json
from datetime import datetime
import torch

from tensorboardX import SummaryWriter




def set_writer(log_dir, comment=''):
    """ setup a tensorboardx summarywriter """
#    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#    log_dir = os.path.join(log_path, current_time + comment)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def save_checkpoint(state, is_best, checkpoint, quiet=False):
    """ saves model and training params at checkpoint + 'last.pt'; if is_best also saves checkpoint + 'best.pt'

    args
        state -- dict; with keys model_state_dict, optimizer_state_dict, epoch, scheduler_state_dict, etc
        is_best -- bool; true if best model seen so far
        checkpoint -- str; folder where params are to be saved
    """

    filepath = os.path.join(checkpoint, 'state_checkpoint.pt')
    if not os.path.exists(checkpoint):
        if not quiet:
            print('Checkpoint directory does not exist Making directory {}'.format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_state_checkpoint.pt'))

    if not quiet:
        print('Checkpoint saved.')


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None, best_metric=None):
    """ loads model state_dict from filepath; if optimizer and lr_scheduler provided also loads them

    args
        checkpoint -- string of filename
        model -- torch nn.Module model
        optimizer -- torch.optim instance to resume from checkpoint
        lr_scheduler -- torch.optim.lr_scheduler instance to resume from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise('File does not exist {}'.format(checkpoint))

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except KeyError:
            print('No optimizer state dict in checkpoint file')

    if best_metric:
        try:
            best_metric = checkpoint['best_val_acc']
        except KeyError:
            print('No best validation accuracy recorded in checkpoint file.')

    if scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
            print('No lr scheduler state dict in checkpoint file')

    return checkpoint['epoch']


# --------------------
# Containers
# --------------------

class RunningAverage:
    """ a class to maintain the running average of a quantity

    example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def __call__(self):
        return self.total/float(self.steps)

    def update(self, val):
        self.steps += 1
        self.total += val



class Params:
    """ class that loads hyperparams from json file.

    example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```
    """

    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)
        self.__dict__['output_dir'] = os.path.dirname(json_path)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """ loads params from json file """
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """ gives dict-like access to Params instances by `params.dict['learning_rate']` """
        return self.__dict__

    def __repr__(self):
        out = ''
        for k, v in self.__dict__.items():
            out += k + ': ' + str(v) + '\n'
        return out

