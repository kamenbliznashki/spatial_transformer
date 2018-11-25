""" Perform hyperparameter search """

import os
import sys
import json
import argparse
from copy import deepcopy
from subprocess import check_call

import torch
import utils


PYTHON = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments', help='Directory containing hyperparams.json to setup a model.')
parser.add_argument('--data_dir', default='./data', help='Directory containing the dataset')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')


def launch_training_job(parent_dir, data_dir, job_name, params):
    """ launch training of the model with a set of hyperparameters in parent_dir/job_name """

    # create new filder in parent_dir with unique name 'job_name'
    output_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # write params in a json file
    json_path = os.path.join(output_dir, 'params.json')
    params.save(json_path)

    print('Launching training job with parameters:')
    print(params)

    # launch training with this config
    if params.device is 'cpu':
        cmd = '{python} train.py --output_dir={output_dir}'.format(
                python=PYTHON, output_dir=output_dir)
    else:
        cmd = '{python} train.py --output_dir={output_dir} --cuda={device}'.format(
                python=PYTHON, output_dir=output_dir, device=int(params.device.split(':')[1]))


    print(cmd)

    check_call(cmd, shell=True)


if __name__ == '__main__':
    # load the references parameters from parent_dir json file
    args = parser.parse_args()

    json_path = os.path.join(args.parent_dir, 'hyperparams.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    hyperparams = utils.Params(json_path)

    json_path = os.path.join(args.parent_dir, 'base_params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    base_params = utils.Params(json_path)

    # set the static parameters
    for param, values in hyperparams.dict.items():
        if isinstance(values, list):
            continue
        base_params.dict[param] = values

    base_params.device = 'cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda else 'cpu'

    # loop through the hyperparameter lists
    for param, values in hyperparams.dict.items():
        if isinstance(values, list):
            for v in values:
                params = deepcopy(base_params)
                # modify the parameter value to that in hyperparms
                params.dict[param] = v

                # launch job with unique name
                job_name = '{}_{}'.format(param, v)
                launch_training_job(args.parent_dir, args.data_dir, job_name, params)

