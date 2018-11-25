import os
import argparse
import pprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import model
from data_loader import fetch_dataloader
from vision_transforms import gen_random_perspective_transform, apply_transform_to_batch
import utils


parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('--output_dir', help='Directory containing params.json and weights')
parser.add_argument('--restore_file', help='Name of the file in containing weights to load')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')


@torch.no_grad()
def visualize_sample(model, dataset, writer, params, step, n_samples=20):
    model.eval()

    sample = torch.stack([dataset[i][0] for i in range(n_samples)], dim=0).to(params.device)

    P = gen_random_perspective_transform(params)[:n_samples]
    perturbed_sample = apply_transform_to_batch(sample, P)
    transformed_sample, scores = model(sample, P)

    perturbed_sample = perturbed_sample.view(n_samples, 1, 28, 28)
    transformed_sample = transformed_sample.view(n_samples, 1, 28, 28)

    sample = torch.cat([sample, perturbed_sample, transformed_sample], dim=0)
    sample = make_grid(sample.cpu(), nrow=n_samples, normalize=True, padding=1, pad_value=1)

    if writer:
        writer.add_image('sample', sample, step)

    save_image(sample, os.path.join(params.output_dir, 'samples__orig_perturbed_transformed' + (step!=None)*'_step_{}'.format(step) + '.png'))


@torch.no_grad()
def evaluate(model, dataloader, writer, params):
    model.eval()

    # init trackers
    accuracy = []
    labels = []
    original = []
    perturbed = []
    transformed = []

    with tqdm(total=len(dataloader), desc='eval') as pbar:
        for i, (im_batch, labels_batch) in enumerate(dataloader):
            im_batch = im_batch.to(params.device)

            # get a random transformation and run through the batch
            P = gen_random_perspective_transform(params)

            transformed_batch, scores = model(im_batch, P)
            log_probs = F.log_softmax(scores, dim=1)

            # get predictions and calculate accuracy
            _, pred = torch.max(log_probs.cpu(), dim=1)
            accuracy.append(pred.eq(labels_batch.view_as(pred)).sum().item() / im_batch.shape[0])


            # record to compute mean image with variance for original, perturbed, and transformed image (cf Lin, Lucey ICSTN paper)
            labels.append(labels_batch)
            original.append(im_batch)
            perturbed.append(apply_transform_to_batch(im_batch, P))
            transformed.append(transformed_batch)

            avg_accuracy = sum(accuracy) / len(accuracy)
            pbar.set_postfix(accuracy='{:.5f}'.format(avg_accuracy))
            pbar.update()

    labels = torch.cat(labels, dim=0)
    unique_labels = torch.unique(labels, sorted=True)
    original = torch.cat(original, dim=0)
    perturbed = torch.cat(perturbed, dim=0)
    transformed = torch.cat(transformed, dim=0)

    # compute mean image with variance for original, perturbed, and transformed image for each digit (cf Lin, Lucey ICSTN paper)
    image = torch.stack([original, perturbed, transformed], dim=0)  # (3, len(data), C, H, W)
    mean_image = [make_grid(torch.mean(image[:, labels==i, ...], dim=1).cpu(), nrow=1) for i in unique_labels]
    var_image = [make_grid(torch.var(image[:, labels==i, ...], dim=1).cpu(), nrow=1) for i in unique_labels]
    var_image = make_grid(var_image, nrow=len(unique_labels))

    # save mean and var image
    save_image(mean_image, os.path.join(params.output_dir, 'test_image_mean.png'), nrow=len(unique_labels))
    save_image(var_image, os.path.join(params.output_dir, 'test_image_var.png'), nrow=len(unique_labels), normalize=True)

    # save accuracy
    with open(os.path.join(params.output_dir, 'eval_accuracy.txt'), 'w') as f:
        f.write('Mean evaluation accuracy {:.3f}'.format(avg_accuracy))

    return avg_accuracy




if __name__ == '__main__':
    args = parser.parse_args()

    # load params
    json_path = os.path.join(args.output_dir, 'params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    # check output folder exist and if it is rel path
    if not os.path.isdir(params.output_dir):
        os.mkdir(params.output_dir)

    writer = utils.set_writer(params.output_dir)

    params.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda else 'cpu')

    # set random seed
    torch.manual_seed(11052018)
    if params.device.type is 'cuda': torch.cuda.manual_seed(11052018)

    # input
    dataloader = fetch_dataloader(params, train=False)

    # load model
    model = model.STN(getattr(model, params.stn_module), params).to(params.device)
    utils.load_checkpoint(args.restore_file, model)

    # run inference
    print('\nEvaluating with model:\n', model)
    print('\n.. and parameters:\n', pprint.pformat(params))
    accuracy = evaluate(model, dataloader, writer, params)
    visualize_sample(model, dataloader.dataset, writer, params, None)
    print('Evaluation accuracy: {:.5f}'.format(accuracy))

    writer.close()
