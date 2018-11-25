import os
import argparse

import torch
from tqdm import tqdm
import pprint

import model
from model import initialize
from data_loader import fetch_dataloader
from evaluate import evaluate, visualize_sample
from vision_transforms import gen_random_perspective_transform, apply_transform_to_batch
import utils


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--output_dir', help='Directory containing params.json and weights')
parser.add_argument('--restore_file', help='Name of the file containing weights to load')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')


def train_epoch(model, dataloader, loss_fn, optimizer, writer, params, epoch):
    model.train()

    loss_avg = utils.RunningAverage()
    loss_history = []
    best_loss = float('inf')
    vis_counter = 0
    samples = {}
    lrs = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]

    with tqdm(total=len(dataloader), desc='epoch {} of {}. lr: [{:.0e}, {:.0e}]'.format(epoch + 1, params.n_epochs, *lrs)) as pbar:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to gpu if available
            train_batch = train_batch.to(params.device)
            labels_batch = labels_batch.to(params.device)

            P = gen_random_perspective_transform(params)

            transformed_train_batch, scores = model(train_batch, P)

            loss = loss_fn(scores, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # update trackers
            loss_avg.update(loss.item())
            pbar.set_postfix(loss='{:.5f}'.format(loss_avg()))
            pbar.update()

            # write summary
            if i % params.save_summary_steps == 0:
                writer.add_scalar('loss', loss.item(), epoch*(i+1))
                loss_history.append(loss.item())

    return loss_history


def train_and_evaluate(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, writer, params):

    best_loss = float('inf')
    start_epoch = 0

    if params.restore_file:
        print('Restoring parameters from {}'.format(params.restore_file))
        start_epoch = utils.load_checkpoint(params.restore_file, model, optimizer, scheduler, best_loss)
        params.n_epochs += start_epoch - 1
        print('Resuming training from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, params.n_epochs):
        scheduler.step()

        loss_history = train_epoch(model, train_dataloader, loss_fn, optimizer, writer, params, epoch)

        # snapshot at end of epoch
        is_best = sum(loss_history[:1000])/1000 < best_loss
        if is_best: best_loss = sum(loss_history[:1000])/1000
        utils.save_checkpoint({'epoch': epoch + 1,
                               'best_loss': best_loss,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict()},
                               is_best=False,
                               checkpoint=params.output_dir,
                               quiet=True)

        # visualize
        visualize_sample(model, val_dataloader.dataset, writer, params, epoch+1)

        # evalutate and visualize
        val_accuracy = evaluate(model, val_dataloader, writer, params)

    # record val accuracy
    writer.add_scalar('val_accuracy', val_accuracy, epoch+1)


if __name__ == '__main__':
    args = parser.parse_args()

    json_path = os.path.join(args.output_dir, 'params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    params.restore_file = args.restore_file

    # check output folder exist and if it is rel path
    if not os.path.isdir(params.output_dir):
        os.mkdir(params.output_dir)

    writer = utils.set_writer(params.output_dir if args.restore_file is None else os.path.dirname(args.restore_file))

    params.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda else 'cpu')

    # set random seed
    torch.manual_seed(11052018)
    if params.device.type is 'cuda': torch.cuda.manual_seed(11052018)

    # input
    train_dataloader = fetch_dataloader(params, train=True)
    val_dataloader = fetch_dataloader(params, train=False)

    # construct model
    # dims out (pytorch affine grid requires 2x3 matrix output; else perspective transform requires 8)
    model = model.STN(getattr(model, params.stn_module), params).to(params.device)
    # initialize
    initialize(model)
    capacity = sum(p.numel() for p in model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss().to(params.device)
    optimizer = torch.optim.Adam([
        {'params': model.transformer.parameters(), 'lr': params.transformer_lr},
        {'params': model.clf.parameters(), 'lr': params.clf_lr}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step, params.lr_gamma)

    # train and eval
    print('\nStarting training with model (capacity {}):\n'.format(capacity), model)
    print('\nParameters:\n', pprint.pformat(params))
    train_and_evaluate(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, writer, params)

    writer.close()


