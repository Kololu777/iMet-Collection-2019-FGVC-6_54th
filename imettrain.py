import os
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import RandomSampler
from tensorboardX import SummaryWriter
from config import model_name, save_weight, cuda, fine_size, optim, all_epoch, log_dir, batch_size, lr, momentum, weight_decay, weight_name, valid_set
from losses.loss_factory import select_criterion
from imet_dataset import iMetDataset
from utils.imetutils import path, data_path
from models.SEResNextmodel.SEResNexthead import GAPseResNext50, GAPseResNext101
from optim import select_optim
if not os.path.isdir(save_weight):
    os.mkdir(save_weight)
    assert os.path.isdir(save_weight)


device = torch.device('cuda:0' if cuda else 'cpu')


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == 1103
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def get_score(y_pred, truths):
    return fbeta_score(truths, y_pred, beta=2, average='samples')


def test(test_loader, model, criterion):
    running_loss = 0.0
    model.eval()
    for count, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            N_outputs = torch.sigmoid(outputs).detach().cpu().numpy()
            outputs = torch.sigmoid(outputs).detach().cpu().numpy() > 0.2

        if count == 0:
            predicts = outputs
            N_predicts = N_outputs
            truths = labels.cpu().numpy()
        else:
            predicts = np.vstack((predicts, outputs))
            truths = np.vstack((truths, labels.cpu().numpy()))
            N_predicts = np.vstack((N_predicts, N_outputs))

        running_loss += loss.item() * inputs.size(0)

    predicts = predicts.astype(np.int)
    truths = truths.astype(np.int)
    epoch_loss = running_loss / val_data.__len__()
    metrics = {}
    argsorted = N_predicts.argsort(axis=1)
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(N_predicts, threshold, argsorted), truths)
    metrics['valid_loss'] = epoch_loss
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    precision = fbeta_score(truths, predicts, beta=2, average="samples")

    return epoch_loss, precision


def train(train_loader, model, optimizer, criterion):
    running_loss = 0.0

    model.train()
    for count, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            logit = model(inputs)
            loss = criterion(logit, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # mb.child.comment = 'train_loss: {}'.format(loss.item())
        epoch_loss = running_loss / train_data.__len__()
    return epoch_loss


if __name__ == '__main__':
    writer1 = SummaryWriter(log_dir)
    criterion = select_criterion('comb')
    if model_name == "ResNet50":
        model = GAPseResNext50()
    elif model_name == "ResNet101":
        model = GAPseResNext101()
    model = nn.DataParallel(model)

    restart_epoch = 0

    cudnn.benchmark = True
    model = model.to(device)

    df = pd.read_csv(data_path + '/imetfold/' + 'folds.csv')

    best_acc = 0

    optimizer = select_optim(optim, model, lr, weight_decay, momentum=momentum)

    def func(lr_epoch):
        if lr_epoch <= 18:
            return 1
        elif lr_epoch > 18:
            return 0.2

    lr_epoch = 0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

    #valid_set = 3
    train_df = df[df['fold'] != valid_set]
    valid_df = df[df['fold'] == valid_set]

    X_train = train_df['id'].values
    y_train = train_df['attribute_ids']
    y_train = [[int(i) for i in s.split()] for s in y_train]
    print(X_train.shape, len(y_train))

    X_val = valid_df['id'].values
    y_val = valid_df['attribute_ids']
    y_val = [[int(i) for i in s.split()] for s in y_val]
    print(X_val.shape, len(y_val))
    print(X_train, y_train)
    train_data = iMetDataset(
        path=path,
        image_list=X_train,
        label_list=y_train,
        mode='train',
        fine_size=fine_size)
    print(train_data[0])
    train_loader = DataLoader(
        train_data,
        shuffle=RandomSampler(train_data),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        # collate_fn=null_collate
    )

    val_data = iMetDataset(path, X_val, y_val, mode='val', fine_size=fine_size)
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        # collate_fn=null_collate
    )
    num_snapshot = 0
    best_acc = 0

    for i in range(restart_epoch):
        lr_scheduler.step()

    epochs = all_epoch
    for epoch in range(restart_epoch, epochs, 1):
        train_loss = train(train_loader, model, optimizer, criterion)
        test_loss, precision = test(val_loader, model, criterion)
        lr_scheduler.step()
        if precision > best_acc:
            best_acc = precision
            best_param = model.state_dict()
            torch.save(
                best_param,
                model_name +
                weight_name +
                str(num_snapshot) +
                '.pth')

        writer1.add_scalars('loss_' + model_name + str(fine_size),
                            {'train': train_loss, 'test': test_loss}, epoch)
        writer1.add_scalars('metric', {model_name: precision}, epoch)
        print(
            'epoch: {} train_loss: {:.3f} test_loss: {:.3f} score: {:.3f}'.format(
                epoch + 1,
                train_loss,
                test_loss,
                precision))
