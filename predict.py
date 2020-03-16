import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.imetutils import path, data_path, binarize_prediction, _make_mask, get_score
from config import log_dir, batch_size, fine_size, cuda, all_fold
from models.SEResNextmodel.SEResNexthead import GAPseResNext50, GAPseResNext101
from imet_dataset import iMetDataset
device = torch.device('cuda:0' if cuda else 'cpu')


def test(model, i, ids):
    N = 5
    for TTA_list in range(N):
        test_data = iMetDataset(
            path,
            ids,
            TTA_list=TTA_list,
            mode='test',
            is_tta=True,
            fine_size=fine_size)
        test_loader = DataLoader(
            test_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True)
        predicts = np.empty([len(ids), 1103], dtype=np.float32)
        model.eval()
        for iter_idx, images in tqdm(
                enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
        with torch.set_grad_enabled(False):
            pred = model(images)
            pred = torch.sigmoid(pred).detach().cpu().numpy()
            predicts[iter_idx * batch_size:(iter_idx + 1) * batch_size] = pred
        if TTA_list == 0:
            N_predicts = predicts
        else:
            N_predicts += predicts
            predicts = N_predicts / N
    return predicts


def model_load(i, model_name):
    if model_name == "GAPResNext101":
        if i == 0:
            param = torch.load(
                "/workspace/ResNet101exp0010.pth")
        elif i == 1:
            param = torch.load(
                "/workspace/ResNet101exp0010.pth")

        elif i == 2:
            param = torch.load(
                "/workspace/ResNet101exp0010.pth")

        elif i == 3:
            param = torch.load(
                "/workspace/ResNet101exp0010.pth")

        elif i == 4:
            param = torch.load(
                "/workspace/ResNet101exp0010.pth")
        return param
    elif model_name == "GAPResNext50":
        if i == 0:
            param = torch.load("/workspace/Resnet50rexp0010.pth")
        elif i == 1:
            param = torch.load("/workspace/Resnet50rexp0010.pth")
        elif i == 2:
            param = torch.load("/workspace/Resnet50rexp0010.pth")
        elif i == 3:
            param = torch.load("/workspace/Resnet50rexp0010.pth")
        elif i == 4:
            param = torch.load("/workspace/Resnet50rexp0010.pth")
        return param


def run():
    df = pd.read_csv(path + 'sample_submission.csv')
    ids = df['id'].values
    for i in range(all_fold):
        param = model_load(i, model_name="GAPResNext101")
        model = GAPseResNext101()
        model = nn.DataParallel(model)
        model.load_state_dict(param)
        cudnn.benchmark = True
        model = model.to(device)

        predicts = test(model, i, ids)
        if i == 0:
            fold_predicts = predicts
        else:
            fold_predicts += predicts
    model_1_predicts = fold_predicts / 5

    for i in range(all_fold):
        param = model_load(i, model_name="GAPResNext50")
        modelR = GAPseResNext50()
        modelR = nn.DataParallel(modelR)

        modelR.load_state_dict(param)
        cudnn.benchmark = True
        modelR = modelR.to(device)

        predicts = test(modelR, i, ids)
        if i == 0:
            R_fold_predicts = predicts
        else:
            R_fold_predicts += predicts
    model_2_predicts = R_fold_predicts / 5
    # ensemble_model 
    predicts = (2 * model_1_predicts) / 3 + (model_2_predicts) / 3
    pred_name = 'submission'
    np.save(pred_name, predicts)

    #  predicts = (predicts > 0.27).astype(np.bool)
    argsorted = predicts.argsort(axis=1)
    predicts = binarize_prediction(predicts, 0.20, argsorted)
    for i, preds in enumerate(predicts):
        df.iloc[i]['attribute_ids'] = ''
        for idx, pred in enumerate(preds):
            if pred:
                df.iloc[i]['attribute_ids'] += (str(idx) + ' ')
        df.iloc[i]['attribute_ids'] = df.iloc[i]['attribute_ids'][:-1]
    df.to_csv(pred_name + '.csv', index=False)


if __name__ == '__main__':
    run()
