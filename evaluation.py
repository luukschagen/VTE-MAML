import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import *
from MAML import *
import json
import pandas as pd
from tqdm import tqdm


def get_loss(loss, benchmark):
    if loss == 'mse':
        loss = F.mse_loss
    elif loss == 'bce':
        loss = F.binary_cross_entropy_with_logits
    elif loss == 'crossentropy':
        loss = F.cross_entropy
    elif loss == 'gaussian_likelihood':
        loss = log_likelihood
    else:
        if benchmark == 'regression':
            loss = F.mse_loss
        elif benchmark == 'classification':
            loss = F.binary_cross_entropy_with_logits
        elif benchmark == 'imagenet':
            loss = F.cross_entropy
        else:
            loss = F.mse_loss
    return loss


def calibration_errors(row):
    return (row[0] + 0.05) - row['result']


def get_calibration(final_preds, targets):
    if final_preds.max() > 1:
        final_preds = final_preds.sigmoid()
    buckets = (final_preds * 10).floor() / 10
    results = pd.DataFrame([buckets.flatten().detach().numpy(), targets.flatten().detach().numpy()]).T

    calibration_df = pd.concat([results.groupby(0).mean(), results.groupby(0).count()], axis=1)

    calibration_df = calibration_df.reset_index()
    calibration_df.columns = ["buckets", "result", "number"]
    ce = calibration_df.apply(calibration_errors, axis=1).abs()
    ece = (ce * calibration_df['number']).sum()/len(targets)
    mce = ce.max()
    return ece, mce


def load_metalearner(settings, dataset, *pkl):
    x_s, __, y_s, __ = dataset[0]

    with open(settings) as file:
        settings = json.load(file)
    with open(settings['config']) as file:
        config = json.load(file)
    network = Learner.load(pkl[0], config)

    loss = get_loss(settings['loss'], settings['benchmark'])
    settings.pop('loss')

    if settings['maml_type'] == "vte-maml":
        if settings['benchmark'] == 'imagenet':
            if settings['hidden_size'] is None:
                settings['hidden_size'] = 128
            modul = ConvVariational_task_encoder.load(pkl[1], x_s.shape[1:], y_s.shape[0],
                                                      settings['convolutional_layers'], **settings)
        else:
            if settings['hidden_size'] is None:
                settings['hidden_size'] = 40
            modul = Variational_task_encoder.load(pkl[1], x_s.shape[-1], y_s.shape[-1], settings['modulation_layers'],
                                                  **settings)
        metalearner = VTE_MAML(network, modul, loss=loss, **settings)

    elif settings['maml_type'] == "mmaml":
        if settings['benchmark'] == 'imagenet':
            if settings['hidden_size'] is None:
                settings['hidden_size'] = 128
            modul = ConvModulation_network.load(pkl[1], x_s.shape[1:], y_s.shape[0],
                                                settings['convolutional_layers'], **settings)
        else:
            if settings['hidden_size'] is None:
                settings['hidden_size'] = 40
            modul = Modulation_network.load(pkl[1], x_s.shape[-1], y_s.shape[-1], settings['modulation_layers'],
                                            **settings)
        metalearner = MMAML(network, modul, loss=loss, **settings)

    else:
        metalearner = MAML(network, loss=loss, **settings)

    return metalearner


def score(metalearner, dataset, samples=1, updates=None, metric='accuracy'):
    score = 0
    final_preds = None
    targets = None
    for (x_s, x_q, y_s, y_q) in tqdm(dataset):
        predictions = None
        x_s, x_q, y_s, y_q = map(lambda x: x.to(metalearner.device), (x_s, x_q, y_s, y_q))
        for i in range(samples):

            args = metalearner.finetune(x_s, y_s, updates)
            with torch.no_grad():
                if type(args) == tuple:
                    pred = metalearner.predict(x_q, *args)
                else:
                    pred = metalearner.predict(x_q, args)
                if pred.shape[-1] > 1:
                    pred = torch.nn.Softmax(dim=-1)(pred)
                pred = pred.view(*pred.shape, 1)
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat([predictions, pred], dim=2)

        final_pred = predictions.mean(dim=2)

        if metric == 'accuracy':
            score += metalearner.get_accuracy(final_pred, y_q)
        if metric == 'mse':
            score += torch.nn.functional.mse_loss(final_pred, y_q)
        if final_preds is None:
            final_preds = final_pred
            targets = y_q
        else:
            final_preds = torch.cat([final_preds, final_pred], dim=0)
            targets = torch.cat([targets, y_q], dim=0)

    final_preds = final_preds.cpu()
    targets = targets.cpu()
    score = score/len(dataset)

    if final_preds.shape[-1] > 1:
        final_preds = final_preds.view(-1)
        onehot = torch.zeros(targets.shape[0], targets.max()+1)
        onehot[range(onehot.shape[0]), targets] = 1
        targets = onehot.view(-1)

    if metric == 'accuracy':
        ece, mce = get_calibration(final_preds, targets)
        return score, ece, mce
    else:
        return score


if __name__ == '__main__':
    pass