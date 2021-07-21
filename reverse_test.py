from time import time

import os
import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import math

import onlinehd


# loads simple mnist dataset
def load():
    # fetches data
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    y = np.array(y)
    x = x.astype(np.float64)
    y = y.astype(np.int64)

    # split and normalize
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    os.makedirs('./quick_data', exist_ok=True)
    torch.save(x, './quick_data/x')
    torch.save(y, './quick_data/y')
    torch.save(x_test, './quick_data/x_test')
    torch.save(y_test, './quick_data/y_test')

    # print('Testset Size:', x_test.size(), y_test.size())
    # print('Labels:', y_test.unique())

    return x, x_test, y, y_test


def quick_load():
    try:
        x = torch.load('./quick_data/x')
        y = torch.load('./quick_data/y')
        x_test = torch.load('./quick_data/x_test')
        y_test = torch.load('./quick_data/y_test')

        # print('Testset Size:', x_test.size(), y_test.size())
        # print('Labels:', y_test.unique())

        return x, x_test, y, y_test
    except:
        return load()


def model_load(model, load_path):
    model_load = torch.load(load_path)

    model.classes = model_load.classes
    model.dim = model_load.dim
    model.encoder.dim = model_load.encoder.dim
    model.encoder.features = model_load.encoder.features

    model.model = model_load.model
    model.encoder.basis = model_load.encoder.basis
    model.encoder.base = model_load.encoder.base


# simple OnlineHD training
def test():
    print('Loading...')
    # x, x_test, y, y_test = load()
    x, x_test, y, y_test = quick_load()
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features)

    if torch.cuda.is_available():
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    model_load(model, 'model.pth')

    print('Validating...')
    t = time()
    raw_prob = model.probabilities_raw(x_test)
    t = time() - t

    yhat_test = raw_prob.argmax(1)
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    first_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 0]
    first_highest_prob = raw_prob[range(raw_prob.shape[0]), first_highest_indices]
    second_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 1]
    second_highest_prob = raw_prob[range(raw_prob.shape[0]), second_highest_indices]

    # noise_prob = 1.0 - second_highest_prob
    noise_prob = first_highest_prob - second_highest_prob

    noise_h = torch.zeros(raw_prob.size())
    noise_h[range(raw_prob.shape[0]), second_highest_indices] = noise_prob
    # print(noise_h.size())


if __name__ == '__main__':
    test()
