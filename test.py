from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import onlinehd


# loads simple mnist dataset
def load():
    # fetches data
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    y = np.array(y)
    x = x.astype(np.float)
    y = y.astype(np.int)

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

    print('Testset Size:', x_test.size(), y_test.size())
    print('Labels:', y_test.unique())

    return x, x_test, y, y_test


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
    x, x_test, y, y_test = load()
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features)

    if torch.cuda.is_available():
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    model_load(model, 'model_new.pth')

    print('Validating...')
    yhat_test = model(x_test)
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc_test = :6f}')


if __name__ == '__main__':
    test()
