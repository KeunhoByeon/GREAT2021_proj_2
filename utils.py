import os

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import torch


# loads simple mnist dataset
def load_mnist(data_dir, print_info=True, save_data=True):
    try:
        # Load local data
        x = torch.load(os.path.join(data_dir, 'x'))
        y = torch.load(os.path.join(data_dir, 'y'))
        x_test = torch.load(os.path.join(data_dir, 'x_test'))
        y_test = torch.load(os.path.join(data_dir, 'y_test'))
    except FileNotFoundError:
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

        if save_data:
            os.makedirs(data_dir, exist_ok=True)
            torch.save(x, os.path.join(data_dir, 'x'))
            torch.save(y, os.path.join(data_dir, 'y'))
            torch.save(x_test, os.path.join(data_dir, 'x_test'))
            torch.save(y_test, os.path.join(data_dir, 'y_test'))

    if print_info:
        print('Train dataset Size:', list(x.size()), list(y.size()))
        print('Train dataset Labels:', np.unique(y.numpy()))
        print('Test dataset Size:', list(x_test.size()), list(y_test.size()))
        print('Test dataset Labels:', np.unique(y_test.numpy()))

    return x, x_test, y, y_test


# Save model state dict
def save_model(model, save_path):
    state_dict = {}
    state_dict['model'] = model.model
    state_dict['classes'] = model.classes
    state_dict['dim'] = model.dim

    state_dict['encoder'] = {}
    state_dict['encoder']['dim'] = model.encoder.dim
    state_dict['encoder']['features'] = model.encoder.features
    state_dict['encoder']['base'] = model.encoder.base
    state_dict['encoder']['basis'] = model.encoder.basis

    torch.save(state_dict, save_path)


# Load trained model
def load_model(model, load_path):
    state_dict = torch.load(load_path)

    model.model = state_dict['model']
    model.classes = state_dict['classes']
    model.dim = state_dict['dim']

    model.encoder.dim = state_dict['encoder']['dim']
    model.encoder.features = state_dict['encoder']['features']
    model.encoder.base = state_dict['encoder']['base']
    model.encoder.basis = state_dict['encoder']['basis']
