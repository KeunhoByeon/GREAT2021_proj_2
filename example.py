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


# simple OnlineHD training
def main():
    print('Loading...')
    x, x_test, y, y_test = quick_load()
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=1.0, lr=0.035, epochs=20)
    t = time() - t

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    torch.save(model, 'model.pth')


if __name__ == '__main__':
    main()
