import argparse
from time import time

import torch

import onlinehd
from utils import load_mnist, load_model


def test(args):
    print('Loading data...')
    x, x_test, y, y_test = load_mnist(args.data)
    classes = y.unique().size(0)
    features = x.size(1)

    print('Loading model...')
    model = onlinehd.OnlineHD(classes, features)
    load_model(model, args.model)

    if torch.cuda.is_available():
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Validating...')
    t = time()
    yhat_test = model(x_test)
    t = time() - t

    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--model', default='./results/model.pth', type=str)
    args = parser.parse_args()

    test(args)
