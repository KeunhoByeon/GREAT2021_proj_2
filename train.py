import argparse
import os
import random
from time import time

import torch
import torch.backends.cudnn as cudnn

import onlinehd
from utils import load_mnist, save_model


# simple OnlineHD training
def train(args):
    print('Loading data...')
    x, x_test, y, y_test = load_mnist(args.data)
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features, dim=args.dimension)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)
    t = time() - t

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    # Save model object
    save_model(model, os.path.join(args.results, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.035, metavar='L')
    parser.add_argument('--epochs', default=20, metavar='E')
    parser.add_argument('--dimension', default=4000, metavar='D')
    parser.add_argument('--bootstrap', default=1.0, metavar='B')
    parser.add_argument('--one_pass_fit', default=True, metavar='O')
    parser.add_argument('--data', default='./data')
    parser.add_argument('--results', default='./results')
    parser.add_argument('--seed', default=103)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    os.makedirs(args.results, exist_ok=True)

    # Save args
    with open(os.path.join(args.results, 'args.txt'), 'w') as wf:
        wf.write(str(args))

    train(args)
