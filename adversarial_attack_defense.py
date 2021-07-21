import argparse
from time import time

import torch

import onlinehd
from utils import load_mnist, load_model


def defence_retrain(args):
    print('Loading data...')
    x, x_test, y, y_test = load_mnist(args.data)
    classes = y.unique().size(0)
    features = x.size(1)

    print('Loading model...')
    model = onlinehd.OnlineHD(classes, features)
    load_model(model, args.model)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

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
    print(noise_h.size())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('--model', default='./results/model.pth')
    args = parser.parse_args()

    defence_retrain(args)
