import argparse
import os
import random
from time import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import onlinehd
from onlinehd.spatial import inverse_cos_cdist
from utils import load_mnist, load_model, txt_on_img


def get_noise_vector(raw_prob, gt):
    first_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 0]
    second_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 1]

    target_indices = np.where(gt == first_highest_indices, second_highest_indices, first_highest_indices)
    noise_prob = 1.0 - raw_prob[range(raw_prob.shape[0]), target_indices]

    # first_highest_prob = raw_prob[range(raw_prob.shape[0]), first_highest_indices]
    # second_highest_prob = raw_prob[range(raw_prob.shape[0]), second_highest_indices]
    # noise_prob = first_highest_prob - second_highest_prob

    noise_vector = torch.zeros(raw_prob.size())
    noise_vector[range(raw_prob.shape[0]), second_highest_indices] = noise_prob

    return noise_vector


def hdvector2img(vector, denormalize=True, resize_ratio=1):
    img = vector.reshape(-1, int(len(vector) ** 0.5))
    if denormalize:
        img = img / np.max(img) * 255.
    if resize_ratio != 1:
        img = cv2.resize(img, (img.shape[0] * resize_ratio, img.shape[1] * resize_ratio))
    return img


def validate(model, x_test, y_test, debug=False):
    print('Validating...')
    t = time()
    raw_prob = model.probabilities_raw(x_test)
    t = time() - t

    yhat_test = raw_prob.argmax(1)
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    noise_vector = get_noise_vector(raw_prob, y_test)
    noise = model.decode(inverse_cos_cdist(noise_vector, model.encode(x_test), model.model))
    noise_x_test = x_test + noise

    print('Noise sum:', torch.sum(noise).item())

    inverse_x_test = model.decode(inverse_cos_cdist(raw_prob, model.encode(x_test), model.model))
    noise_x_prob = model.probabilities_raw(x_test + noise)
    noise_x_yhat = noise_x_prob.argmax(1)
    inverse_yhat = model(inverse_x_test)
    print('inverse_x_test acc:', (y_test == inverse_yhat).float().mean().item())
    print('noise_x_test acc:', (y_test == noise_x_yhat).float().mean().item())

    if debug:
        debug_max_num = 100  # -1 for all

        print('noise_x_test corrected ({}):'.format(len(np.where(y_test == noise_x_yhat)[0])), np.where(y_test == noise_x_yhat))
        first_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 0]
        second_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 1]
        if debug_max_num == -1:
            debug_max_num = len(y_test)
        for debug_index in tqdm(range(len(y_test)), desc='Debug', total=debug_max_num):
            if debug_index >= debug_max_num:
                break

            gt = y_test[debug_index].item()
            inverse_pred = inverse_yhat[debug_index].item()
            first_highest_index = first_highest_indices[debug_index].item()
            second_highest_index = second_highest_indices[debug_index].item()
            noise_yhat = noise_x_yhat[debug_index].item()
            raw_probability = raw_prob[debug_index].numpy()
            noise_probability = noise_x_prob[debug_index].numpy()

            resize_ratio = 16
            img_x_test = hdvector2img(x_test[debug_index].numpy(), resize_ratio=resize_ratio)
            img_inverse_x_test = hdvector2img(inverse_x_test[debug_index].numpy(), resize_ratio=resize_ratio)
            img_noise = hdvector2img(noise[debug_index].numpy(), resize_ratio=resize_ratio)
            img_noise_x_test = hdvector2img(noise_x_test[debug_index].numpy(), resize_ratio=resize_ratio)
            txt_on_img(img_x_test, 'img_x_test')
            txt_on_img(img_inverse_x_test, 'img_inverse_x_test')
            txt_on_img(img_noise, 'img_noise')
            txt_on_img(img_noise_x_test, 'img_noise_x_test')

            stack_img = np.hstack((img_x_test, img_inverse_x_test, img_noise, img_noise_x_test))

            l = img_x_test.shape[0]
            text_img = np.zeros((stack_img.shape[0] // 3 * 2, stack_img.shape[1]))
            txt_on_img(text_img, 'Ground truth: {}'.format(gt), coord=(10, 30 * 1))
            txt_on_img(text_img, '1st highest pred: {}'.format(first_highest_index), coord=(10, 30 * 2))
            txt_on_img(text_img, '2st highest pred: {}'.format(second_highest_index), coord=(10, 30 * 3))
            txt_on_img(text_img, 'Inverse image pred: {}'.format(inverse_pred), coord=(10 + l * 1, 30 * 1))
            txt_on_img(text_img, 'Noise image pred: {}'.format(noise_yhat), coord=(10 + l * 3, 30 * 1))
            txt_on_img(text_img, 'Original image probability:', coord=(10, 30 * 5))
            txt_on_img(text_img, '{}'.format(raw_probability), coord=(10, 30 * 6), scale=0.8)
            txt_on_img(text_img, 'Noise image probability:', coord=(10, 30 * 8))
            txt_on_img(text_img, '{}'.format(noise_probability), coord=(10, 30 * 9), scale=0.8)

            stack_img = np.vstack((stack_img, text_img))

            debug_dir = './debug'
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, '{}.png'.format(debug_index)), stack_img)


def train(args, model, x, y):
    print('Extracting noise...')
    t = time()
    raw_prob = model.probabilities_raw(x)
    t = time() - t

    yhat = raw_prob.argmax(1)
    acc = (y == yhat).float().mean()
    print(f'{acc = :6f}')
    print(f'{t = :6f}')

    noise_vector = get_noise_vector(raw_prob, y)
    noise = model.decode(inverse_cos_cdist(noise_vector, model.encode(x), model.model))
    noise_x = x + noise

    print('Training...')
    t = time()
    model = model.fit(noise_x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs)
    t = time() - t

    yhat = model(x)
    acc = (y == yhat).float().mean()
    print(f'{acc = :6f}')
    print(f'{t = :6f}')


def defence_retrain(args):
    print('Loading data...')
    x, x_test, y, y_test = load_mnist(args.data)
    classes = y.unique().size(0)
    features = x.size(1)

    print('Loading model...')
    model = onlinehd.OnlineHD(classes, features, args.dimension)
    load_model(model, args.model)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('\nValidate Before Retrain ----------------------------------------')
    validate(model, x_test, y_test, debug=True)
    print('\nRetrain --------------------------------------------------------')
    train(args, model, x, y)
    print('\nValidate After Retrain -----------------------------------------')
    validate(model, x_test, y_test)  # , debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.035, metavar='L')
    parser.add_argument('--epochs', default=20, metavar='E')
    parser.add_argument('--dimension', default=4000, metavar='D')
    parser.add_argument('--bootstrap', default=1.0, metavar='B')
    parser.add_argument('--one_pass_fit', default=True, metavar='O')
    parser.add_argument('--data', default='./data')
    parser.add_argument('--model', default='./results/model.pth')
    parser.add_argument('--results', default='./results')
    parser.add_argument('--seed', default=103)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    defence_retrain(args)
