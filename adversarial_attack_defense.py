import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from time import time

import onlinehd
from onlinehd.spatial import reverse_cos_cdist
from utils import save_model, load_mnist, load_model, txt_on_img, hdvector2img


def get_noise_probability(prob, gt, alpha=0.0001):
    first_high = prob.topk(k=2, dim=1).indices[:, 0]
    second_high = prob.topk(k=2, dim=1).indices[:, 1]
    highest_error_label = np.where(gt == first_high, second_high, first_high)  # Error label with highest probability

    # # Do not use it anymore
    # noise_prob = 1.0 - prob[range(prob.shape[0]), highest_error_indexes]

    gt_prob = prob[range(prob.shape[0]), gt]
    highest_error_prob = prob[range(prob.shape[0]), highest_error_label]
    noise_prob = gt_prob - highest_error_prob + alpha
    noise_prob = noise_prob.clamp(min=0., max=1.)

    noise_probability = torch.zeros(prob.size())
    noise_probability[range(prob.shape[0]), highest_error_label] = noise_prob

    return noise_probability


def validate(model, x_test, y_test, print_name='Validate', debug=False, debug_dir='./debug', debug_max_num=100, debug_resize_ratio=16):
    h_test = model.encode(x_test)
    prob = model.probabilities_raw(h_test, encoded=True)

    yhat_test = prob.argmax(1)
    acc_test = (y_test == yhat_test).float().mean()

    noise_probability = get_noise_probability(prob, y_test, alpha=args.alpha)
    noise_h = reverse_cos_cdist(noise_probability, model.encode(x_test), model.model)
    h_test_noised = h_test + noise_h
    x_test_noised = model.decode(h_test_noised)

    x_test_noised_prob = model.probabilities_raw(h_test_noised, encoded=True)
    x_test_noised_yhat = x_test_noised_prob.argmax(1)
    x_test_noised_acc = (y_test == x_test_noised_yhat).float().mean().item()

    noise = model.decode(noise_h)
    noise_mean = torch.mean(noise).item()
    print("[{}]    acc x_test: {:.6f}    acc x_test_noised: {:.6f}    noise mean: {:.6f}".format(print_name, acc_test, x_test_noised_acc, noise_mean))

    if debug:
        first_highest_indices = prob.topk(k=2, dim=1).indices[:, 0]
        second_highest_indices = prob.topk(k=2, dim=1).indices[:, 1]

        if debug_max_num == -1:  # -1 for all
            debug_max_num = len(y_test)

        for debug_index in tqdm(range(len(y_test)), desc='Debugging', total=debug_max_num):
            if debug_index >= debug_max_num:
                break

            gt = y_test[debug_index].item()
            probability = prob[debug_index].cpu().detach().numpy()
            first_high = first_highest_indices[debug_index].item()
            second_high = second_highest_indices[debug_index].item()

            noised_yhat = x_test_noised_yhat[debug_index].item()
            noised_probability = x_test_noised_prob[debug_index].cpu().detach().numpy()

            img_x_test = hdvector2img(x_test[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            img_noise = hdvector2img(noise[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            img_x_test_noised = hdvector2img(x_test_noised[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            txt_on_img(img_x_test, 'x_test')
            txt_on_img(img_noise, 'noise')
            txt_on_img(img_x_test_noised, 'x_test_noised')

            stack_img = np.hstack((img_x_test, img_noise, img_x_test_noised))

            l = img_x_test.shape[0]
            text_img = np.zeros((stack_img.shape[0] // 3 * 2, stack_img.shape[1]))
            txt_on_img(text_img, 'Ground truth: {}'.format(gt), coord=(10, 30 * 1))
            txt_on_img(text_img, '1st highest pred: {}'.format(first_high), coord=(10, 30 * 2))
            txt_on_img(text_img, '2st highest pred: {}'.format(second_high), coord=(10, 30 * 3))
            txt_on_img(text_img, 'Noised image pred: {}'.format(noised_yhat), coord=(10 + l * 2, 30 * 1))
            txt_on_img(text_img, 'Original image probability:', coord=(10, 30 * 5))
            txt_on_img(text_img, '{}'.format(probability), coord=(10, 30 * 6), scale=0.6)
            txt_on_img(text_img, 'Noised image probability:', coord=(10, 30 * 8))
            txt_on_img(text_img, '{}'.format(noised_probability), coord=(10, 30 * 9), scale=0.6)

            stack_img = np.vstack((stack_img, text_img))

            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, '{}.png'.format(debug_index)), stack_img)

    return acc_test, x_test_noised_acc, noise_mean


def retrain(args, model, x, y, print_name=''):
    t = time()
    h = model.encode(x)
    prob = model.probabilities_raw(h, encoded=True)

    yhat = prob.argmax(1)
    acc = (y == yhat).float().mean()

    noise_probability = get_noise_probability(prob, y, alpha=args.alpha)
    noise_h = reverse_cos_cdist(noise_probability, model.encode(x), model.model)
    h_noised = h + noise_h

    model = model.fit(h_noised, y, encoded=True, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)
    t = time() - t

    yhat = model(x)
    acc_after_retrain = (y == yhat).float().mean()
    print("[{}]    acc: {:.6f}    acc after retrain: {:.6f}    time: {:.2f}".format(print_name, acc, acc_after_retrain, t))

    return acc, acc_after_retrain


def main(args):
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

    # Validate Before Retrain
    validate(model, x_test, y_test, print_name='Validate Before Retrain', debug=args.debug, debug_dir='./debug_before_retrain', debug_max_num=100)

    # Retrain
    for retrain_i in range(args.retrain_iter):
        retrain(args, model, x, y, print_name='Retrain  {}/{}'.format(retrain_i + 1, args.retrain_iter))
        validate(model, x_test, y_test, print_name='Validate {}/{}'.format(retrain_i + 1, args.retrain_iter),
                 debug=False, debug_dir='./debug_after_retrain_{}'.format(retrain_i), debug_max_num=100)  # No debug when validating during retraining

    # Validate After Retrain
    validate(model, x_test, y_test, print_name='Validate After Retrain',debug=args.debug, debug_dir='./debug_after_retrain', debug_max_num=100)

    # Save model object
    save_model(model, os.path.join(args.results, 'model_retrained.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.02, type=float, metavar='L')
    parser.add_argument('--epochs', default=30, type=int, metavar='E')
    parser.add_argument('--bootstrap', default=1.0, type=float, metavar='B')
    parser.add_argument('--one_pass_fit', default=False, type=bool, metavar='O')
    parser.add_argument('--retrain_iter', default=20, type=int)
    parser.add_argument('--alpha', default=0.0001, type=float)
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--model', default='./results/model.pth', type=str)
    parser.add_argument('--seed', default=103, type=int)
    parser.add_argument('--results', default='./results', type=str)
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Save args
    with open(os.path.join(args.results, 'args_retrained.txt'), 'w') as wf:
        wf.write(str(args))

    main(args)
