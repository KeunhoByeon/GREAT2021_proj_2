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
from onlinehd.spatial import reverse_cos_cdist
from utils import save_model, load_mnist, load_model, txt_on_img, hdvector2img


def get_noise_vector(raw_prob, gt):
    first_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 0]
    second_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 1]

    highest_error_indices = np.where(gt == first_highest_indices, second_highest_indices, first_highest_indices)
    noise_prob = 1.0 - raw_prob[range(raw_prob.shape[0]), highest_error_indices]

    # # I think it doesn't work well
    # first_highest_prob = raw_prob[range(raw_prob.shape[0]), first_highest_indices]
    # second_highest_prob = raw_prob[range(raw_prob.shape[0]), second_highest_indices]
    # noise_prob = first_highest_prob - second_highest_prob

    noise_vector = torch.zeros(raw_prob.size())
    noise_vector[range(raw_prob.shape[0]), highest_error_indices] = noise_prob

    return noise_vector


def validate(model, x_test, y_test, debug=False, debug_dir='./debug', debug_max_num=100, debug_resize_ratio=16):
    # print('Validating...')
    t = time()
    raw_prob = model.probabilities_raw(x_test)
    t = time() - t

    yhat_test = raw_prob.argmax(1)
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    noise_vector = get_noise_vector(raw_prob, y_test)
    noise = model.decode(reverse_cos_cdist(noise_vector, model.encode(x_test), model.model))
    noise_x_test = x_test + noise
    print('Noise sum:', torch.sum(noise).item())

    if debug:
        reverse_x_test = model.decode(reverse_cos_cdist(raw_prob, model.encode(x_test), model.model))
        noise_x_prob = model.probabilities_raw(x_test + noise)
        noise_x_yhat = noise_x_prob.argmax(1)
        reverse_yhat = model(reverse_x_test)
        reverse_x_test_acc = (y_test == reverse_yhat).float().mean().item()
        print(f'{reverse_x_test_acc = :6f}')
        noise_x_test_acc = (y_test == noise_x_yhat).float().mean().item()
        print(f'{noise_x_test_acc = :6f}')
        print('noise_x_test corrected ({}): {} + a'.format(len(np.where(y_test == noise_x_yhat)[0]), np.where(y_test == noise_x_yhat)[:10]))
        first_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 0]
        second_highest_indices = raw_prob.topk(k=2, dim=1).indices[:, 1]
        if debug_max_num == -1:  # -1 for all
            debug_max_num = len(y_test)
        for debug_index in tqdm(range(len(y_test)), desc='Debugging', total=debug_max_num):
            if debug_index >= debug_max_num:
                break

            gt = y_test[debug_index].item()
            reverse_pred = reverse_yhat[debug_index].item()
            first_highest_index = first_highest_indices[debug_index].item()
            second_highest_index = second_highest_indices[debug_index].item()
            noise_yhat = noise_x_yhat[debug_index].item()
            raw_probability = raw_prob[debug_index].cpu().detach().numpy()
            noise_probability = noise_x_prob[debug_index].cpu().detach().numpy()

            img_x_test = hdvector2img(x_test[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            img_reverse_x_test = hdvector2img(reverse_x_test[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            img_noise = hdvector2img(noise[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            img_noise_x_test = hdvector2img(noise_x_test[debug_index].cpu().detach().numpy(), resize_ratio=debug_resize_ratio)
            txt_on_img(img_x_test, 'img_x_test')
            txt_on_img(img_reverse_x_test, 'img_reverse_x_test')
            txt_on_img(img_noise, 'img_noise')
            txt_on_img(img_noise_x_test, 'img_noise_x_test')

            stack_img = np.hstack((img_x_test, img_reverse_x_test, img_noise, img_noise_x_test))

            l = img_x_test.shape[0]
            text_img = np.zeros((stack_img.shape[0] // 3 * 2, stack_img.shape[1]))
            txt_on_img(text_img, 'Ground truth: {}'.format(gt), coord=(10, 30 * 1))
            txt_on_img(text_img, '1st highest pred: {}'.format(first_highest_index), coord=(10, 30 * 2))
            txt_on_img(text_img, '2st highest pred: {}'.format(second_highest_index), coord=(10, 30 * 3))
            txt_on_img(text_img, 'reverse image pred: {}'.format(reverse_pred), coord=(10 + l * 1, 30 * 1))
            txt_on_img(text_img, 'Noise image pred: {}'.format(noise_yhat), coord=(10 + l * 3, 30 * 1))
            txt_on_img(text_img, 'Original image probability:', coord=(10, 30 * 5))
            txt_on_img(text_img, '{}'.format(raw_probability), coord=(10, 30 * 6), scale=0.8)
            txt_on_img(text_img, 'Noise image probability:', coord=(10, 30 * 8))
            txt_on_img(text_img, '{}'.format(noise_probability), coord=(10, 30 * 9), scale=0.8)

            stack_img = np.vstack((stack_img, text_img))

            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, '{}.png'.format(debug_index)), stack_img)


def retrain(args, model, x, y):
    print('Extracting noise...')
    t = time()
    raw_prob = model.probabilities_raw(x)
    t = time() - t

    yhat = raw_prob.argmax(1)
    acc = (y == yhat).float().mean()
    print(f'{acc = :6f}')
    # print(f'{t = :6f}')

    noise_vector = get_noise_vector(raw_prob, y)
    noise = model.decode(reverse_cos_cdist(noise_vector, model.encode(x), model.model))
    noise_x = x + noise

    print('Retraining...')
    t = time()
    model = model.fit(noise_x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)
    t = time() - t

    yhat = model(x)
    acc = (y == yhat).float().mean()
    print(f'{acc = :6f}')
    # print(f'{t = :6f}')


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

    print('\n[Validate Before Retrain]')
    validate(model, x_test, y_test, debug=True, debug_dir='./debug_before_retrain')

    for retrain_i in range(args.retrain_iter):
        print('\n[Retrain {}/{}]'.format(retrain_i, args.retrain_iter))
        retrain(args, model, x, y)

        print('\n[Validate {}/{}]'.format(retrain_i, args.retrain_iter))
        validate(model, x_test, y_test)  # , debug=True, debug_dir='./debug_after_retrain_{}'.format(retrain_i))

    print('\n[Validate After Retrain]')
    validate(model, x_test, y_test, debug=True, debug_dir='./debug_after_retrain')

    # Save model object
    save_model(model, os.path.join(args.results, 'model_retrained.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, metavar='L')
    parser.add_argument('--epochs', default=10, metavar='E')
    parser.add_argument('--bootstrap', default=1.0, metavar='B')
    parser.add_argument('--one_pass_fit', default=False, metavar='O')
    parser.add_argument('--retrain_iter', default=5)
    parser.add_argument('--data', default='./data')
    parser.add_argument('--model', default='./results/model.pth')
    parser.add_argument('--results', default='./results')
    parser.add_argument('--seed', default=103)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Save args
    with open(os.path.join(args.results, 'args_retrained.txt'), 'w') as wf:
        wf.write(str(args))

    main(args)
