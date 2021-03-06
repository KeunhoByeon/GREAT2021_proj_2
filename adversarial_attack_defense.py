import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from time import time

import onlinehd
from onlinehd.spatial import reverse_cos_cdist
from utils import save_model, load_mnist, load_model, txt_on_img


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
    t = time()
    h_test = model.encode(x_test)
    prob = model.probabilities_raw(h_test, encoded=True)
    yhat_test = prob.argmax(1)
    acc_test = (y_test == yhat_test).float().mean()

    noise_prob = get_noise_probability(prob, y_test, alpha=args.alpha)
    prob_noised = prob + noise_prob
    h_test_noised = reverse_cos_cdist(prob_noised, h_test, model.model)
    x_test_noised = model.decode(h_test_noised)
    x_test_noised_prob = model.probabilities_raw(x_test_noised)
    noise_x = x_test - x_test_noised

    t = time() - t

    noise_mean = torch.mean(torch.abs(noise_x)).item()
    x_test_noised_yhat = x_test_noised_prob.argmax(1)
    acc_test_noised = (y_test == x_test_noised_yhat).float().mean().item()
    print("[{}]    acc x_test: {:.6f}    acc x_test_noised: {:.6f}    noise mean: {:.6f}    time: {:.2f}".format(print_name, acc_test, acc_test_noised, noise_mean, t))

    if debug:
        import cv2
        first_highest_indices = prob.topk(k=2, dim=1).indices[:, 0]
        second_highest_indices = prob.topk(k=2, dim=1).indices[:, 1]

        img_size = int(x_test.size()[1] ** 0.5)
        desired_size = int(img_size * debug_resize_ratio)

        # Tiled image
        os.makedirs('./debug/x_noised_tile', exist_ok=True)
        noised_debug_imgs = x_test_noised.cpu().detach().numpy()
        # noised_debug_imgs = (noised_debug_imgs - np.amin(noised_debug_imgs, axis=1)[:, np.newaxis]) \
        #                     / (np.amax(noised_debug_imgs, axis=1)[:, np.newaxis] - np.amin(noised_debug_imgs, axis=1)[:, np.newaxis]) * 255.
        noised_debug_imgs = noised_debug_imgs / np.amax(noised_debug_imgs, axis=1)[:, np.newaxis] * 255.
        noised_debug_imgs = noised_debug_imgs.reshape(-1, img_size, img_size)
        tile_img = noised_debug_imgs[:100].reshape(10, 10, img_size, img_size)
        tile_img = cv2.vconcat([cv2.hconcat(tmp) for tmp in tile_img])
        tile_img = cv2.resize(tile_img, (tile_img.shape[0] * 2, tile_img.shape[1] * 2))

        info_img = np.zeros((180, tile_img.shape[1]))
        txt_on_img(info_img, "[{}]".format(print_name), coord=(10, 30 * 1))
        txt_on_img(info_img, "acc x_test: {:.6f}".format(acc_test), coord=(10, 30 * 2))
        txt_on_img(info_img, "acc x_test_noised: {:.6f}".format(acc_test_noised), coord=(10, 30 * 3))
        txt_on_img(info_img, "noise mean: {:.6f}".format(noise_mean), coord=(10, 30 * 4))
        txt_on_img(info_img, "x_test_noised ({:.2f} ~ {:.2f})".format(x_test_noised.min(), x_test_noised.max()), coord=(10, 30 * 5))

        tile_img = np.vstack((tile_img, info_img))
        cv2.imwrite(os.path.join('./debug/x_noised_tile', '{}.png'.format(os.path.basename(debug_dir))), tile_img)

        # Stacked image
        os.makedirs(debug_dir, exist_ok=True)
        if debug_max_num == -1:  # -1 for all
            debug_max_num = len(y_test)

        imgs_x_test = x_test.cpu().detach().numpy().reshape(-1, img_size, img_size)
        imgs_noise = noise_x.cpu().detach().numpy().reshape(-1, img_size, img_size)
        imgs_x_test_noised = x_test_noised.cpu().detach().numpy().reshape(-1, img_size, img_size)

        def process_debug_img(img, dsize, title):
            img = cv2.resize(img, (dsize, dsize))
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
            txt_on_img(img, title)
            return img

        for i in tqdm(range(len(y_test)), desc='Debugging', total=debug_max_num, leave=False):
            if i >= debug_max_num:
                break
            img_x_test = process_debug_img(imgs_x_test[i], desired_size, 'x_test')
            img_noise = process_debug_img(imgs_noise[i], desired_size, 'noise')
            img_x_test_noised = process_debug_img(imgs_x_test_noised[i], desired_size, 'x_test_noised')
            stack_img = np.hstack((img_x_test, img_noise, img_x_test_noised))

            text_img = np.zeros((120, stack_img.shape[1]))
            txt_on_img(text_img, 'Ground truth: {}'.format(y_test[i].item()), coord=(10, 30 * 1))
            txt_on_img(text_img, '1st highest pred: {}'.format(first_highest_indices[i].item()), coord=(10, 30 * 2))
            txt_on_img(text_img, '2st highest pred: {}'.format(second_highest_indices[i].item()), coord=(10, 30 * 3))
            txt_on_img(text_img, 'Noised image pred: {}'.format(x_test_noised_yhat[i].item()), coord=(10 + img_x_test.shape[0] * 2, 30 * 1))
            stack_img = np.vstack((stack_img, text_img))

            cv2.imwrite(os.path.join(debug_dir, '{}.png'.format(i)), stack_img)

    return acc_test, acc_test_noised, noise_mean


def retrain(args, model, x, y, print_name=''):
    t = time()
    h = model.encode(x)
    prob = model.probabilities_raw(h, encoded=True)

    yhat = prob.argmax(1)
    acc = (y == yhat).float().mean()

    noise_prob = get_noise_probability(prob, y, alpha=args.alpha)
    prob_noised = prob + noise_prob
    h_noised = reverse_cos_cdist(prob_noised, h, model.model)
    x_noised = model.decode(h_noised)
    model = model.fit(x_noised, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)

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
    validate(model, x_test, y_test, print_name='Validate Before Retrain', debug=args.debug, debug_dir='./debug/before_retrain', debug_max_num=100)

    # Retrain
    import matplotlib.pyplot as plt
    acc_test_list = []
    noise_mean_list = []
    for retrain_i in range(args.retrain_iter):
        retrain(args, model, x, y, print_name='Retrain  {}/{}'.format(retrain_i + 1, args.retrain_iter))
        acc_test, acc_test_noised, noise_mean = validate(model, x_test, y_test, print_name='Validate {}/{}'.format(retrain_i + 1, args.retrain_iter),
                 # debug=False)  # No debug when validating during retraining
                 debug=args.debug, debug_dir='./debug/retrain_iter/{}'.format(retrain_i + 1), debug_max_num=100)
        noise_mean_list.append(noise_mean)
        acc_test_list.append(acc_test)

    if args.debug:
        plt.plot(range(len(noise_mean_list)), noise_mean_list)
        plt.xlabel('Train iter')
        plt.ylabel('noise mean')
        plt.savefig('./debug/noise_mean.png', dpi=300)
        plt.clf()
        plt.plot(range(len(acc_test_list)), acc_test_list)
        plt.xlabel('Train iter')
        plt.ylabel('Acc original x_test')
        plt.savefig('./debug/acc_test.png', dpi=300)
        plt.clf()

    # Validate After Retrain
    validate(model, x_test, y_test, print_name='Validate After Retrain', debug=args.debug, debug_dir='./debug/after_retrain', debug_max_num=100)

    # Save model object
    save_model(model, os.path.join(args.results, 'model_retrained.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.035, type=float, metavar='L')
    parser.add_argument('--epochs', default=20, type=int, metavar='E')
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
