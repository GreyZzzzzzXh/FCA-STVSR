import os
import math
from datetime import datetime
import random
import logging
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
import glob
import re


def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0 * params / (1000 * 1000))


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info(
            'Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name,
                 root,
                 phase,
                 level=logging.INFO,
                 screen=False,
                 tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root,
                                phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0]
                                      )  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)),
                           normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'
            .format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def read_image(img_path):
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images'''
    img_path_l = glob.glob(img_seq_path + '/*')
    # img_path_l.sort(key=lambda x: int(os.path.basename(x)[:-4]))
    img_path_l.sort(
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1], torch
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(
        np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def test_index_generation(skip, N_out, len_in):
    '''
    params: 
    skip: if skip even number; 
    N_out: number of frames of the network; 
    len_in: length of input frames

    example:
  len_in | N_out  | times | (no skip)                  |   (skip)
    5    |   3    |  4/2  | [0,1], [1,2], [2,3], [3,4] | [0,2],[2,4]
    7    |   3    |  5/3  | [0,1],[1,2][2,3]...[5,6]   | [0,2],[2,4],[4,6] 
    5    |   5    |  2/1  | [0,1,2] [2,3,4]            | [0,2,4]
    '''
    # number of input frames for the network
    N_in = 1 + N_out // 2
    # input length should be enough to generate the output frames
    assert N_in <= len_in

    sele_list = []
    if skip:
        right = N_out  # init
        while (right <= len_in):
            h_list = [right - N_out + x for x in range(N_out)]
            l_list = h_list[::2]
            right += (N_out - 1)
            sele_list.append([l_list, h_list])
    else:
        right = N_out  # init
        right_in = N_in
        while (right_in <= len_in):
            h_list = [right - N_out + x for x in range(N_out)]
            l_list = [right_in - N_in + x for x in range(N_in)]
            right += (N_out - 1)
            right_in += (N_in - 1)
            sele_list.append([l_list, h_list])
    # check if it covers the last image, if not, we should cover it
    # https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/pull/25/commits/c86c6b9b31aa5041fba05be6cdd0739e161fbd34
    # maybe this commit is wrong?

    # if (skip) and (right < len_in - 1):
    if (skip) and (right != len_in + N_out - 1):
        h_list = [len_in - N_out + x for x in range(N_out)]
        l_list = h_list[::2]
        sele_list.append([l_list, h_list])
    # print(right_in, len_in)
    # if (not skip) and (right_in < len_in - 1):
    if (not skip) and (right_in != len_in + N_in - 1):
        right = len_in * 2 - 1
        h_list = [right - N_out + x for x in range(N_out)]
        l_list = [len_in - N_in + x for x in range(N_in)]
        sele_list.append([l_list, h_list])
    return sele_list


def test_index_generation_REDS():
    # [0, 2, 4, 6]      => [3, 4]
    # [2, 4, 6, 8]      => [5, 6]
    # ...
    # [92, 94, 96, 98]  => [95, 96]
    sele_list = []
    for i in range(0, 94, 2):
        l_list = [i + 2 * j for j in range(4)]
        h_list = [i + 3, i + 4]
        sele_list.append([l_list, h_list])
    sele_list.append([[94, 96, 98, 98], [97, 98]])
    sele_list.append([[6, 4, 2, 0], [3, 2]])
    sele_list.append([[4, 2, 0, 0], [1, 0]])
    return sele_list


if __name__ == '__main__':
    select_idx_list = test_index_generation_REDS()
    for i, select_idxs in enumerate(select_idx_list):
        print(i, select_idxs)
