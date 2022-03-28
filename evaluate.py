import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import utils.util as util
from model.arch import FCA

SCALE = 4


def main():

    ############################ PSNR / SSIM ###########################
    #            odd (ST-SR)       even (S-SR)       overall           #
    # REDSval    22.61 / 0.6144    29.39 / 0.8306    26.03 / 0.7236    #
    ####################################################################

    model = FCA()
    model_path = './checkpoints/stage2_vgg/1000_G.pth'

    # dataset
    data_mode = 'REDSval'  # REDSval4 | REDSval | REDStest

    if data_mode in ['REDSval4', 'REDSval']:
        test_dataset_folder = '/path/to/REDS/val/val_sharp_bicubic/X4/*'
    elif data_mode == 'REDStest':
        test_dataset_folder = '/path/to/REDS/test/test_sharp_bicubic/X4/*'
    else:
        raise NotImplementedError(
            'data_mode [{:s}] is not implemented.'.format(data_mode))

    # save imgs
    save_imgs = True  # True | False

    save_folder = './results/{}'.format(data_mode)
    save_sub_folder_name = util.get_timestamp()
    util.mkdirs(save_folder)

    util.setup_logger('base',
                      save_folder,
                      'test',
                      level=logging.INFO,
                      screen=True,
                      tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    def single_forward(model, imgs_in, F10, F12):
        # imgs_in: [n, 4, 3, h, w]
        # output:  [n, 2, 3, 4h, 4w]
        with torch.no_grad():
            _, _, _, h, w = imgs_in.size()
            h_n = int(8 * np.ceil(h / 8))
            w_n = int(8 * np.ceil(w / 8))
            padding = (0, w_n - w, 0, h_n - h, 0, 0)
            imgs_in = F.pad(imgs_in, padding, mode='replicate')
            outs, F21, F23 = model(imgs_in, F10, F12)
            outs = outs[:, :, :, 0:SCALE * h, 0:SCALE * w]
        return outs, F21, F23

    avg_psnr_l = []
    avg_ssim_l = []
    sub_folder_name_l = []
    psnr_odd = []
    psnr_even = []
    ssim_odd = []
    ssim_even = []

    select_idx_list = util.test_index_generation_REDS()
    # print(select_idx_list)

    sub_folder_l = sorted(glob.glob(test_dataset_folder))
    assert sub_folder_l, "Please modify the dataset path."

    if data_mode == 'REDSval4':
        # The first four sequences of REDSval: 000, 001, 002, 003
        sub_folder_l = sub_folder_l[:4]

    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)

        save_sub_folder = osp.join(save_folder, save_sub_folder_name)
        if save_imgs:
            util.mkdirs(save_sub_folder)

        # read LR images
        imgs = util.read_seq_imgs(sub_folder)
        # read GT images
        img_GT_l = []

        if 'REDStest' not in data_mode:
            sub_folder_GT = osp.join(
                sub_folder.replace('/val_sharp_bicubic/X4/', '/val_sharp/'))
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT, '*'))):
                img_GT_l.append(util.read_image(img_GT_path))

        cal_n = 0
        avg_psnr, avg_psnr_sum = 0, 0
        avg_ssim, avg_ssim_sum = 0, 0
        F10, F12 = None, None

        # process each image
        for i, select_idxs in enumerate(select_idx_list):
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(
                0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = single_forward(model, imgs_in, F10, F12)
            F10, F12 = output[1], output[2]
            # if i < len(select_idx_list) - 3:
            #     F10, F12 = output[1], output[2]
            # else:
            #     F10, F12 = None, None

            outputs = output[0].data.float().cpu().squeeze(0)

            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)

                output_f = outputs[idx, :, :, :].squeeze(0)
                output = util.tensor2img(output_f)

                if save_imgs:
                    cv2.imwrite(
                        osp.join(
                            save_sub_folder,
                            '{:03d}_{:08d}.png'.format(int(sub_folder_name),
                                                       name_idx)), output)
                # calculate PSNR and SSIM
                if 'REDStest' not in data_mode:
                    output = output / 255
                    GT = img_GT_l[name_idx]

                    crt_psnr = peak_signal_noise_ratio(output,
                                                       GT,
                                                       data_range=1)
                    crt_ssim = structural_similarity(
                        output,
                        GT,
                        multichannel=True,
                        gaussian_weights=True,
                        use_sample_covariance=False,
                        data_range=1)

                    logger.info(
                        '{:3d} - {:25}.png \tPSNR: {:.6f} dB  SSIM: {:.4f}'.
                        format(name_idx, name_idx, crt_psnr, crt_ssim))
                    avg_psnr_sum += crt_psnr
                    avg_ssim_sum += crt_ssim
                    cal_n += 1

                    if name_idx % 2 == 0:
                        psnr_even.append(crt_psnr)
                        ssim_even.append(crt_ssim)
                    else:
                        psnr_odd.append(crt_psnr)
                        ssim_odd.append(crt_ssim)

        if 'REDStest' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_ssim = avg_ssim_sum / cal_n
            avg_psnr_l.append(avg_psnr)
            avg_ssim_l.append(avg_ssim)
            logger.info(
                'Folder {} - Average PSNR: {:.6f} dB  SSIM: {:.4f} for {} frames'
                .format(sub_folder_name, avg_psnr, avg_ssim, cal_n))
        else:
            logger.info('Folder {}'.format(sub_folder_name))

    if 'REDStest' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, ssim in zip(sub_folder_name_l, avg_psnr_l, avg_ssim_l):
            logger.info(
                'Folder {} - Average PSNR: {:.6f} dB  SSIM: {:.4f}. '.format(
                    name, psnr, ssim))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info(
            'Total Average PSNR: {:.2f} dB\tPSNR (odd): {:.2f} dB\tPSNR (even): {:.2f} dB'
            .format(np.mean(avg_psnr_l), np.mean(psnr_odd),
                    np.mean(psnr_even)))
        logger.info(
            'Total Average SSIM: {:.4f}\tSSIM (odd): {:.4f}\tSSIM (even): {:.4f}'
            .format(np.mean(avg_ssim_l), np.mean(ssim_odd),
                    np.mean(ssim_even)))


if __name__ == '__main__':
    main()
