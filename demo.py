import numpy as np
import cv2
import torch
from torch.nn import functional as F
from utils import util
from model.arch import FCA


def main():
    SCALE = 4
    model = FCA()
    model_path = './checkpoints/stage2_vgg/1000_G.pth'
    demo_folder = './demo'
    save_folder = './results/demo'
    util.mkdirs(save_folder)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    def single_forward(model, imgs_in, F10, F12):
        # imgs_in: [1, 4, 3, h, w]
        # output:  [1, 4, 3, 4h, 4w]
        with torch.no_grad():
            _, _, _, h, w = imgs_in.size()
            h_n = int(8 * np.ceil(h / 8))
            w_n = int(8 * np.ceil(w / 8))
            padding = (0, w_n - w, 0, h_n - h, 0, 0)
            imgs_in = F.pad(imgs_in, padding, mode='replicate')
            outs, F21, F23 = model(imgs_in, F10, F12)
            outs = outs[:, :, :, 0:SCALE * h, 0:SCALE * w]
        return outs, F21, F23

    # read and preprocess
    inputs = util.read_seq_imgs(demo_folder).unsqueeze(0).to(device)
    print(inputs.shape)

    # inference
    outputs = single_forward(model, inputs, None, None)[0]
    print(outputs.shape)
    outputs = outputs.data.float().cpu().squeeze(0)

    # postprocess
    output_t = outputs[0, :, :, :]
    output_2 = outputs[1, :, :, :]
    output_t = util.tensor2img(output_t)
    output_2 = util.tensor2img(output_2)

    cv2.imwrite(save_folder + '/ItSR.png', output_t)
    cv2.imwrite(save_folder + '/I2SR.png', output_2)


if __name__ == '__main__':
    main()
