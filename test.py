import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10

from model import Deraining
from data import outdoor_rain_test, outdoor_rain_train
from helper import *
import time
from scipy import io
from PIL import Image
import cv2
from skimage.measure import compare_ssim as ssim


parser = argparse.ArgumentParser(description='Deraining')

# validation data
parser.add_argument('--val_data_dir', required=False, default='D:/DATASETS/Heavy_rain_image_cvpr2019/test') # modifying to your SR_data folder path
# parser.add_argument('--rain_valDataroot', required=False, default='G:/DATASET/JORDER_DATASET/test/rain_data_test_Light') # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=1)

parser.add_argument('--pretrained_model', default='save/Deraining/model/model_lastest.pt', help='save result')

parser.add_argument('--nchannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patch_size', type=int, default=256, help='patch size')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
					help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def get_testdataset(args):
    data_test = outdoor_rain_test(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def test(args):

    # SR network
    my_model = Deraining(args)
    my_model = nn.DataParallel(my_model)
    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))

    ''' if dont want use nn.DataParallel
    # original saved file with DataParallel
    state_dict = torch.load(args.pretrained_model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    my_model.load_state_dict(new_state_dict)
    # '''

    test_dataloader = get_testdataset(args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    count = 0
    for idx, (rain_img, keypoints_in, clean_img_LR, clean_img_HR) in enumerate(test_dataloader):
        count = count + 1
        with torch.no_grad():
            rain_img = Variable(rain_img.cuda(), volatile=False)
            clean_img_HR = Variable(clean_img_HR.cuda())
            keypoints_in = Variable(keypoints_in.cuda())
            output, out_combine, clean_layer, add_layer, mul_layer = my_model(rain_img, keypoints_in)
            #print(output.shape)

        output = output.cpu()
        output = output.data.squeeze(0)
        out_combine = out_combine.cpu()
        out_combine = out_combine.data.squeeze(0)
        #print(output.shape)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, t1, m, s in zip(output[0], out_combine[0], mean, std):
            t.mul_(s).add_(m)
            t1.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        output = output.transpose(1, 2, 0)
        out_combine = out_combine.numpy()
        out_combine *= 255.0
        out_combine = out_combine.clip(0, 255)
        out_combine = out_combine.transpose(1, 2, 0)

        out = np.uint8(output)
        # out_pil = Image.fromarray(out, mode='RGB')
        # out_pil.save('results/out_img/out_img_%04d.jpg' % (count))
        cv2.imwrite('results/out_img/out_img_%04d.jpg' %(count), out)

        comb = np.uint8(out_combine) # clean layer - output of network
        # clean_layer_pil = Image.fromarray(clean, mode='RGB')
        # clean_layer_pil.save('results/clean_img/clean_img_%04d.jpg' % (count))
        cv2.imwrite('results/clean_img/clean_img_%04d.jpg' % (count), comb)

        # =========== Target Image ===============
        clean_img_HR = clean_img_HR.cpu()
        clean_img_HR = clean_img_HR.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(clean_img_HR, mean, std):
            t.mul_(s).add_(m)

        clean_img_HR = clean_img_HR.numpy() # clean_img - ground truth
        clean_img_HR *= 255.0
        clean_img_HR = clean_img_HR.clip(0, 255)
        clean_img_HR = clean_img_HR.transpose(1, 2, 0)
        clean_img_HR = np.uint8(clean_img_HR)

        # clean_img_pil = Image.fromarray(clean_img, mode='RGB')
        # clean_img_pil.save('results/GT/GT_%03d.png' %(count))
        cv2.imwrite('results/GT/GT_%03d.png' %(count), clean_img_HR)

        # output_shape = np.array(output).shape
        # if(np.array(clean_img).shape[0] > output_shape[0]):
        #     clean_img = np.delete(clean_img, -1, axis = 0)
        # if(np.array(clean_img).shape[1] > output_shape[1]):
        #     clean_img = np.delete(clean_img, -1, axis = 1)

        mse = ((out - clean_img_HR) ** 2).mean()
        psnr_val = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        avg_psnr += psnr_val

        ssim_val = ssim(out, clean_img_HR, data_range=255, multichannel=True, gaussian_weights=True)
        avg_ssim += ssim_val
        print('%03d_img: PSNR = %2.5f, SSIM = %2.5f'%(count, psnr_val, ssim_val))

    avg_psnr /= (count)
    avg_ssim /= (count)
    print('AVG PSNR = %2.5f, Average SSIM = %2.5f'%(avg_psnr, avg_ssim))


if __name__ == '__main__':
    test(args)