import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import argparse
import numpy as np
import time

from model_full_no_stage2_4RIM import Deraining
from data_with_grid import outdoor_rain_test, real_rain_test
#
# model = models.resnext101_32x8d(pretrained = True)
# print('model: ',model )
# #model = list(model.children())[:-1]
# #model.append(nn.Linear(512, 2))
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
# model = list(model.children())[:-2]
# #model.append(nn.Linear(512, 2))
# mymodel = nn.Sequential(*model)
# print(mymodel)



############################################# create dataset - pix2pix type
# path = 'D:/DATASETS/real_rain/new'
# path = 'D:/NewFolder/New-dataset/test/DID-MDN_gt'
# in_path = path + '/' + 'in'
# gt_path = path + '/' + 'gt'
# file_in = sorted(os.listdir(path))
# file_gt = sorted(os.listdir(path))
#
# for i in range(len(file_gt)):
#     # print("python test_simple.py --image_path assets/train_301_350/"+file_gt[i]+" --model_name mono+stereo_1024x320")
#     print("python .\demo.py --img_path D:/NewFolder/New-dataset/test/DID-MDN_gt/"+file_gt[i]+" --save_path D:/NewFolder/New-dataset/test/DID-MDN_depth/"+file_gt[i])

# for i in range(len(file_in)):
#     file_in_path = in_path + '/' + file_in[i]
#     file_gt_path = gt_path + '/' + file_gt[i]
#     img_in = cv2.imread(file_in_path)
#     img_gt = cv2.imread(file_gt_path)
#     img_pix2pix = np.concatenate((img_in, img_gt), axis=1)
#     cv2.imwrite(path+'/pix2pix/%s' %(file_in[i]), img_pix2pix)
#############################################


############################################# extractor feature from model
parser = argparse.ArgumentParser(description='Deraining')
# validation data
parser.add_argument('--data_type', type=str, default='synthetic', help='testdata type [real|synthetic]')
parser.add_argument('--val_data_dir', required=False, default='D:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param_v5')
# parser.add_argument('--val_data_dir', required=False, default='D:/DATASETS/SPANET_dataset/real_test_1000')
parser.add_argument('--real_rain_data_dir', required=False, default='D:/DATASETS/real_rain/new/in')
parser.add_argument('--valBatchSize', type=int, default=1)


# parser.add_argument('--pretrained_model', default='save/Deraining/model_S65/model_only_SCA_no_stage2/model_99_S65_only_SCA_no_stage2.pt', help='save result')
# parser.add_argument('--pretrained_model', default='save/Deraining/model_S65/model_only_add_no_stage2/model_99_S65_only_add_no_stage2.pt', help='save result')
# parser.add_argument('--pretrained_model', default='save/Deraining/model_S65/Nov06_model_149_S65_only_mul_no_stage2.pt', help='save result')
# parser.add_argument('--pretrained_model', default='save/Deraining/model_S66/model_SCA_add_no_stage2/Dec08_model_139_S66_no_stage2_SCA_add.pt', help='save result')
# parser.add_argument('--pretrained_model', default='save/Deraining/model_S66/model_add_mul_no_stage2/Dec21_model_149_S66_no_stage2_add_mul.pt', help='save result')

# parser.add_argument('--pretrained_model', default='save/Deraining/model_S66/Nov06_model_199_S66_full_model_no_stage2.pt', help='save result')
parser.add_argument('--pretrained_model', default='save/Deraining/model_1_2_3_4RIM/Feb15_model_199_S66_no_stage2_4RIM.pt', help='save result')
parser.add_argument('--save_dir_syn_data', required=False, default='./extract_syn/4RIM')
parser.add_argument('--save_dir_real_data', required=False, default='./extract_real/Dec21_epoch199')

parser.add_argument('--nchannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patch_size', type=int, default=200, help='patch size')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')

args = parser.parse_args()
print(args.pretrained_model)

def get_testdataset(args):
    if args.data_type == 'synthetic':
        data_test = outdoor_rain_test(args)
    elif args.data_type == 'real':
        data_test = real_rain_test(args)

    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def test_synthetic(args):

    my_model = Deraining(args)
    my_model = nn.DataParallel(my_model)
    my_model.cuda()
    my_model.load_state_dict(torch.load(args.pretrained_model))
    # print("model: \n",my_model)

    test_dataloader = get_testdataset(args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    count = 0
    total_time = 0

    for idx, (rain_img, keypoints_in, clean_img_LR, clean_img_HR, rain_img_name) in enumerate(test_dataloader):
        count = count + 1
        with torch.no_grad():
            rain_img = Variable(rain_img.cuda())
            clean_img_HR = Variable(clean_img_HR.cuda())
            keypoints_in = Variable(keypoints_in.cuda())
            start = time.time()
            output, out_combine, clean_layer, add_layer, mul_layer, add_res, mul_res = my_model(rain_img, keypoints_in)
            end = time.time()
            print("infer time: ", (end-start))
            total_time += end - start

        # add_res = rain_img-clean_img_HR#-rain_img
        # mul_res = rain_img/clean_img_HR#/rain_img

        save_dir = args.save_dir_syn_data
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0, 0, 0]
        std  = [1, 1, 1]
        output = output.cpu()
        output = output.data.squeeze(0)
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)
        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        output = output.transpose(1, 2, 0)
        out = np.uint8(output)
        ensure_dir(save_dir+'/out_img')
        cv2.imwrite(save_dir+'/out_img/out_%s' % (rain_img_name[0]), out)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        out_combine = out_combine.cpu()
        out_combine = out_combine.data.squeeze(0)
        for t, m, s in zip(out_combine, mean, std):
            t.mul_(s).add_(m)
        out_combine = out_combine.numpy()
        out_combine *= 255.0
        out_combine = out_combine.clip(0, 255)
        out_combine = out_combine.transpose(1, 2, 0)
        comb = np.uint8(out_combine)
        ensure_dir(save_dir+'/comb_img')
        cv2.imwrite(save_dir+'/comb_img/comb_%s' % (rain_img_name[0]), comb)  # cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        clean_layer = clean_layer.cpu()
        clean_layer = clean_layer.data.squeeze(0)
        for t, m, s in zip(clean_layer, mean, std):
            t.mul_(s).add_(m)
        clean_layer = clean_layer.numpy()
        clean_layer *= 255.0
        clean_layer += 30
        clean_layer = clean_layer.clip(0, 255)
        clean_layer = clean_layer.transpose(1, 2, 0)
        clean = np.uint8(clean_layer)
        ensure_dir(save_dir+'/clean_img')
        cv2.imwrite(save_dir+'/clean_img/clean_%s' % (rain_img_name[0]), clean)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        add_layer = add_layer.cpu()
        add_layer = add_layer.data.squeeze(0)
        for t, m, s in zip(add_layer, mean, std):
            t.mul_(s).add_(m)
        add_layer = add_layer.numpy()
        add_layer *= 255.0
        add_layer += 30
        add_layer = add_layer.clip(0, 255)
        add_layer = add_layer.transpose(1, 2, 0)
        add = np.uint8(add_layer)
        ensure_dir(save_dir+'/add_img')
        cv2.imwrite(save_dir+'/add_img/add_%s' % (rain_img_name[0]), add)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        mul_layer = mul_layer.cpu()
        mul_layer = mul_layer.data.squeeze(0)
        for t, m, s in zip(mul_layer, mean, std):
            t.mul_(s).add_(m)
        mul_layer = mul_layer.numpy()
        mul_layer *= 255.0
        mul_layer += 30
        mul_layer = mul_layer.clip(0, 255)
        mul_layer = mul_layer.transpose(1, 2, 0)
        mul = np.uint8(mul_layer)
        ensure_dir(save_dir+'/mul_img')
        cv2.imwrite(save_dir+'/mul_img/mul_%s' % (rain_img_name[0]), mul)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        add_res = add_res.cpu()
        add_res = add_res.data.squeeze(0)
        for t, m, s in zip(add_res, mean, std):
            t.mul_(s).add_(m)
        add_res = add_res.numpy()
        add_res *= 255.0
        add_res = add_res.clip(0, 255)
        add_res = add_res.transpose(1, 2, 0)
        a_res = np.uint8(add_res)
        ensure_dir(save_dir+'/add_res_img')
        cv2.imwrite(save_dir+'/add_res_img/add_%s' % (rain_img_name[0]), a_res)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        mul_res = mul_res.cpu()
        mul_res = mul_res.data.squeeze(0)
        for t, m, s in zip(mul_res, mean, std):
            t.mul_(s).add_(m)
        mul_res = mul_res.numpy()
        mul_res *= 255.0
        mul_res = mul_res.clip(0, 255)
        mul_res = mul_res.transpose(1, 2, 0)
        m_res = np.uint8(mul_res)
        ensure_dir(save_dir+'/mul_res_img')
        cv2.imwrite(save_dir+'/mul_res_img/mul_%s' % (rain_img_name[0]), m_res)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        # =========== Target Image ===============
        clean_img_HR = clean_img_HR.cpu()
        clean_img_HR = clean_img_HR.data.squeeze(0)
        for t, m, s in zip(clean_img_HR, mean, std):
            t.mul_(s).add_(m)

        clean_img_HR = clean_img_HR.numpy() # clean_img - ground truth
        clean_img_HR *= 255.0
        clean_img_HR = clean_img_HR.clip(0, 255)
        clean_img_HR = clean_img_HR.transpose(1, 2, 0)
        clean_img_HR = np.uint8(clean_img_HR)

        cv2.imwrite('results/GT/GT_%s' %(rain_img_name[0]), clean_img_HR)

        psnr_val = psnr(out, clean_img_HR, data_range=255)
        avg_psnr += psnr_val

        ssim_val = ssim(out, clean_img_HR, data_range=255, multichannel=True, gaussian_weights=True)
        avg_ssim += ssim_val
        log = "{}:\t PSNR = {:.5f}, SSIM = {:.5f}".format(rain_img_name[0], psnr_val, ssim_val)
        # print('%s: PSNR = %2.5f, SSIM = %2.5f' %(rain_img_name, psnr_val, ssim_val))
        print(log)


    avg_psnr /= (count)
    avg_ssim /= (count)
    print('AVG PSNR = %2.5f, Average SSIM = %2.5f'%(avg_psnr, avg_ssim))
    print('total time: ', total_time)

def test_real(args):

    my_model = Deraining(args)
    my_model = nn.DataParallel(my_model)
    my_model.cuda()
    my_model.load_state_dict(torch.load(args.pretrained_model))
    # print("model: \n",my_model)

    test_dataloader = get_testdataset(args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    count = 0

    for idx, (rain_img, keypoints_in, rain_img_name) in enumerate(test_dataloader):
        count = count + 1
        with torch.no_grad():
            rain_img = Variable(rain_img.cuda())
            keypoints_in = Variable(keypoints_in.cuda())
            output, out_combine, clean_layer, add_layer, mul_layer, add_res, mul_res = my_model(rain_img, keypoints_in)

        # add_res = rain_img-clean_img_HR#-rain_img
        # mul_res = rain_img/clean_img_HR#/rain_img

        output = output.cpu()
        output = output.data.squeeze(0)
        out_combine = out_combine.cpu()
        out_combine = out_combine.data.squeeze(0)
        clean_layer = clean_layer.cpu()
        clean_layer = clean_layer.data.squeeze(0)
        add_layer = add_layer.cpu()
        add_layer = add_layer.data.squeeze(0)
        mul_layer = mul_layer.cpu()
        mul_layer = mul_layer.data.squeeze(0)
        add_res = add_res.cpu()
        add_res = add_res.data.squeeze(0)
        mul_res = mul_res.cpu()
        mul_res = mul_res.data.squeeze(0)
        #print(output.shape)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0, 0, 0]#[0.5, 0.5, 0.5]
        std  = [1, 1, 1]#[0.5, 0.5, 0.5]
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(out_combine, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(clean_layer, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(add_layer, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(mul_layer, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(add_res, mean, std):
            t.mul_(s).add_(m)
        for t, m, s in zip(mul_res, mean, std):
            t.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        output = output.transpose(1, 2, 0)
        out_combine = out_combine.numpy()
        out_combine *= 255.0
        out_combine = out_combine.clip(0, 255)
        out_combine = out_combine.transpose(1, 2, 0)
        clean_layer = clean_layer.numpy()
        clean_layer *= 255.0
        clean_layer = clean_layer.clip(0, 255)
        clean_layer = clean_layer.transpose(1, 2, 0)
        add_layer = add_layer.numpy()
        add_layer *= 255.0
        add_layer = add_layer.clip(0, 255)
        add_layer = add_layer.transpose(1, 2, 0)
        mul_layer = mul_layer.numpy()
        mul_layer *= 255.0
        mul_layer = mul_layer.clip(0, 255)
        mul_layer = mul_layer.transpose(1, 2, 0)
        add_res = add_res.numpy()
        add_res *= 255.0
        add_res = add_res.clip(0, 255)
        add_res = add_res.transpose(1, 2, 0)
        mul_res = mul_res.numpy()
        mul_res *= 255.0
        mul_res = mul_res.clip(0, 255)
        mul_res = mul_res.transpose(1, 2, 0)

        save_dir = args.save_dir_real_data
        out = np.uint8(output)
        ensure_dir(save_dir + '/out_img')
        cv2.imwrite(save_dir + '/out_img/out_%s' %(rain_img_name[0]), out)#cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        comb = np.uint8(out_combine)
        ensure_dir(save_dir + '/comb_img')
        cv2.imwrite(save_dir + '/comb_img/comb_%s' %(rain_img_name[0]), comb)#cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        clean = np.uint8(clean_layer)
        ensure_dir(save_dir + '/clean_img')
        cv2.imwrite(save_dir + '/clean_img/clean_%s' % (rain_img_name[0]), clean)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        add = np.uint8(add_layer)
        ensure_dir(save_dir + '/add_img')
        cv2.imwrite(save_dir + '/add_img/add_%s' % (rain_img_name[0]), add)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        mul = np.uint8(mul_layer)
        ensure_dir(save_dir + '/mul_img')
        cv2.imwrite(save_dir + '/mul_img/mul_%s' % (rain_img_name[0]), mul)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        a_res = np.uint8(add_res)
        ensure_dir(save_dir + '/add_res_img')
        cv2.imwrite(save_dir + '/add_res_img/add_%s' % (rain_img_name[0]), a_res)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        m_res = np.uint8(mul_res)
        ensure_dir(save_dir + '/mul_res_img')
        cv2.imwrite(save_dir + '/mul_res_img/mul_%s' % (rain_img_name[0]), m_res)  # cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        print(rain_img_name[0])

if __name__ == '__main__':
    if args.data_type == 'synthetic':
        test_synthetic(args)
    elif args.data_type == 'real':
        test_real(args)