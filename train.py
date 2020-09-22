# --------------------------
# Deraining
# Vu Dac Tung
# 2020-03-11
# --------------------------
from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    # print('*'*10)
    if classname.find('Conv2d') != -1:
        # print(classname)
        # print(m.weight.data.size())
        init.xavier_normal_(m.weight.data)

def get_dataset(opt):
    data_train = outdoor_rain_train(opt)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, drop_last=True, shuffle=True,
                                             num_workers=int(opt.nThreads), pin_memory=True)
    return dataloader

def get_testdataset(opt):
    data_test = outdoor_rain_test(opt)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=opt.val_batch_size,
                                             drop_last=True, shuffle=False, num_workers=int(opt.nThreads), pin_memory=True)
    return dataloader

def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def set_loss(opt):
    lossType = opt.lossType
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# writer = SummaryWriter()
def test(opt, model, dataloader):
    # print('test-------------')
    opt.phase = 'test'
    avg_psnr = 0
    mse = mse1 = psnr = psnr1 = np.zeros(opt.val_batch_size)
    avg_psnr1 = 0
    psnr_val = 0
    loss_val = 0
    ###############################################
    # writer = SummaryWriter()
    # dataiter = iter(dataloader)
    # rain_img_tr_tb, keypoints_tr_tb, clean_img_LR_tr_tb, clean_img_tr_tb  = dataiter.next()
    ###############################################

    for idx, (rain_img, keypoints_in, clean_img_LR, clean_img_HR, rain_img_name) in enumerate(dataloader):
        # print('inx:', batch)
        with torch.no_grad():
            rain_img = Variable(rain_img.cuda(), volatile=False)
            keypoints_in = Variable(keypoints_in.cuda())
            clean_img_LR = Variable(clean_img_LR.cuda())
            clean_img_HR = Variable(clean_img_HR.cuda())
            output, out_combine, clean_layer, add_layer, mul_layer = model(rain_img, keypoints_in)

        loss_function = set_loss(opt)
        loss_function.cuda()
        loss = loss_function(output, clean_img_HR)
        # loss_stage1 = loss_function(out_combine, clean_img_LR)
        loss_ssim = 1 - ssim((output + 1) / 2, (clean_img_HR + 1) / 2, data_range=1, size_average=True)
        loss_val += (loss_ssim + loss).cpu().numpy()

        output = output.cpu()
        output = output.data.squeeze(0)
        out_combine = out_combine.cpu()
        out_combine = out_combine.data.squeeze(0)

        # denormalization
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t,t1, m, s in zip(output, out_combine, mean, std):
            t.mul_(s).add_(m)
            t1.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        out_combine = out_combine.numpy()
        out_combine *= 255.0
        out_combine = out_combine.clip(0, 255)
        # output = Image.fromarray(np.uint8(output[0]), mode='RGB')

        # =========== Target Image ===============
        clean_img_HR = clean_img_HR.cpu()
        clean_img_HR = clean_img_HR.data.squeeze(0)
        clean_img_LR = clean_img_LR.cpu()
        clean_img_LR = clean_img_LR.data.squeeze(0)
        for t1, t2, m, s in zip(clean_img_HR, clean_img_LR, mean, std):
            t1.mul_(s).add_(m)
            t2.mul_(s).add_(m)

        clean_img_HR = clean_img_HR.numpy()
        clean_img_HR *= 255.0
        clean_img_HR = clean_img_HR.clip(0, 255)
        # im_hr = Image.fromarray(np.uint8(im_hr[0]), mode='RGB')
        clean_img_LR = clean_img_LR.numpy()
        clean_img_LR *= 255.0
        clean_img_LR = clean_img_LR.clip(0, 255)

        mse = ((clean_img_HR[:, 8:-8,8:-8] - output[:, 8:-8,8:-8]) ** 2).mean()
        psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        avg_psnr += psnr

        mse1 = ((clean_img_LR[:, 8:-8, 8:-8] - out_combine[:, 8:-8, 8:-8]) ** 2).mean()
        psnr1 = 10 * log10(255 * 255 / (mse1 + 10 ** (-10)))
        avg_psnr1 += psnr1

    total_loss_val = loss_val / ((idx + 1) * opt.batch_size)
    avg_psnr = avg_psnr / (opt.val_batch_size * len(dataloader))
    avg_psnr1 = avg_psnr1 / (opt.val_batch_size * len(dataloader))
    return avg_psnr, avg_psnr1, total_loss_val

def train(opt, train_dataloader, test_dataloader, model):
    opt.phase = 'train'
    model = nn.DataParallel(model)
    model.apply(weights_init)
    model.cuda()

    save = saveData(opt)
    Numparams = count_parameters(model)
    print('Number of param = ', Numparams)

    last_epoch = 0
    # if opt.finetuning:
    #     model.load_state_dict(torch.load(opt.pretrained_model))
    start_epoch = last_epoch

    vgg = Vgg16()
    vgg.cuda()
    mse_loss = nn.MSELoss()
    mse_loss.cuda()
    l1_loss = nn.L1Loss()
    l1_loss.cuda()
    loss_function = set_loss(opt)
    loss_function.cuda()
    total_loss = 0
    total_time = 0


    #########################
    writer = SummaryWriter()
    dataiter_tr = iter(test_dataloader) #  dataset
    rain_img_tr_tb, keypoints_tr_tb, clean_img_LR_tr_tb, clean_img_tr_tb, _ = dataiter_tr.next() #tensorboard
    # dataiter_val = iter(test_dataloader) # val dataset
    # rain_img_val_tb, keypoints_val_tb, clean_img_LR_val_tb, clean_img_val_tb = dataiter_val.next() #tensorboard

    # transform_list = [transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))]
    #                    # transforms.Normalize((0, 0, 0), (1/255, 1/255, 1/255))]
    # denormalize = transforms.Compose(transform_list)
    # for i in range(tmp):
    #     clean_tmp[i] = denormalize(clean_img_tr_tb[i])
    #     rain_tmp[i] = denormalize(rain_img_tr_tb[i])
    # writer.add_images('image', clean_tmp, 0)
    # writer.add_images('image', rain_tmp, 1)
    ##########################

    date_time = str(datetime.datetime.now())
    save.save_log(date_time)
    for epoch in range(start_epoch, opt.epochs):
        start = time.time()
        optimizer = optim.Adam(model.parameters())
        learning_rate = set_lr(opt, epoch, optimizer)
        total_loss_ = 0
        loss_ = 0

        for idx, (rain_img, keypoints_in, clean_image_LR, clean_image_HR) in enumerate(train_dataloader):
            # print('*'*10)
            start_iter = time.time()
            rain_img = Variable(rain_img.cuda())
            keypoints_in = Variable(keypoints_in.cuda())
            clean_image_LR = Variable(clean_image_LR.cuda())
            clean_image_HR = Variable(clean_image_HR.cuda())
            # t1 = time.time() - start_iter
            model.zero_grad()
            # t2 = time.time() - t1 - start_iter
            output, out_combine, clean_layer, add_layer, mul_layer = model(rain_img, keypoints_in)
            # t3 = time.time() - t2 - t1 - start_iter

            loss = loss_function(output, clean_image_HR)
            grad_h_est, grad_v_est = gradient(output)
            grad_h_gt, grad_v_gt = gradient(clean_image_HR)
            loss_edge = l1_loss(grad_h_est, grad_h_gt) + l1_loss(grad_v_est, grad_v_gt)
            # t4 = time.time() - t3 - t2 - t1 - start_iter
            # loss_stage1 = loss_function(out_combine, clean_image_LR)
            loss_ssim = 1 - ssim((output+1)/2, (clean_image_HR+1)/2, data_range=1, size_average=True)
            # feature_output = vgg(output)
            # feature_GT_HR = vgg(clean_image_HR)
            # loss_vgg = mse_loss(feature_output.relu3_3, feature_GT_HR.relu3_3)
            loss_clean = loss_function(clean_layer, clean_image_LR)
            loss_add = loss_function(add_layer, clean_image_LR)
            loss_mul = loss_function(mul_layer, clean_image_LR)
            # total_loss = loss + loss_clean + loss_add + loss_mul
            total_loss = loss + loss_ssim + loss_edge + (loss_clean+loss_mul+loss_add)# + loss_stage1 + loss_vgg
            total_loss.backward()
            optimizer.step()

            # loss_ += loss.data.cpu().numpy()
            total_loss_ += total_loss.data.cpu().numpy()
            end_iter = time.time()
            iter_time = end_iter - start_iter
        save.save_model(model, epoch)

        with torch.no_grad():
            # loss_ = loss_ / (idx * opt.batch_size)
            total_loss_ = total_loss_ / ((idx + 1) * opt.batch_size)
            writer.add_scalar('Loss', total_loss_, epoch)

            #############################################
            rain_img_tr_tb = rain_img_tr_tb.cuda()
            # clean_img_val_tb = clean_img_val_tb.cuda()
            keypoints_tr_tb = keypoints_tr_tb.cuda()
            output_tr_tb, _, _, _, _ = model(rain_img_tr_tb, keypoints_tr_tb)

            clean_tmp = clean_img_tr_tb
            output_tmp = output_tr_tb
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            tmp = clean_tmp.size(0)
            for i in range(tmp):
                for t1, t2, m, s in zip(clean_tmp[i], output_tmp[i], mean, std):
                    t1.mul_(s).add_(m)
                    t2.mul_(s).add_(m)

            permute = [2, 1, 0] # permute RGB --> BGR
            writer.add_images('rain image', output_tmp[:, permute, :, :], epoch)
            # cv2.imwrite('results/rain_img_%04d.png' % (epoch), np.uint8(output_tmp[0, permute, :, :] * 255.0))
            writer.add_images('clean image', clean_tmp[:, permute, :, :], epoch)

            ############################################
            end = time.time()
            epoch_time = (end - start)
            total_time = total_time + epoch_time

            if(epoch + 1) % opt.period == 0:
                model.eval()
                avg_psnr, avg_psnr1, loss_val = test(opt, model, test_dataloader)
                writer.add_scalar('Loss_Val', loss_val, epoch)
                writer.add_scalar('PSNR', avg_psnr, epoch)
                writer.add_scalar('PSNR1', avg_psnr1, epoch)

                model.train()
                log = "[{} / {}] \tLearning_rate: {:.8f}\t Train total_loss: {:.4f}\t Val Loss: {:.4f} \t Val PSNR: {:.4f} \t Val PSNR1: {:.4f} Time: {:.4f}".format(
                    epoch, opt.epochs, learning_rate, total_loss_, loss_val, avg_psnr, avg_psnr1, total_time)
                print(log)
                save.save_log(log)
                save.save_model(model, epoch)
                total_time = 0
    writer.close()

if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torchvision

    from torch.autograd import Variable
    import numpy as np
    from torch.nn import init
    import torch.optim as optim
    import math
    from math import log10
    import time
    import datetime
    import cv2
    from pytorch_msssim import ssim, ms_ssim

    from helper import *
    from model import Deraining
    from data import outdoor_rain_train, outdoor_rain_test
    from options import TrainOptions

    opt = TrainOptions().parse()
    torch.manual_seed(opt.seed)

    model = Deraining(opt)
    train_data = get_dataset(opt)
    test_data = get_testdataset(opt)

    train(opt, train_data, test_data, model)

