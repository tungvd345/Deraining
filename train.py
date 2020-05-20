# --------------------------
# Deraining
# Vu Dac Tung
# 2020-03-11
# --------------------------
from __future__ import print_function

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
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, drop_last=True, shuffle=False,
                                             num_workers=int(opt.nThreads), pin_memory=False)
    return dataloader

def get_testdataset(opt):
    data_test = outdoor_rain_test(opt)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=opt.val_batch_size,
                                             drop_last=True, shuffle=False, num_workers=int(opt.nThreads), pin_memory=False)
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

def test(args, model, dataloader):
    # print('test-------------')
    avg_psnr = 0
    mse = mse1 = psnr = psnr1 = np.zeros(args.val_batch_size)
    avg_psnr1 = 0
    psnr_val = 0
    ###############################################
    # writer = SummaryWriter()
    # dataiter = iter(dataloader)
    # # rain_img_tb, clean_img_tb = dataiter.next()
    ###############################################

    for idx, (rain_img, clean_img) in enumerate(dataloader):
        # print('inx:', batch)
        with torch.no_grad():
            rain_img = Variable(rain_img.cuda(), volatile=False)
            clean_img = Variable(clean_img.cuda())
            output, out_combine, clean_layer, add_layer, mul_layer = model(rain_img)

        output = output.cpu()
        output = output.data.squeeze(0)
        out_combine = out_combine.cpu()
        out_combine = out_combine.data.squeeze(0)

        # denormalization
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
        clean_img = clean_img.cpu()
        clean_img = clean_img.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(clean_img, mean, std):
            t.mul_(s).add_(m)

        clean_img = clean_img.numpy()
        clean_img *= 255.0
        clean_img = clean_img.clip(0, 255)
        # im_hr = Image.fromarray(np.uint8(im_hr[0]), mode='RGB')

        ###############################
        # if idx < 2:
        #     writer.add_image('test rain image', np.uint8(output), idx)
        #     writer.add_image('test rain1 image', np.uint8(clean_layer), idx)
        #     writer.add_image('test clean image', np.uint8(clean_img), idx)
        #     writer.close()
        ################################
        for i in range(args.val_batch_size):
            mse[i] = ((clean_img[i, :, 8:-8,8:-8] - output[i, :, 8:-8,8:-8]) ** 2).mean()
            psnr[i] = 10 * log10(255 * 255 / (mse[i] + 10 ** (-10)))
            avg_psnr += psnr[i]

            mse1[i] = ((clean_img[i, :, 8:-8, 8:-8] - out_combine[i, :, 8:-8, 8:-8]) ** 2).mean()
            psnr1[i] = 10 * log10(255 * 255 / (mse1[i] + 10 ** (-10)))
            avg_psnr1 += psnr1[i]

    avg_psnr = avg_psnr / (args.val_batch_size*len(dataloader))
    avg_psnr1 = avg_psnr1 / (args.val_batch_size*len(dataloader))
    return avg_psnr, avg_psnr1

def train(opt, train_dataloader, test_dataloader, model):
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

    loss_function = set_loss(opt)
    loss_function.cuda()
    total_loss = 0
    total_time = 0

    #########################
    writer_t = SummaryWriter()
    dataiter = iter(train_dataloader)
    rain_img_tb, clean_img_tb = dataiter.next()
    # img_grid = torchvision.utils.make_grid(rain_img)
    ##########################
    for epoch in range(start_epoch, opt.epochs):
        start = time.time()
        optimizer = optim.Adam(model.parameters())
        learning_rate = set_lr(opt, epoch, optimizer)
        total_loss_ = 0
        loss_ = 0

        for idx, (rain_img, clean_image) in enumerate(train_dataloader):
            # print('*'*10)
            rain_img = Variable(rain_img.cuda())
            clean_image = Variable(clean_image.cuda())

            model.zero_grad()
            output, out_combine, clean_layer, add_layer, mul_layer = model(rain_img)
            # print(model)
            loss = loss_function(output, clean_image)
            # loss_clean = loss_function(clean_layer, clean_image)
            # loss_add = loss_function(add_layer, clean_image)
            # loss_mul = loss_function(mul_layer, clean_image)
            # print('out img', output)
            # print(clean_image)
            # total_loss = loss + loss_clean + loss_add + loss_mul
            total_loss = loss
            total_loss.backward()
            optimizer.step()

            # loss_ += loss.data.cpu().numpy()
            total_loss_ += total_loss.data.cpu().numpy()

        with torch.no_grad():
            # loss_ = loss_ / (idx * opt.batch_size)
            total_loss_ = total_loss_ / ((idx + 1) * opt.batch_size)
            writer_t.add_scalar('Loss', total_loss_, epoch + 1)

            #############################################
            rain_img_tb = rain_img_tb.cuda()
            clean_img_tb = clean_img_tb.cuda()
            output_tb, _, _, _, _ = model(rain_img_tb)
            writer_t.add_images('rain image', output_tb, epoch)
            writer_t.add_images('clean image', clean_img_tb, epoch)
            ############################################

            end = time.time()
            epoch_time = (end - start)
            total_time = total_time + epoch_time

            if(epoch + 1) % opt.period == 0:
                model.eval()
                avg_psnr, avg_psnr1 = test(opt, model, test_dataloader)
                writer_t.add_scalar('PSNR', avg_psnr, epoch+1)
                writer_t.add_scalar('PSNR1', avg_psnr1, epoch+1)
                model.train()
                log = "[{} / {}] \tLearning_rate: {:.8f}\t Train total_loss: {:.4f}\t Train Loss: {:.4f} \t Val PSNR: {:.4f} \t Val PSNR1: {:.4f} Time: {:.4f}".format(
                    epoch + 1, opt.epochs, learning_rate, total_loss_, total_loss_, avg_psnr, avg_psnr1, total_time)
                print(log)
                save.save_log(log)
                save.save_model(model, epoch)
                total_time = 0
    writer_t.close()

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

    from helper import *
    from model import Deraining
    from data import outdoor_rain_train, outdoor_rain_test
    from options import TrainOptions
    from torch.utils.tensorboard import SummaryWriter

    opt = TrainOptions().parse()
    model = Deraining(opt)
    train_data = get_dataset(opt)
    test_data = get_testdataset(opt)

    train(opt, train_data, test_data, model)

