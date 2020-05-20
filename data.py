import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import glob
from PIL import Image
import torchvision.transforms as transforms

def RGB_np2tensor(imgIn, imgTar, channel):
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        imgTar = np.sum(imgTar * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn/255.0 - 0.5) * 2
    imgTar = (imgTar/255.0 - 0.5) * 2

    return imgIn, imgTar

def getPatch(imgIn, imgTar, args):
    (ih, iw, c) = imgIn.shape
    patch_size_w = (args.patch_size//2) * 3
    patch_size_h = args.patch_size # HR image patch size
    ix = random.randrange(0, iw - patch_size_w + 1)
    iy = random.randrange(0, ih - patch_size_h + 1)
    imgIn = imgIn[iy:iy + patch_size_h, ix:ix + patch_size_w, :]
    imgTar = imgTar[iy:iy + patch_size_h, ix:ix + patch_size_w, :]

    return imgIn, imgTar

def get_keypoints(pos, imgIn, keypoint_size):
    # imgIn = np.pad(imgIn, ([0, keypoint_size], [0, keypoint_size//2*3], [0,0]), 'constant',constant_values=0)
    imgIn = np.pad(imgIn, ([0, keypoint_size], [0, keypoint_size//2*3], [0,0]), 'edge')
    keypoint_size = keypoint_size
    keypoints = np.zeros([keypoint_size, keypoint_size//2*3,3*128])
    for i in range(128):
        keypoints[:,:,i*3:i*3+3] = imgIn[pos[i,1] : pos[i,1]+keypoint_size, pos[i,0] : pos[i,0]+keypoint_size//2*3, :]

    return keypoints

def augment(imgIn, imgTar):
    if random.random() < 0.3: # horizontal flip
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if random.random() < 0.3: # vertical flip
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]

    rot = random.randint(0, 3) # rotate
    imgIn = np.rot90(imgIn, rot, (0, 1))
    imgTar = np.rot90(imgTar, rot, (0, 1))

    return imgIn, imgTar

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

class outdoor_rain_train(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.crop = transforms.RandomCrop(args.patch_size)
        apath = args.data_dir

        ''' read data base on os.path
        dir_rainy = 'rain'
        dir_clear = 'norain'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)
        '''

        ''' read data base on glob - in multi file
        # dir_rainy = 'rain/*.png'
        # dir_clear = 'norain/*.png'
        # self.file_in_list = sorted(glob.glob(self.dir_in))
        # self.file_tar_list = sorted(glob.glob(self.dir_tar))
        '''

        # self.file_in_list = sorted(make_dataset(self.dir_in), key=lambda x: x[:-6])
        # self.file_tar_list = sorted(make_dataset(self.dir_tar))

        # self.dir = args.data_dir
        dir_rainy = 'in'
        dir_clear = 'gt'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)

        self.file_in_list = sorted(make_dataset(self.dir_in))
        self.file_tar_list = sorted(make_dataset(self.dir_tar))
        self.transform = get_transform(args)
        self.len = len(self.file_in_list)

    def __getitem__(self, idx):
        args = self.args
        img_in = cv2.imread(self.file_in_list[idx])
        img_tar = cv2.imread(self.file_tar_list[idx//15])
        img_in, img_tar = augment(img_in, img_tar)
        if args.need_patch:
            img_in, img_tar = getPatch(img_in, img_tar, self.args)
        img_in, img_tar = RGB_np2tensor(img_in, img_tar, args.nchannel)

        return img_in, img_tar

        #################### read dataset in JORDER
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx]).convert("RGB")
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        ################################################

        ################## read dataset in DID-MDN
        # img_in_tar = Image.open(self.file_list[idx]).convert("RGB")
        # img_in_tar = self.transform(img_in_tar)
        # c, h, w = img_in_tar.size()
        # img_in = img_in_tar[:, :, :w//2]
        # img_tar = img_in_tar[:, :, w//2:]
        # if args.need_patch:
        #     img_in, img_tar = getPatch(img_in, img_tar, args)
        ###############################################################

        # #####################read data heavy rain
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx // 15]).convert("RGB")
        #
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        # if args.need_patch:
        #     img_in, img_tar = getPatch(img_in, img_tar, args)
        # ###############################################################
        # return img_in, img_tar


    def __len__(self):
        return self.len

    def get_file_name(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar

class outdoor_rain_test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.channel = args.nchannel
        apath = args.val_data_dir

        # dir_rainy = 'rain/X2'
        # dir_clear = 'norain'
        # self.dir_in = os.path.join(apath, dir_rainy)
        # self.dir_tar = os.path.join(apath, dir_clear)
        # self.file_in_list = sorted(make_dataset(self.dir_in), key=lambda x: x[:-6])
        # self.file_tar_list = sorted(make_dataset(self.dir_tar))

        dir_rainy = 'in'
        dir_clear = 'gt'
        self.dir_in = os.path.join(apath, dir_rainy)
        self.dir_tar = os.path.join(apath, dir_clear)
        self.file_in_list = sorted(make_dataset(self.dir_in))
        self.file_tar_list = sorted(make_dataset(self.dir_tar))

        # self.file_list = sorted(make_dataset(apath))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len = len(self.file_in_list)

    def __getitem__(self, idx):
        args = self.args
        #####################read data in DID-MDN
        # img_in_tar = Image.open(self.file_list[idx]).convert('RGB')
        # img_in_tar = self.transform(img_in_tar)
        # c, h, w = img_in_tar.size()
        # img_in = img_in_tar[:, :, :w // 2]
        # img_tar = img_in_tar[:, :, w // 2:]
        #################################################################

        ##################### use PIL image
        # img_in = Image.open(self.file_in_list[idx]).convert("RGB")
        # img_tar = Image.open(self.file_tar_list[idx // 15]).convert("RGB")
        # img_in = self.transform(img_in)
        # img_tar = self.transform(img_tar)
        ##################################################################

        ##################### use cv2 image
        img_in = cv2.imread(self.file_in_list[idx])
        img_tar = cv2.imread(self.file_tar_list[idx // 15])
        img_in, img_tar = RGB_np2tensor(img_in, img_tar, args.nchannel)
        ##################################################################
        return img_in, img_tar

    def __len__(self):
        return self.len

    def get_file_name(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        # osize = [opt.loadSizeX, opt.loadSizeY]
        osize = [opt.fineSize, opt.fineSize]
        # transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        # transform_list.append(transforms.RandomCrop(opt.patch_size))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop((opt.patch_size, opt.patch_size//2*3)))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        print("augment is ok")
        if random.random() < 0.3:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif random.random() < 0.6:
            transform_list.append(transforms.RandomVerticalFlip())
        else:
            transform_list.append(transforms.RandomRotation((0,360)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
