from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
from os.path import join, splitext, basename
from glob import glob
import math

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_recon(Dataset):
    # prepare data for self-reconstruction, 
    # where the two input photos and the intermediate region are obtained from the same image 

    def __init__(self, root='', transforms=None, crop='rand', imgSize=256):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None
    
        file_list = sorted(glob(join(root, '*.png')))

        for name in file_list:
            img = Image.open(name)
            if (img.size[0]>= (self.imgSize * 3)) and (img.size[1] >= self.imgSize):
                self.img_list += [name]
            
        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        img = Image.open(self.img_list[index])

        if self.cropFunc is not None:
            img = self.cropFunc(img, self.imgSize, 3*self.imgSize)
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        img_split = torch.stack(torch.split(img, self.imgSize, dim=2))
            
        return img_split[0], img_split[2], img_split[1]
    
    def __len__(self):
        return self.size
    
class dataset_diff(Dataset):
    # prepare data for our WRIB objective, 
    # where the two input photos and the intermediate region are obtained from different images

    def __init__(self, root='', transforms=None, crop='center', imgSize=256, width=1, select_root=None, select_k=3):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize & width: the size of each returned image is imgSize x (imgSize*width) if crop is not 'none'
        # select_root: the path for the pre-calculated metric for selecting image pairs, if selected_root = None, the image pairs will be randomly selected 
        # select_k: the top-k similar images for each image will be selected to form image pairs, i.e. the number of image pairs used in training/testing will be (num of total images * k)

        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.width = width
        self.select_k = select_k
        self.rand_select = True if select_root is None else False
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None
    
        file_list = sorted(glob(join(root, '*.png')))
        
        for idx, name in enumerate(file_list):
            img = Image.open(name)
            if (img.size[0]>= (self.imgSize * self.width)) and (img.size[1] >= self.imgSize):
                self.img_list += [name]

        if self.rand_select is False:
            metric = torch.load(select_root)
            ind = np.diag_indices(metric.shape[0])
            metric[ind[0], ind[1]] = torch.ones(metric.shape[0])
            _, self.pair_list = torch.topk(metric, k=select_k, dim=1, largest=False) 
            
        self.size = len(self.img_list) * self.select_k

    def __getitem__(self, index):
        # --RETURN--
        # input1, input2 (which are from differnt images)

        index = index % self.size
        I1_idx = math.floor(index / self.select_k)
        if self.rand_select is False:
            I2_idx = self.pair_list[I1_idx][index % self.select_k]
        else:
            weights = torch.ones(len(self.img_list))
            weights[I1_idx] = 0
            I2_idx = torch.multinomial(weights, num_samples=1, replacement=False)[0]
        
        img1 = Image.open(self.img_list[I1_idx])
        img2 = Image.open(self.img_list[I2_idx])
        if self.cropFunc is not None:
            img1 = self.cropFunc(img1, self.imgSize, self.imgSize * self.width)
            img2 = self.cropFunc(img2, self.imgSize, self.imgSize * self.width)
        
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            
        return img1, img2
    
    def __len__(self):
        return self.size
    
class dataset_complete(Dataset):
    # prepare data for the discriminator

    def __init__(self, root='', transforms=None, crop='rand', imgSize=256, width=3):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize & width: the size of each returned image is imgSize x (imgSize*width) if crop is not 'none'
        
        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.width = width
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None
    
        file_list = sorted(glob(join(root, '*.png')))
        
        for name in file_list:
            img = Image.open(name)
            if (img.size[0]>= (self.imgSize * self.width)) and (img.size[1] >= self.imgSize):
                self.img_list += [name]
            
        self.size = len(self.img_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        img = Image.open(self.img_list[index])
        if self.cropFunc is not None:
            img = self.cropFunc(img, self.imgSize, self.imgSize * self.width)
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img
    
    def __len__(self):
        return self.size

class dataset_test(Dataset):
    # prepare data for testing

    def __init__(self, root1='', root2='', transforms=None, crop='rand', rand_pair=False, imgSize=256):
        # --PARAMS--
        # root1: the path of the data for image1
        # root2: the path of the data for image2
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img1_list = []
        self.img2_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.rand_pair = rand_pair
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None
    
        file_list1 = sorted(glob(join(root1, '*.png')))
        file_list1.extend(sorted(glob(join(root1, '*.jpg'))))
        file_list2 = sorted(glob(join(root2, '*.png')))
        file_list2.extend(sorted(glob(join(root2, '*.jpg'))))

        if self.cropFunc is not None:
            for name in file_list1:
                img = Image.open(name)
                if (img.size[0]>= (self.imgSize)) and (img.size[1] >= self.imgSize):
                    self.img1_list += [name]
            for name in file_list2:
                img = Image.open(name)
                if (img.size[0]>= (self.imgSize)) and (img.size[1] >= self.imgSize):
                    self.img2_list += [name]
        else:
            self.img1_list = file_list1
            self.img2_list = file_list2
            
        self.size = min(len(self.img1_list), len(self.img1_list))

    def __getitem__(self, index):
        # --RETURN--
        # input1, input2

        index = index % self.size
        name = self.img1_list[index]
        img1 = Image.open(name)
        if self.rand_pair:
            index2 = np.random.randint(len(self.img2_list), size=1)[0]
            img2 = Image.open(self.img2_list[index2])
        else:
            img2 = Image.open(self.img2_list[index])

        if self.cropFunc is not None:
            img1 = self.cropFunc(img1, self.imgSize, self.imgSize)
            img2 = self.cropFunc(img2, self.imgSize, self.imgSize)
        
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            
        return img1, img2, splitext(basename(name))[0]
    def __len__(self):
        return self.size
   