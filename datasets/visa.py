import os
#import tarfile
from PIL import Image
#import urllib.request
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = [
    'candle', 
    'capsules', 
    'cashew', 
    'chewinggum', 
    'fryum', 
    'macaroni1',
    'macaroni2',
    'pcb1',
    'pcb2',
    'pcb3',
    'pcb4',
    'pipe_fryum'
]

class VisaDataset(Dataset):
    def __init__(self,
                 dataset_path='./data/VisA/1cls',
                 class_name='candle',
                 is_train=True,
                 resize=256,
                 aug_mode=False
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.aug_mode = aug_mode
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.LANCZOS),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-90,90))
        ])

        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST),
             transforms.CenterCrop(resize),
             transforms.ToTensor()])
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        if self.aug_mode:
            x = self.transform_aug(x)
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize,self.resize])
        else: 
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask
    
    def __len__(self):
        return len(self.x)
    
    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [],[],[]

        img_dir = os.path.join(self.dataset_path,self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name,'ground_truth')
        #import pdb; pdb.set_trace()
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)

            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.JPG')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

