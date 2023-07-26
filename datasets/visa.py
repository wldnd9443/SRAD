import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
# from utils import bicubic_sampling


# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=1.):
#         self.std = std
#         self.mean = mean
         
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean
     
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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
                 patch_size=128,
                 scale_factor=2,
                 aug_mode=False
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.aug_mode = aug_mode
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(self.resize, Image.BICUBIC),
            transforms.CenterCrop(self.resize),
            #transforms.RandomCrop((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_aug = transforms.Compose([
            #transforms.ToTensor(),
            transforms.RandomApply([
                #AddGaussianNoise(0.1,0.08),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((-90, -90))
                ], p=0.5)
        ])

        self.transform_randomcrop = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(self.patch_size),
            #transforms.ToTensor()
        ])

        self.transform_bicubic = transforms.Compose([
            transforms.Resize(int(self.patch_size/self.scale_factor), Image.BICUBIC)

        ])
        
        self.transform_mask = transforms.Compose(
            [transforms.Resize(self.resize, Image.NEAREST),
             transforms.CenterCrop(self.resize),
             transforms.ToTensor()])
        
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        
        if self.aug_mode:
            x = self.transform_aug(x)
        x = self.transform_x(x)

        hr_patch = self.transform_randomcrop(x)
        lr_patch = self.transform_bicubic(hr_patch)
        # lr_patch = lr_patch.clamp(-1,1)
        if y == 0:
            mask = torch.zeros([1, self.resize,self.resize])
        else: 
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask, hr_patch, lr_patch 
    
    def __len__(self):
        return len(self.x)
    

    # def bicubic_sampling(self, tensor, scale_factor):
    #     """
    #         input: 4D tensor(including batch)
    #         output: tensor 
    #     """
    #     tensor = tensor.to(torch.float32)
    #     result = F.interpolate(tensor, scale_factor=scale_factor,mode="bicubic")
    #     result = result.to(torch.uint8)
    #     return result

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

# if __name__ == "__main__":
#     dataset = VisaDataset()
#     print(dataset[0][3].shape)
