# This code is based on this github: https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_dataloader.py

import random
import glob
import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from PIL import Image

class REIDdataset(Dataset):
    def __init__(self, cfg, path, transform=None):
        self.imageFolderDataset = dset.ImageFolder(root=path)
        self.transform = transform
        self.cfg = cfg
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    def __getitem__(self, index):
        # Get a random iamge which will be used as an anchor
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # img0_tuple[1] is class_label
        # img0_tuple = (img_path, class_id)
        
        while True:
            # keep looping till a different class(Negative) image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img1_tuple[1]:
                # If class id is different than anchor image, this will be used as a negative image
				# Exit the loop if a negative image is found
                break
        
        # Getting anchor class and negative class
        anchor_class_name = img0_tuple[0].split("/")[-2]
        negative_class_name = img1_tuple[0].split("/")[-2]
        
        if int(anchor_class_name) < self.cfg.DATA.CLASS_SEP:
            anchor_class_b = 0
        else:
            anchor_class_b = 1
        
        if int(negative_class_name) < self.cfg.DATA.CLASS_SEP:
            negative_class_b = 0
        else:
            negative_class_b = 1
        
        anchor_class = torch.Tensor([anchor_class_b]).long()
        negative_class = torch.Tensor([negative_class_b]).long()
        
        # Getting all the images which belong to the same class as anchor image.
        all_files_in_class = glob.glob(self.imageFolderDataset.root+anchor_class_name+'/*')
		# Only those images which belong to the same class as anchor image but isn't anchor image will 
		# be selected as a candidate for positive sample
        all_files_in_class = [x for x in all_files_in_class if x!=img0_tuple[0]]
        
        if len(all_files_in_class) == 0:
            # If there is no image (other than anchor image) belonging to the anchor image class, anchor 
			# image will be taken as positive sample
            positive_image = img0_tuple[0]
        else:
            # Choose random image (of same class as anchor image) as positive sample
            positive_image = random.choice(all_files_in_class)
        
        if anchor_class_name != positive_image.split("/")[-2]:
            print("FAIL TO Retrieving the same class (postive) image as anchor")
        
        
        anchor = np.array(Image.open(img0_tuple[0]).convert("RGB"))
        positive = np.array(Image.open(positive_image).convert("RGB"))
        negative = np.array(Image.open(img1_tuple[0]).convert("RGB"))
        
        if self.transform is not None:
            t_anchor = self.transform(image=anchor)
            t_positive = self.transform(image=positive)
            t_negative = self.transform(image=negative)
            
            anchor = t_anchor["image"]
            positive = t_positive["image"]
            negative = t_negative["image"]
        
        return anchor, positive, negative, (anchor_class, negative_class)
        # # We return class and subtract it by 1 since our dataset start with class 1
        # return anchor, positive, negative, torch.Tensor([int(anchor_class_name) - 1]).long()

class REIDTest(Dataset):
    def __init__(self, path, transform=None):
        self.imageFolderDataset = dset.ImageFolder(root=path)
        self.transform = transform
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    def __getitem__(self, index):
        # Get a random iamge which will be used as an anchor
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img1_tuple = random.choice(self.imageFolderDataset.imgs)
        # while True:
        #     # Get another image
        #     img1_tuple = random.choice(self.imageFolderDataset.imgs)
        #     if img0_tuple[0] != img1_tuple[0]:
        #         # Get an image that is not the same as current image even it is the same class and same id
        #         break
        
        img0_class_name = int(img0_tuple[0].split("/")[-2])
        img1_class_name = int(img1_tuple[0].split("/")[-2])
        
        if img0_class_name == img1_class_name:
            y = 1
        else:
            y = -1
        
        img0 = np.array(Image.open(img0_tuple[0]).convert("RGB"))
        img1 = np.array(Image.open(img1_tuple[0]).convert("RGB"))
        
        
        if self.transform is not None:
            t_img0 = self.transform(image=img0)
            t_img1 = self.transform(image=img1)
            img0 = t_img0["image"]
            img1 = t_img1["image"]
        
        return img0, img1, y
