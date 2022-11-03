import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    def __init__(self,
                 cfg,
                 path="./cars/train",
                 transform=None):
        
        assert os.path.exists(path), f"{path} is not exist!"

        self.path = path
        self.cfg = cfg
        self.transform = transform

        cwd = os.getcwd()
        image_formats = ["bmp", "jpg", "jpeg", "png"]
        
        # Get all image path from path
        self.data_path = [os.path.join(cwd, os.path.join(path, dname))
                          for dname in os.listdir(path)
                          if dname.split(".")[-1] in image_formats]

        assert os.path.exists(self.data_path[0])

        # Load label_path
        self.label_path = [d_path[:-3] + "txt"
                           for d_path in self.data_path]

        assert os.path.exists(self.label_path[0])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        image_path = self.data_path[index]
        label_path = self.label_path[index]

        # Load image
        img = np.array(Image.open(image_path).convert("RGB"))

        # Load label
        label = np.loadtxt(label_path).reshape(-1, 5)

        if self.transform:
            # Move cls to index - 1 to make sure that it fits albumentation format
            label = np.roll(label, 4, 1)
            transformed = self.transform(image=img, bboxes=label)
            
            img = transformed["image"]
            label = np.array(transformed["bboxes"])
            
            # Roll the label back to normal format
            if label.ndim > 1:
                label = np.roll(label, 1, 1)
            else:
                label = np.roll(label ,1)

        else:
            img = cv2.resize(img, (self.cfg.DATA.WIDTH, self.cfg.DATA.HEIGHT), interpolation=cv2.INTER_LINEAR)
            img = np.array(img, dtype=np.float32)

        label = np.array([[0] + l.tolist() for l in label], dtype=np.float32)

        return img, torch.from_numpy(label)
