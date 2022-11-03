from .modeling import build_detector
from .config import cfg as det_cfg
from .utils import handle_preds, xyxy2xywh

import torch
import torch.nn as nn
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

__all__ = ["MultiDetector", "MultiDetector_onnx", "xyxy2xywh", "handle_preds"]

def build_transforms():
    transform = A.Compose([
        A.LongestMaxSize(max_size=int(det_cfg.DATA.WIDTH)),
        A.PadIfNeeded(
            min_height=int(det_cfg.DATA.HEIGHT),
            min_width=int(det_cfg.DATA.WIDTH),
            border_mode=cv2.BORDER_CONSTANT
            ),
        A.Normalize(),
        ToTensorV2()
        ])

    return transform

def get_detector():
    model = build_detector(det_cfg.DATA.NUM_CLASSES)
    if len(det_cfg.MODEL.DETWEIGHT) > 0:
        weight_state_dict = torch.load(det_cfg.MODEL.DETWEIGHT, map_location="cpu")["model_state_dict"]
        model.load_state_dict(weight_state_dict)
    
    model.eval()
    return model

class MultiDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = get_detector()
        self.transform = build_transforms()

    def preprocess(self, x):
        x = self.transform(image=x)["image"].unsqueeze(0)
        return x
    
    def forward(self, x):
        x = self.preprocess(x)
        with torch.no_grad():
            x = self.model(x)
        return x

class MultiDetector_onnx(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = get_detector()

    def forward(self, x):
        x = self.model(x)
        return x
