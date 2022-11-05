import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

def build_transforms(cfg, is_train=True):
    
    if is_train:
        
        transform = A.Compose([
            A.LongestMaxSize(max_size=int(cfg.INPUT.WIDTH)),
            A.PadIfNeeded(
                min_height=int(cfg.INPUT.HEIGHT),
                min_width=int(cfg.INPUT.WIDTH),
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.ColorJitter(brightness=cfg.INPUT.BRIGHT_JITTER,
                          contrast=cfg.INPUT.CONTRAST_JITTER,
                          saturation=cfg.INPUT.SAT_JITTER,
                          hue=cfg.INPUT.HUE_JITTER,
                          p=cfg.INPUT.PROB_JITTER),
            A.HorizontalFlip(p=cfg.INPUT.PROB_HOR),
            A.Blur(p=cfg.INPUT.PROB_BLUR),
            A.ToGray(p=cfg.INPUT.PROB_TOGRAY),
            A.CoarseDropout(p=cfg.INPUT.PROB_CDROP, max_holes=8, min_holes=1),
            A.Normalize(),
            ToTensorV2()
        ])
    
    else:
        
        transform = A.Compose([
            A.LongestMaxSize(max_size=int(cfg.INPUT.WIDTH)),
            A.PadIfNeeded(
                min_height=int(cfg.INPUT.HEIGHT),
                min_width=int(cfg.INPUT.WIDTH),
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(),
            ToTensorV2()
        ])
    
    return transform
