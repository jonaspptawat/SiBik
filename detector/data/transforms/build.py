import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

def build_transforms(cfg, is_train=True):

    if is_train:

        transform = A.Compose([
            A.LongestMaxSize(max_size=int(cfg.DATA.WIDTH)),
            A.PadIfNeeded(
                min_height=int(cfg.DATA.HEIGHT),
                min_width=int(cfg.DATA.WIDTH),
                border_mode=cv2.BORDER_CONSTANT
                ),
            A.HorizontalFlip(p=cfg.T.PROB_HFLIP),
            A.RandomBrightnessContrast(brightness_limit=cfg.T.BRIGHT_JITTER,
                                       contrast_limit=cfg.T.CONTRAST_JITTER,
                                       p=cfg.T.PROB_BRIGHTCONTRAST),
            A.Normalize(),
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=cfg.T.BBOX_MINVIS, label_fields=[]))
    else:
        
        transform = A.Compose([
            A.LongestMaxSize(max_size=int(cfg.DATA.WIDTH)),
            A.PadIfNeeded(
                min_height=int(cfg.DATA.HEIGHT),
                min_width=int(cfg.DATA.WIDTH),
                border_mode=cv2.BORDER_CONSTANT
                ),
            A.Normalize(),
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=cfg.T.BBOX_MINVIS, label_fields=[]))

    return transform
