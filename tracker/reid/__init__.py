from .modeling import get_reid
from .data import build_transforms
from .config import cfg
import torch

__all__ = ["feature_extractor", "transformer"]

def feature_extractor(cfg):
    model = get_reid()
    weight = torch.load(cfg.REID.WEIGHT_PATH, map_location=torch.device(cfg.REID.DEVICE))
    model.load_state_dict(weight)
    model.eval()
    return model

def transformer():
    transform = build_transforms(cfg, is_train=False)
    return transform
