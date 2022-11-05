from .osnet import *

def build_reid(cfg):
    model = osnet_x0_25(cfg)
    return model
