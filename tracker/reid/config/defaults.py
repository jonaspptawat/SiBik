from yacs.config import CfgNode as CN
import torch

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
# Device for model
_C.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Backbone weight path
_C.MODEL.BACKBONE = "" 
# REID weight
_C.MODEL.REID = ""

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Width and Height
_C.DATA.WIDTH = 128
_C.DATA.HEIGHT = 128
# Dataset Path
_C.DATA.TRAIN = "./data/datasets/cars/train"
_C.DATA.VAL = "./data/datasets/cars/val"
# Batch size
_C.DATA.SIZE_TRAIN = 16
_C.DATA.SIZE_VAL = 1
