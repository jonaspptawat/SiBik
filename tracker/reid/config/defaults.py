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
# Class config
_C.DATA.CLASS_SEP = 25

# -----------------------------------------------------------------------------
# INPUT TRANSFORMATION
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Scale of Image resizing
_C.INPUT.RESIZE_SCALE = 1.1
# Brightness jittering
_C.INPUT.BRIGHT_JITTER = 0.25
# Contrast jittering
_C.INPUT.CONTRAST_JITTER = 0.25
# Saturation jittering
_C.INPUT.SAT_JITTER = 0.25
# Hue jittering
_C.INPUT.HUE_JITTER = 0.25
# Jitter Probability
_C.INPUT.PROB_JITTER = 0.5
# Horizontal Flip Probability
_C.INPUT.PROB_HOR = 0.5
# Blur Probability
_C.INPUT.PROB_BLUR = 0.45
# To gray Probability
_C.INPUT.PROB_TOGRAY = 0.05
# Coarse Dropout Probability
_C.INPUT.PROB_CDROP = 0.55

# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CHECKPOINT = CN()
# Checkpoint file path
_C.CHECKPOINT.PATH = "./checkpoint/check.pth"
# Checkpoint file path for feature extractor
_C.CHECKPOINT.PATH_FE = "./checkpoint/reid_weight.pth"
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.EPOCHS = 1000

_C.SOLVER.BASE_LR = 0.005

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (100, 150, 200, 250, 300)

_C.SOLVER.WARMUP_EXPO = 1.0 / 3
_C.SOLVER.WARMUP_ITERS_TIMES = 1

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 1

# Triplet loss margin value
_C.SOLVER.TRIPLET_MARGIN = 1
