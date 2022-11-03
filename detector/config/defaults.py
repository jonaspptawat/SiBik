from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Model 
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Device for model
_C.MODEL.DEVICE = "cpu"
# Backbone weight path
_C.MODEL.BACKBONE = "" 
# Detector weight path
_C.MODEL.DETWEIGHT = ""

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Number of classes
_C.DATA.NUM_CLASSES = 2
# Classes
_C.DATA.CLASSES = ["car", "motorcycle"]
# Width and Height
_C.DATA.WIDTH = 416
_C.DATA.HEIGHT = 416
# Dataset Path
_C.DATA.TRAIN = "./data/datasets/cars/train"
_C.DATA.VAL = "./data/datasets/cars/val"
# Batch size
_C.DATA.SIZE_TRAIN = 1
_C.DATA.SIZE_VAL = 1

# -----------------------------------------------------------------------------
# Transformation
# -----------------------------------------------------------------------------
_C.T = CN()
# Horizontal Flip Prob
_C.T.PROB_HFLIP = 0.4
# RandomBrightness and Contrast config
_C.T.BRIGHT_JITTER = 0.2
_C.T.CONTRAST_JITTER = 0.2
_C.T.PROB_BRIGHTCONTRAST = 0.1
# Bounding box min visibility
_C.T.BBOX_MINVIS = 0.55

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Confident, NMS, MAP Threshold
_C.TEST.CONF_THRESH = 0.85
_C.TEST.NMS_THRESH = 0.005
_C.TEST.MAP_THRESH = 0.85

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.EPOCHS = 1000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (150, 300, 450, 600)

_C.SOLVER.WARMUP_EXPO = 1.0 / 3
_C.SOLVER.WARMUP_ITERS_TIMES = 5

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 1

# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CHECKPOINT = CN()

_C.CHECKPOINT.PATH = "./checkpoint/check.pth"
