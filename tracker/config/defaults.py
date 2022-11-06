from yacs.config import CfgNode as CN


_C = CN()

#########################
### Strong sort config ##
#########################

_C.SCONFIG = CN()
_C.SCONFIG.MAX_DIST = 0.3
_C.SCONFIG.MAX_IOU = 0.2
_C.SCONFIG.MAX_AGE = 70
_C.SCONFIG.N_INIT = 5
_C.SCONFIG.BUDGET = 100
_C.SCONFIG.MC_LAMBDA = 0.995
_C.SCONFIG.EMA_LAMBDA = 0.9
