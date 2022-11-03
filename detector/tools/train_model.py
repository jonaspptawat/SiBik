import sys
import torch

sys.path.append(".")
from engine import Trainer
from modeling import build_detector
from config import cfg
from data import build_dataloader
from utils import DetectorLoss, compute_map

if __name__ == "__main__":
    model = build_detector(cfg.DATA.NUM_CLASSES)
    train_loader = build_dataloader(cfg, is_train=True, shuffle=True)
    val_loader = build_dataloader(cfg, is_train=False, shuffle=True) # Need Changeeee !!!
    evaluator = compute_map
    
    loss_fn = DetectorLoss(cfg.MODEL.DEVICE)
    optimizer_name = cfg.SOLVER.OPTIMIZER_NAME
    optim = getattr(torch.optim, optimizer_name)
    optimizer = optim(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                      momentum=cfg.SOLVER.MOMENTUM,
                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)
    trainer = Trainer(cfg,
                      model,
                      train_loader,
                      val_loader,
                      optimizer,
                      loss_fn,
                      scheduler,
                      evaluator,
                      load_checkpoint=False)

    trainer.train()
