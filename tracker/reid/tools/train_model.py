import sys
import torch
import torch.nn as nn
from torchsummary import summary

sys.path.append(".")
from config import cfg
from engine import Trainer
from modeling import build_reid
from data import build_dataloader
from utils import REIDLoss, compute_accuracy

if __name__ == "__main__":
    model = build_reid(cfg)
    train_loader = build_dataloader(cfg, is_train=True)
    val_loader = build_dataloader(cfg, is_train=False)
    
    loss_fn = REIDLoss(cfg)
    evaluator_loss = nn.CosineEmbeddingLoss(margin=0)
    
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
                      evaluator_loss,
                      evaluator=compute_accuracy,
                      load_checkpoint=False)
    trainer.train()
