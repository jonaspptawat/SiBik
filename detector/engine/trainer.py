import math
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 cfg,
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_fn,
                 scheduler,
                 evaluator,
                 load_checkpoint=False):
        
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.EPOCHS = self.cfg.SOLVER.EPOCHS
        self.device = self.cfg.MODEL.DEVICE
        
        if load_checkpoint:
            checkpoint_path = self.cfg.CHECKPOINT.PATH
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # Pushing optimizer to CUDA automatically if self.device is cuda
            if self.device == "cuda":
              self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
              for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            
            self.start_epoch = checkpoint["epoch"]
            
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        else:
            self.start_epoch = 0
        
        # LR warmup
        # We will use LR warmup for 3 * (number of train batch) 
        # This mean we will use LR warmup 3 times for whole dataset befpre using base LR
        self.warmup_num = int(self.cfg.SOLVER.WARMUP_ITERS_TIMES * len(train_loader))
    
    def train(self):
        print(f"Total images:{len(self.train_loader.dataset)}")
        print(f"Starting training from epoch {self.start_epoch+1}/{self.EPOCHS}")
        
        self.batch_num = self.start_epoch * self.cfg.INPUT.SIZE_TRAIN
        if self.batch_num <= self.warmup_num:
            print("Learning Rate WARMUP is used")

        self.model.to(self.device)
        for epochi in range(self.start_epoch, self.EPOCHS+1):
            batch_loss = []
            self.model.train()
            pbar = tqdm(self.train_loader)
            for imgs, labels in pbar:
                # Push img and label to self.device
                imgs, labels = imgs.to(self.device) / 255.0, labels.to(self.device)
                # Predict
                preds = self.model(imgs)
                # Compute Loss
                iou_loss, obj_loss, cls_loss, total_loss = self.loss_fn(preds, labels)
                
                # Calculate Grad
                total_loss.backward()
                # Update params and set Grad to zero after updated
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # LR warm up
                if self.batch_num <= self.warmup_num:
                    for g in self.optimizer.param_groups:
                        scale = math.pow(self.batch_num/self.warmup_num, self.cfg.SOLVER.WARMUP_EXPO)
                        g["lr"] = self.cfg.SOLVER.BASE_LR * scale
                    lr = g["lr"]
                    info = f"(WARMUP)Epoch:{epochi+1} LR:{lr:.9f} IOU_L:{iou_loss:.3f} OBJ_L:{obj_loss:.3f} CLS_L:{cls_loss:.3f} TOTAL_L:{total_loss:.3f}"
                else:
                    info = f"Epoch:{epochi+1} IOU_L:{iou_loss:.3f} OBJ_L:{obj_loss:.3f} CLS_L:{cls_loss:.3f} TOTAL_L:{total_loss:.3f}"
                    
                # Set tqdm info
                pbar.set_description(info)
                # Add total_loss for every batch
                batch_loss.append(total_loss)
                self.batch_num += 1
            
            print(f"(Average) Total Loss for epoch{epochi+1}: {sum(batch_loss) / len(batch_loss):.2f}")
            # Evaluate on Validation set for every 5 epochs
            if (epochi+1) % self.cfg.SOLVER.LOG_PERIOD == 0:
                self.model.eval()
                print("Computing Mean Average Precision of Validation set.....")
                metrics = self.evaluator(self.cfg, self.val_loader, self.model)
                if metrics is not None:
                    p, r, mAP, f1 = metrics
                    print(f"mAP:{mAP*100:.2f}% Precision:{p*100:.2f}% Recall:{r*100:.2f}% F1:{f1*100:.2f}%")
                else:
                    print("---- No detections over whole validation set ----")
                
                if (epochi+1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                    torch.save({
                        "epoch": epochi,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict()
                    }, self.cfg.CHECKPOINT.PATH)
            
            if self.scheduler is not None:
                self.scheduler.step()
