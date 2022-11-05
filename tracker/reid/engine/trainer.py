import math
import sys
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
                 evaluator_loss,
                 evaluator=None,
                 load_checkpoint=False):
        
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.evaluator_loss = evaluator_loss
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
        
        self.model.to(self.device)
        
        # LR warmup
        # We will use LR warmup for 3 * (number of train batch) 
        # This mean we will use LR warmup 3 times for whole dataset befpre using base LR
        self.warmup_num = int(self.cfg.SOLVER.WARMUP_ITERS_TIMES * len(train_loader))
    
    def train(self):
        print(f"Starting training from epoch {self.start_epoch+1}/{self.EPOCHS}")
        
        self.batch_num = self.start_epoch * self.cfg.INPUT.SIZE_TRAIN
        best_val_loss = 0.10
        for epochi in range(self.start_epoch, self.EPOCHS+1):
            batch_loss = []
            batch_closs = []
            batch_tloss = []
            self.model.train()
            pbar = tqdm(self.train_loader)
            i_batch = 1
            for anchor, positive, negative, (anchor_class, negative_class) in pbar:
                # Push img and label to self.device
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                targets = anchor_class.to(self.device), negative_class.to(self.device)
                # Predict
                preds = self.model(anchor, positive, negative)
                # Compute Loss
                cls_loss, triplet_loss, total_loss = self.loss_fn(preds, targets)
                
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
                    info = f"(WARMUP)Epoch:{epochi+1} LR:{lr:.6f} Cls_L:{float(cls_loss):.3f} TRIP_L:{float(triplet_loss):.3f} TOTAL_L:{float(total_loss):.3f}"
                else:
                    info = f"Epoch:{epochi+1} Cls_L:{float(cls_loss):.3f} TRIP_L:{float(triplet_loss):.3f} TOTAL_L:{float(total_loss):.3f}"
                    
                # Set tqdm info
                pbar.set_description(info)
                # Add total_loss for every batch
                batch_loss.append(float(total_loss))
                batch_closs.append(float(cls_loss))
                batch_tloss.append(float(triplet_loss))
                
                self.batch_num += 1
            
            self.model.eval()
            total_batch_loss = sum(batch_loss) / len(batch_loss)
            class_batch_loss = sum(batch_closs) / len(batch_closs)
            trip_batch_loss = sum(batch_tloss) / len(batch_tloss)
            print(f"(Average) Epoch{epochi+1} Cls_L:{class_batch_loss:.2f} TRIP_L:{trip_batch_loss:.2f} TOTAL_L: {total_batch_loss:.2f}")
            
            if (epochi+1) % self.cfg.SOLVER.LOG_PERIOD == 0:
                if self.evaluator is not None:
                    print("Testing on validation set ...........")
                    cosine_loss = self.evaluator(self.cfg, self.val_loader, self.model, self.evaluator_loss)
                    print(f"(Testing) Consine Loss:{cosine_loss:.3f}")
                    print("="*30)

            if (cosine_loss < best_val_loss) and ((epochi+1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0):
              torch.save({
                    "epoch": epochi,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()
                }, self.cfg.CHECKPOINT.PATH)
              print(f"Checkpoint has been save at this epoch with loss:{cosine_loss}")
              best_val_acc = cosine_loss

            
        if self.scheduler is not None:
                self.scheduler.step()
