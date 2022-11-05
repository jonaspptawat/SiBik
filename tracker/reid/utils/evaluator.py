import torch
from tqdm import tqdm

def compute_accuracy(cfg, val_loader, model, loss_fn):
    
    pbar = tqdm(val_loader)
    device = cfg.MODEL.DEVICE
    
    batch_loss = []
    
    for img0, img1, y in pbar:
        img0, img1, y = img0.to(device), img1.to(device), y.to(device)

        with torch.no_grad():
            p_img0 = model.get_features(img0)
            p_img1 = model.get_features(img1)
            consine_loss = loss_fn(p_img0, p_img1, y).float()
        
        batch_loss.append(float(consine_loss))
        
    total_batch_loss = sum(batch_loss) / len(batch_loss)
    
    return total_batch_loss
