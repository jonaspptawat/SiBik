import torch

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    
    return torch.stack(img), torch.cat(label, 0)
