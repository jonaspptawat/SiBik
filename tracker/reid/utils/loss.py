import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    From :https://github.com/heejae1213/object_tracking_deepsort/blob/master/feature%20extractor/siamese_net.py
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor,positive) #Each is batch X 512 
        distance_negative = F.cosine_similarity(anchor,negative)  # .pow(.5)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case. 
        return losses.mean() if size_average else losses.sum()

class REIDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Class Loss
        self.clsloss = nn.CrossEntropyLoss()
        
        # TripletMarginLoss
        self.triplet = TripletLoss(cfg.SOLVER.TRIPLET_MARGIN)
        
    def forward(self, preds, targets):
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, triplet_loss = ft([0]), ft([0])
        
        anchor_v, positive_v, negative_v = preds[:3]
        anchor_class, negative_class = preds[3:]
        
        t_anchor_class, t_negative_class = targets
        t_anchor_class = t_anchor_class.reshape(-1)
        t_negative_class = t_negative_class.reshape(-1)
        
        # Calculate TripletLoss
        triplet_loss += self.triplet(anchor_v, positive_v, negative_v)
        
        # Calculate Class Loss
        clsloss_anchor = self.clsloss(anchor_class, t_anchor_class)
        clsloss_negative = self.clsloss(negative_class, t_negative_class)
        cls_loss += clsloss_anchor + clsloss_negative
        
        cls_loss *= 16
        triplet_loss *= 64
        total_loss = (cls_loss) + (triplet_loss)
        
        return cls_loss, triplet_loss, total_loss
