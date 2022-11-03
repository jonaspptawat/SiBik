#============================================================================#
# Source: https://github.com/dog-qiuqiu/FastestDet/blob/main/module/loss.py #
#============================================================================#
import math
import torch
import torch.nn as nn

class DetectorLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.cfg = cfg

        # Define Loss function for Object and Class Loss
        self.clsloss = nn.CrossEntropyLoss()
        self.objloss = nn.SmoothL1Loss(reduction="none")

    def bbox_iou(self, box1, box2):      
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        eps = 1e-7
        
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou
     
    def build_target(self, preds, targets):
        N, C, H, W = preds.shape
        
        # This store box, cls and index for each batch
        # Labeled data in specific batch
        gt_box, gt_cls, ps_index = [], [], []
        
        # Four vertices of each grid (Reference points where the box center point will return)
        # This is like you have 1 target at some specific cell then you assign the cell around its point to be candidate as well
        # For example, [0, 0] is the original target label position
        # [1, 0] will move object (x, y) to left 1 cell
        # [0, 1] will move object to the right 1 cell
        quadrant = torch.tensor([[0, 0], [1, 0], 
                                 [0, 1], [1, 1]], device=self.device) # Normally in FastestDet it has [1, 1] but in this case i will ignore it
        
        # Check if there is at least one target
        # Otherwise we will return gt_box, gt_cls, ps_index as empty list (For no labeled image)
        if targets.shape[0] > 0:
            # Map coordinates (labeled x,y,w,h) to feature map scales (e.g. grid 26x26)
            scale = torch.ones(6).to(self.device) # 6 is like the same shape of [obj, cls, x, y, w, h]
            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]] # Change index 2 to 6 to be size of grid (H and W)
            # Scale our labeled box by multiplying grid height and width to x,y,w,h but cls and obj will multiply with 1
            # scale > [1, 1, 26, 26, 26, 26]
            gt = targets * scale
            
            # Make four copy of one gt(scaled labeled)
            # 4 is from number of vertices (quadrant)
            # Example, From tensor([[1.0000, 1.0000, 0.2000, 0.3000, 0.8000, 0.7000],
                                    # [1.0000, 1.0000, 0.4000, 0.1000, 0.2000, 0.3000]]) # Of size (2, 6)
            # TO ----->>>> size (4, 2, 6)
            # tensor([[[1.0000, 1.0000, 0.2000, 0.3000, 0.8000, 0.7000],
            #  [2.0000, 1.0000, 4.0000, 3.0000, 0.8000, 0.7000]],

            # [[1.0000, 1.0000, 0.2000, 0.3000, 0.8000, 0.7000],
            #  [2.0000, 1.0000, 4.0000, 3.0000, 0.8000, 0.7000]],

            # [[1.0000, 1.0000, 0.2000, 0.3000, 0.8000, 0.7000],
            #  [2.0000, 1.0000, 4.0000, 3.0000, 0.8000, 0.7000]],

            # [[1.0000, 1.0000, 0.2000, 0.3000, 0.8000, 0.7000],
            #  [2.0000, 1.0000, 4.0000, 3.0000, 0.8000, 0.7000]]])
            # 
            gt = gt.repeat(4, 1, 1) # gt now size (4, number of label in that batch, 6)
            
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant # gij will contain four possible candidate target (x,y,w,h) ..# For one instance
            # Filter out of bounds(grid) coordinates
            # For example, to get rid of "gij" that out of our matrix
            # We will set all out of bounds coordinates to "0" and filter out!
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0
            
            # Filter using j(boolean)
            gi, gj = gij[j].T # grid position
            batch_index = gt[..., 0].long()[j] # Batch index (since dataset.py first index of target is batch_index)
            ps_index.append((batch_index, gi, gj)) # (Batch, x, y)
            
            gbox = gt[..., 2:][j] # Get (x, y, w, h) in grid's scale
            gt_box.append(gbox)
            
            gt_cls.append(gt[..., 1].long()[j])
        
        # Return grid scale
        return gt_box, gt_cls, ps_index
    
    def forward(self, preds, targets):
        # Initialize loss
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])
        
        
        # Get ground truth
        # This return as list of one element, to access we need to use index "0"
        gt_box, gt_cls, ps_index = self.build_target(preds, targets)
        
        # permute preds to match "ps_index" position > (Batch_num, H, W)
        pred = preds.permute(0, 2, 3, 1) # From (N, C, H, W) to => (N, H, W, C)
        
        # Obj pred
        pobj = pred[:, :, :, 0]
        # Box pred
        preg = pred[:, :, :, 1:5]
        # Cls pred
        pcls = pred[:, :, :, 5:]
        
        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj)
        factor = torch.ones_like(pobj) * 0.75
        
        # Check if there is gt box
        if len(gt_box) > 0:
            # Calculate Regression detection box loss
            b, gx, gy = ps_index[0] # Return batch_number, gx and gy
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            # We apply tanh to x and y and add gx and gy to make our prediction has the same scale (grid scale(HxW instead of 0-1))
            # We also apply sigmoid to width and height and multiply it by W and H to make grid scale for preds
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H
            
            # IoU loss for detection box
            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f] # Get only coordinates that has IoU greater overall mean of IoU
            
            # Calculate IoU loss
            iou = iou[f] # only select score that greater than overall mean (get rid of weak cross-grid candidates)
            iou_loss = (1.0 - iou).mean()
            
            ps = pcls[b, gy, gx]
            # ps = torch.log(pcls[b, gy, gx]) # This is LogSoftmax which is used for NLLLOSS
            # Detail here: https://jamesmccaffrey.wordpress.com/2020/06/11/pytorch-crossentropyloss-vs-nllloss-cross-entropy-loss-vs-negative-log-likelihood-loss/
            cls_loss = self.clsloss(ps, gt_cls[0][f])
            
            
            # Object loss (IoU aware concept from FastestDet github
            # Set target obj confidence score to IoU score !
            tobj[b, gy, gx] = iou.float() # Set some point from (b, gy, gx) to iou score
            # Count the number of positive samples for each image
            # This return number of batch that appear in "b"
            n = torch.bincount(b) # Count the frequency of each value in an array of non-negative ints.
            # if the image that has many ground truths box(many candidates) we will reduce the factor (low loss scale) of it
            # On the other hand, if the image that has less ground truths then object loss will be high !
            factor[b, gy, gx] = (1. / (n[b] / (H * W))) * 0.25 # Set factor for point (b, gy, gx) accroding to this formula but other will remain "0.75"
        
        obj_loss = (self.objloss(pobj, tobj) * factor).mean() # We use mean because we use reduction="none"
        
        # Combine all loss
        # This mean we pay more attention on Object Loss since it has 16 as a weight
        iou_loss *= 64
        obj_loss *= 64
        cls_loss *= 8
        loss = iou_loss + obj_loss + cls_loss
        
        return iou_loss, obj_loss, cls_loss, loss
