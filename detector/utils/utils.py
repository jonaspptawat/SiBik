import torch
import torchvision
import numpy as np

def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    # This code is from: https://github.com/dog-qiuqiu/FastestDet/blob/main/utils/tool.py

    output_bboxes = []

    # Convert the feature map to the coordinates of the detection box
    N, C, H, W = preds.shape
    # This contains x,y,w,h,conf,class
    bboxes = torch.zeros((N, H, W, 6))
    # From (Batch, Chanels, H, W) to (Batch, H, W, channels)
    pred = preds.permute(0, 2, 3, 1)

    # Object score
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1) # Unsqueeze to get the same size as others
    # Box (x, y, w, h)
    preg = pred[:, :, :, 1:5]
    # Class size(torch.Size([16, 26, 26, 2]))
    pcls = pred[:, :, :, 5:].softmax(dim=-1)

    # Assign object confidence and class to index 4 and 5 respectively
    # bboxes[..., 4] = pobj.squeeze(-1) # Testing purpose
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    # Get Coordinates of detection's box
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
    # make width and height between 0-1
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
    # We apply tanh then divided by W and H
    # tanh make value between -1 to 1
    # We need to apply tanh and +gx and +gy like we do in Loss function
    # But also need to divide by W and H to normalize it to 0-1 scale of whole image instead of grid scale(HxW)
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    # cx, cy, w, h to x1,y1,x2,y2 (by subtracting half of width and height)
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    # Flatten from H x W to (H*W) x 6 
    bboxes = bboxes.reshape(N, H*W, 6)

    # Non-max suppression on all box in bboxes
    # Loop through each batch
    for p in bboxes:
        output, temp = [], []
        # box, score, class
        b, s, c = [], [], []

        # Check if confidence score > "conf_thresh"
        # Get mask(t)
        t = p[:, 4] > conf_thresh
        # Apply mask to p(each box)
        pb = p[t]
        # Loop through each box in each batch
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            # store all current box details in temp(list)
            temp.append([x1, y1, x2, y2, obj_score, category])

        # Using Torchvision NMS
        if len(b) > 0:
            # Convert to torch.Tensor to make it able to use in batched_nms()
            b = torch.Tensor(b).to(device) # box (format: Tensor[N, 4])
            c = torch.Tensor(c).squeeze(1).to(device) # class (format: Tensor[N])
            s = torch.Tensor(s).squeeze(1).to(device) # scores (format: Tensor[N])
            # Get index from batched_nms with checking confidence score "nms_thresh"
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

