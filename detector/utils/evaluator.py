# This code is based on these two github with some modification:
# https://github.com/dog-qiuqiu/FastestDet/blob/main/utils/evaluation.py
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py

import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from .utils import handle_preds, bbox_iou, xywh2xyxy


def get_batch_statistics(outputs, targets, iou_threshold, device):
    """ Compute true positives, predicted scores and predicted labels per sample """
    
    batch_metrics = []
    for sample_i in range(len(outputs)): # Loop through each batch sample (One image per batch)
        if outputs[sample_i].size(0) == 0:
            continue
        
        output = outputs[sample_i] # This select output for one sample in each batch which one sample may contain more than 1 box
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
        
        true_positives = np.zeros(pred_boxes.shape[0])
        
        # Select all target's box that label(t)'s batch number is the same as n(current batch) (Find the same image target)
        # And get only [1:] (class, x, y, w, h)
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        
        # If there is true box in image(sample_i)
        if len(annotations):
            detected_boxes = []
            
            # Get target class if there is really target box in sample_i image , otherwise we get empty list
            target_labels = annotations[:, 0] if len(annotations) else []
            # Get all target' box in specific sample_i image
            target_boxes = annotations[:, 1:] # (x, y, w, h) > (x, y, x, y)
            
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                pred_box = pred_box
                pred_label = pred_label
                
                # If all targets are found then break the loop go next sample_i
                if len(detected_boxes) == len(annotations):
                    break
                
                # Check if current pred_label appear in true target_labels list
                # If not then skip the current loop
                if pred_label not in target_labels:
                    continue
                
                # Calculate IoU and get Box index where pred_box gets max on IoU among all target box
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # If pred_box IoU >= mAP(iou_threshold) and not in detected_boxes list
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    # Batch metrics contains (N, 3) list
    return batch_metrics

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    
    # Sort by objectness
    i = np.argsort(-conf) # This sort from High conf score to Low conf
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] # Sorted tp, conf and pred_cls
    
    # Get all unique classes
    unique_classes = np.unique(target_cls)
    
    ap, p, r = [], [], []
    
    for c in unique_classes:
        # Get pred_cls index mask that are in the same class "c"
        i = pred_cls == c
        n_gt = (target_cls == c).sum() # Number of ground truth objects
        n_p = i.sum() # Number of predicted objects
        
        
        if n_p == 0 and n_gt == 0: # No pred and No ground truth in class "c"
            continue
        elif n_p == 0 or n_gt == 0: # Either pred or ground truth are no box in class "c"
            ap.append(0)
            r.append(0)
            p.append(0)
            continue
        else: # Both pred and ground truth contain box in class "c"
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1]) # Get only last index since it is cumsum
            
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1]) # Get only last index since it is cumsum
            
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    
    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(cfg, val_loader, model):
    #================================================================================#
    # Source: https://github.com/dog-qiuqiu/FastestDet/blob/main/utils/evaluation.py #
    #================================================================================#
    
    labels = []
    sample_metrics = [] # List for get_batch_statistic's output
    pbar = tqdm(val_loader)
    for i, (imgs, targets) in enumerate(pbar):
        
        imgs = imgs.to(cfg.MODEL.DEVICE)
        tarets = targets.to(cfg.MODEL.DEVICE)
        
        # Extract labels
        labels += targets[:, 1].tolist()
        # Change targets box format
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        
        with torch.no_grad():
            preds = model(imgs)
            output = handle_preds(preds, 
                                cfg.MODEL.DEVICE,
                                cfg.TEST.CONF_THRESH,
                                cfg.TEST.NMS_THRESH)
            
        sample_metrics += get_batch_statistics(output, targets, cfg.TEST.MAP_THRESH, cfg.MODEL.DEVICE)
        pbar.set_description("Evaluation:")
    
    if len(sample_metrics) == 0:  # No detections over whole validation set.
        return None
    
    # Concatenate sample metrics
    tp, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(tp, pred_scores, pred_labels, labels)

    return metrics_output # np.mean(p), np.mean(r), np.mean(ap), np.mean(f1)
