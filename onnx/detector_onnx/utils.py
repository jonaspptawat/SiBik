import cv2
import numpy as np

__all__ = ["detection"]


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

def softmax(x):
    """compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def nms(dets, thresh=0.45):
    if len(dets) < 1:
        return []
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return np.array(output)

def preprocess(src_img, size=(416, 416), mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    output = (output / 255 - mean) / std
    output = output.transpose([2, 0, 1]).reshape((1, 3, size[1], size[0]))
    return output.astype('float32')

def detection(session, image, size, thresh):
    H, W = size
    pred = []
    data = preprocess(image)

    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]
    
    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]
    
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            
            data = feature_map[h][w]
            cls_score_softm = softmax(data[5:])
            obj_score, cls_score = data[0], cls_score_softm.max()
            
            score = (obj_score ** 0.6) * (cls_score ** 0.4) # Get confidence score base on class and obj score itself
            
            if score > thresh:
                cls_index = np.argmax(cls_score_softm)
                x_offset, y_offset = tanh(data[1]), tanh(data[2]) # Scale [-1, 1] so we can discard all value that exceed 1 and below 0
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4]) # Scale [0, 1]
                # Add both offset with [h, w] which is grid scale
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                
                if (x1 < 0 or x1 > 1) or (y1 < 0 or y1 > 1) or (x2 < 0 or x2 > 1) or (y2 < 0 or y2 > 1):
                    continue
                else:
                    x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                    pred.append([x1, y1, x2, y2, score, cls_index])
    
    # return weighted_nms(np.array(pred), thresh_in=0.7, thresh_out=0.1)
    return nms(np.array(pred), thresh=0.05)
