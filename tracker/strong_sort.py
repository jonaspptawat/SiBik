# This code is based on this github repo:
# https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/blob/master/strong_sort/strong_sort.py

import numpy as np

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

from .config import cfg

__all__ = ["StrongSORT"]

class StrongSORT(object):
    def __init__(self, reid_model):
        
        self.model = reid_model
        
        self.max_dist = cfg.SCONFIG.MAX_DIST
        self.max_iou_distance = cfg.SCONFIG.MAX_IOU
        self.max_age = cfg.SCONFIG.MAX_AGE
        self.n_init = cfg.SCONFIG.N_INIT
        self.nn_budget = cfg.SCONFIG.BUDGET
        self.mc_lambda = cfg.SCONFIG.MC_LAMBDA
        self.ema_alpha = cfg.SCONFIG.EMA_LAMBDA
        
        
        metric = NearestNeighborDistanceMetric(
            self.max_dist, self.nn_budget
        )
        
        self.tracker = Tracker(
            metric,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
            ema_alpha=self.ema_alpha,
            mc_lambda=self.mc_lambda
        )
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2
    
    def _xyxy_to_tlwh_1(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h
    
    def _xyxy_to_tlwh(self, bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        
        bbox_tlwh[:, 2:4] = bbox_xyxy[:, 2:4] - bbox_xyxy[:, :2]
        return bbox_tlwh
    
    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2
    
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh
    
    def _get_features(self, bbox_xywh, ori_img):
        """Get features from cropped image

        Args:
            bbox_xywh (array): array that contain bounding boxes
            ori_img (array): current image

        Returns:
            _type_: _description_
        """
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            # Need to fix self.model to be able to get a list of imgs
            features = self.model(im_crops)
        else:
            features = np.array([])
        
        return features
    
    def update(self, bbox_xyxy, confidences, classes, ori_img, time_frame):
        self.height, self.width = ori_img.shape[:2]
        
        # Generate detections
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i])
                      for i, conf in enumerate(confidences)]
        
        # # Run non max supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        
        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences, time_frame)
        
        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            box = track.to_tlwh() # (x, y, w, h)
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            
            cosine_trajectory, distance = track.get_warning_value()
            
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf, cosine_trajectory, distance]))
        
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        
        return outputs

    def increment_ages(self):
        self.tracker.increment_ages()
