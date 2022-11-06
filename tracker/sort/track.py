# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
from .kalman_filter import KalmanFilter
from statistics import median
from scipy import spatial

R_DISTANCE = 80.0
MOTORCYCLE_WIDTH = 150.0
CAR_WIDTH = 300.0
REF_MOTOR_WIDTH_PX = 300.0
REF_CAR_WIDTH_PX = 500.0


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, detection, track_id, class_id, conf, n_init, max_age, ema_alpha,
                 feature=None):
        self.track_id = track_id
        self.class_id = int(class_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)
        
        # To get previous midpoint
        self.frame_count = 0
        self.get_sample_every = 3
        self.prev_point = []
        self.cosine_list = []
        self.n_prev_cosine = 5
        self.current_cosine = 0
        
        
        # Distance
        if class_id == 1:
            self.F = (REF_MOTOR_WIDTH_PX * R_DISTANCE) / MOTORCYCLE_WIDTH # Focal length
        else:
            self.F = (REF_CAR_WIDTH_PX * R_DISTANCE) / MOTORCYCLE_WIDTH
        
        self.distance = []
        self.speed_list = []
        self.n_speed = 5
        self.curr_speed = 0.1
    
    def get_warning_value(self):
        # This return all warning value for warning system
        return self.current_cosine, [self.curr_speed, self.distance[1]]
    
    def to_blbr(self):
        # (x, y, a, h)
        output = [0,0, 0,0, 0,0]
        ret = self.mean[:4].copy()
        ret[2] *= ret[3] # (x, y, w, h)
        output[:2] = ret[:2]
        output[2:4] = [ret[0] - ret[2]*0.5, ret[1] - ret[3]/2]
        output[4:6] = [ret[0] + ret[2]*0.5, ret[1] - ret[3]/2]
        return output
    

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret


    def ECC(self, src, dst, warp_mode = cv2.MOTION_EUCLIDEAN, eps = 1e-5,
        max_iter = 100, scale = 0.1, align = False):
        """Compute the warp matrix from src to dst.
        Parameters
        ----------
        src : ndarray 
            An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
        dst : ndarray
            An NxM matrix of target img(BGR or Gray).
        warp_mode: flags of opencv
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray
        """

        # skip if current and previous frame are not initialized (1st inference)
        if (src.any() or dst.any() is None):
            return None, None
        # skip if current and previous fames are not the same size
        elif (src.shape != dst.shape):
            return None, None

        # BGR2GRAY
        if src.ndim == 3:
            # Convert images to grayscale
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        # make the imgs smaller to speed up
        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                if scale != 1:
                    src_r = cv2.resize(src, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                    scale = [scale, scale]
                else:
                    src_r, dst_r = src, dst
                    scale = None
            else:
                if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                    src_r = cv2.resize(src, (scale[0], scale[1]), interpolation = cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
                else:
                    src_r, dst_r = src, dst
                    scale = None
        else:
            src_r, dst_r = src, dst

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, warp_matrix) = cv2.findTransformECC (src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
        except cv2.error as e:
            return None, None
        

        if scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        if align:
            sz = src.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
            else :
                # Use warpAffine for Translation, Euclidean and Affine
                src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
            return warp_matrix, src_aligned
        else:
            return warp_matrix, None


    def get_matrix(self, matrix):
        eye = np.eye(3)
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, previous_frame, next_frame):
        warp_matrix, src_aligned = self.ECC(previous_frame, next_frame)
        if warp_matrix is None and src_aligned is None:
            return
        [a,b] = warp_matrix
        warp_matrix=np.array([a,b,[0,0,1]])
        warp_matrix = warp_matrix.tolist()
        matrix = self.get_matrix(warp_matrix)

        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]


    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, class_id, conf, time_frame):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        
        #################
        ## Trajectory ###
        #################
        if len(self.prev_point) < 2:
            self.prev_point.append(self.to_blbr())
        else:
            u_xy = self.prev_point[1][:2]
            bl1 = self.prev_point[0][2:4]
            bl2 = self.prev_point[1][2:4]
            br1 = self.prev_point[0][4:6]
            br2 = self.prev_point[1][4:6]
            
            u = (np.array([0.5, 1]) - np.array(u_xy)).reshape(-1, 2) + 1e-5
            v_l = (np.array(bl2) - np.array(bl1)).reshape(-1, 2) + 1e-5
            v_r = (np.array(br2) - np.array(br1)).reshape(-1, 2) + 1e-5
            v1 = (v_l + v_r)
            v2 = (np.array(u_xy) - np.array(self.prev_point[0][:2])).reshape(-1, 2) + 1e-5
            current_cosine_for_two_point = max(1. - spatial.distance.cosine(u, v1), 1. - spatial.distance.cosine(u, v2))
            self.cosine_list.append(current_cosine_for_two_point)
            self.prev_point.pop(0)
            
            if len(self.cosine_list) > self.n_prev_cosine:
                self.cosine_list.pop(0)
                self.current_cosine = median(self.cosine_list)
            elif len(self.cosine_list) == self.n_prev_cosine:
                self.current_cosine = median(self.cosine_list)
        
        #################
        #### Speed #####
        #################
        if len(self.distance) < 2:
            if class_id == 1:
                self.distance.append((MOTORCYCLE_WIDTH * self.F) / (self.mean[2] * self.mean[3])) # a x h
            else:
                self.distance.append((CAR_WIDTH * self.F) / (self.mean[2] * self.mean[3]))
        else:
            speed = (self.distance[0] - self.distance[1]) / (time_frame + 1e-16)
            self.speed_list.append(speed)
            
            
            if len(self.speed_list) > self.n_speed:
                self.speed_list.pop(0)
                self.curr_speed = median(self.speed_list)
            elif len(self.speed_list) == self.n_speed:
                self.curr_speed = median(self.speed_list)
            
            self.distance.pop(0)
            if class_id == 1:
                self.distance.append((MOTORCYCLE_WIDTH * self.F) / (self.mean[2] * self.mean[3]))
            else:
                self.distance.append((CAR_WIDTH * self.F) / (self.mean[2] * self.mean[3]))
    
        
        self.conf = conf
        self.class_id = int(class_id)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)

        feature = detection.feature / np.linalg.norm(detection.feature)

        smooth_feat = self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
