from .utils import detection
import onnxruntime

__all__ = ["Detector"]

class Detector:
    def __init__(self):
        self.session = onnxruntime.InferenceSession('./detector/quantized_det.onnx')
    
    def __call__(self, x):
        H, W = x.shape[:2]
        bboxes = detection(self.session, x, (H, W), 0.85)
        return bboxes
