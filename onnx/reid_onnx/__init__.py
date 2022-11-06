from .utils import inference
import onnxruntime
import numpy as np

__all__ = ["MultiREID"]

class MultiREID:
    def __init__(self):
        self.session = onnxruntime.InferenceSession("./tracker/reid/quantized_reid.onnx")
        # self.session = onnxruntime.InferenceSession("./reid/quantized_reid.onnx")
    
    def __call__(self, x_list):
        outputs = []
        
        for i, x in enumerate(x_list):
            pred_x = inference(self.session, x)
            outputs.append(pred_x)
        
        return np.array(outputs).reshape(len(x_list), -1)
