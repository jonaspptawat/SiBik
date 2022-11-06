import cv2
import numpy as np

__all__ = ["inference"]

def preprocess(src_img, size=(128, 128), mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    output = (output / 255 - mean) / std
    output = output.transpose([2, 0, 1]).reshape((1, 3, size[1], size[0]))
    return output.astype('float32')

def inference(session, image):
    data = preprocess(image)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: data})[0][0]
    return output
