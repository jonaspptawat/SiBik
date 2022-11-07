# Real-time approaching vehicle detection with Yolo and StrongSORT using smartphone camera for cyclist
SiBik (Sight and Bike) built with PyTorch, OpenCV and ONNX using Yolo and StrongSORT with OSNet in order to detect and track vehicle. It aims to assist and prevent cyclists from approaching vehicles using their own smartphone camera in real-time. For an approaching vehicle warning conditions, SiBik uses Cosine similarity of two frames and Object's speed and distance.

![Alt Text](https://github.com/jonaspptawat/SiBik/blob/main/example1.gif)

Originally, SiBik was trained on my custom datasets which consist only two classes. Also, SiBik's model (object detection and re-identification model) has around 300k trainable parameters and has 70% mAP for two classes. However, you can also train SiBik on your own custom datasets (see instructions below).

## Model Explanation
SiBik consists of object detection model (YOLO) and object re-identification model (OSNet) which are trainable.

### Object detection model (YOLO)
YOLO (You Only Look Once) in SiBik has three main parts which are Backbone, Neck and Prediction. 

For Backbone, SiBik uses outputs from ShuffleNetV2's stage2 to stage4. Next, the Backbone's output will be passed through Feature Pyramid Network and Spatial Pyramid Pooling in Neck part and Prediction part respectively as shown in the following image.
![Alt Text](https://github.com/jonaspptawat/SiBik/blob/main/overview_detector.png)

### Object Re-Identification (OSNet)
SiBik uses OSNet to extract features from detected object in detection process in order to make StrongSORT stronger.
![Alt Text](https://github.com/jonaspptawat/SiBik/blob/main/REID_OSNet.png)
