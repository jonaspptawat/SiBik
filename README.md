# Real-time approaching vehicle detection with Yolo and StrongSORT using smartphone camera for cyclist
SiBik (Sight and Bike) built with PyTorch, OpenCV and ONNX using Yolo and StrongSORT with OSNet in order to detect and track vehicle. It aims to assist and prevent cyclists from approaching vehicles using their own smartphone camera in real-time. Originally, SiBik was trained on my custom datasets which consist only two classes. Also, SiBik's model (object detection and re-identification model) has around 300k trainable parameters and has 70% mAP for two classes. However, you can also train SiBik on your own custom datasets (see instructions below).

![](https://github.com/jonaspptawat/SiBik/blob/main/example1.gif)
