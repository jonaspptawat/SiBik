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

## How to use

### 1. Install all dependencies in this project
```bash
pip install -r requirements.txt
```

### 2. Paste your mp4 video and Run tracker_vid
```bash
python3 tracker_vid.py
```

## Train on your custom datasets

### 1. Data Gathering and Labeling
The datasets that I used for training SiBik are collected by attaching camera under my bicycle and recording video while cycling in Chaing Mai, Thailand. To label data, i saved images from video that i gathered from riding a bike around Chiang Mai every N seconds and labelled each image using [Yolo_Label](https://github.com/developer0hye/Yolo_Label).

### 2. Store labelled datasets
2.1 Object detector
```bash
cd detector/
```

```bash
├── data
│   ├── __init__.py
│   ├── build.py
│   ├── collate_batch.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── build.py
│   │   └── cars
│   │       ├── README.md
│   │       ├── train
│   │       │   ├── 001.jpg
│   │       │   ├── 001.txt
│   │       │   ├── 002.jpg
│   │       │   ├── 002.txt
│   │       │   └── README.md
│   │       └── val
│   │           ├── 003.jpg
│   │           ├── 003.txt
│   │           ├── 004.jpg
│   │           ├── 004.txt
│   │           └── README.md
│   └── transforms
│       ├── __init__.py
│       └── build.py
```
2.2 Object Re-Identification

For object re-identification dataset preparation, I used the labelled dataset and cropped all detected images. Then, manually selected cropped images.
```bash
cd tracker/reid/
```
```bash
├── data
│   ├── __init__.py
│   ├── build.py
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── build.py
│   │   └── cars
│   │       ├── train
│   │       │   ├── 1
│   │       │   │   ├── 001_1.jpg
│   │       │   │   └── 001_2.jpg
│   │       │   ├── 2
│   │       │   │   ├── 002_1.jpg
│   │       │   │   └── 002_2.jpg
│   │       │   ├── 3
│   │       │   │   ├── 003_1.jpg
│   │       │   │   └── 003_2.jpg
│   │       │   └── README.md
│   │       └── val
│   │           ├── 1
│   │           │   ├── 001_3.jpg
│   │           │   └── 001_4.jpg
│   │           ├── 2
│   │           │   ├── 002_3.jpg
│   │           │   └── 002_4.jpg
│   │           └── README.md
│   └── transform
│       ├── __init__.py
│       └── build.py
```

### 3. Set hyperparameters in config file
You can set all hyperparameter in config file for every model in **defaults.py**.
```bash
config
├── __init__.py
└── defaults.py
```

### 4. Train
4.1 Train detector
```bash
cd detector/
python3 tools/train_model.py
```

4.2 Train Re-ID
```bash
cd tracker/reid/
python3 tools/train_model.py
```

After finishing training process, the model's weight will be in **checkpoint/**

```bash
checkpoint
├── README.md
└── checkpt.pth
```
