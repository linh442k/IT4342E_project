# Face Mask Detector

Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning

## Requirement

- Python 3.9.5

- pip 21.3.1

## Install

#### Necessary Packages

```
 pip install -r requirements.txt
```

#### Jupyter Notebook(optional)

```
 pip install notebook
```

<hr>

## Model Training and Evaluating

Extract `dataset/dataset.rar` to `dataset` folder and then delete the archived file.

### With Python only

Edit the following variable in `Face Mask Detector Train.py`

```python
# path to the dataset, saved model and the plot of the training loss and accuracy
datasetPath = r'D:\ComputerVision\dataset'
savedModelPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
savedPlotPath = r'D:\ComputerVision\plot_facemask.png'
```

Then run in the terminal

```
python '.\Face Mask Detector Train.py'
```

### With Jupyter Notebook

Edit the following variable in `Face Mask Detector Train.ipynb`

```python
# path to the dataset, saved model and the plot of the training loss and accuracy
datasetPath = r'D:\ComputerVision\dataset'
savedModelPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
savedPlotPath = r'D:\ComputerVision\plot_facemask.png'
```

Then click `Run All`

<hr>

## Implementing our face mask detector for images

### With Python only

Edit the following variable in `Face Mask Detector Image.py`

```python
# path to the input image, the face detector model directory, the face mask detector model that we trained
# faceDetectionThreshold: minimum probability to filter weak face detections

inputImagePath = r'D:\ComputerVision\test_images\small_people_without_mask.jpg'
faceDetectorModelPath = r'D:\ComputerVision\face_detector'
facemaskDetectorPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
faceDetectionThreshold = 0.5
```

Then run in the terminal

```
python '.\Face Mask Detector Image.py'
```

### With Jupyter Notebook

Edit the following variable in `Face Mask Detector Image.ipynb`

```python
# path to the input image, the face detector model directory, the face mask detector model that we trained
# faceDetectionThreshold: minimum probability to filter weak face detections

inputImagePath = r'D:\ComputerVision\test_images\small_people_without_mask.jpg'
faceDetectorModelPath = r'D:\ComputerVision\face_detector'
facemaskDetectorPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
faceDetectionThreshold = 0.5
```

Then click `Run All`

<hr>

## Implementing our face mask detector in real-time video streams

### With Python only

Edit the following variable in `Face Mask Detector Video.py`

```python
# path to the face detector model directory, the face mask detector model that we trained
# faceDetectionThreshold: minimum probability to filter weak face detections

faceDetectorModelPath = r'D:\ComputerVision\face_detector'
facemaskDetectorPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
faceDetectionThreshold = 0.5
```

Then run in the terminal

```
python '.\Face Mask Detector Video.py'
```

### With Jupyter Notebook

Edit the following variable in `Face Mask Detector Video.ipynb`

```python
# path to the face detector model directory, the face mask detector model that we trained
# faceDetectionThreshold: minimum probability to filter weak face detections

faceDetectorModelPath = r'D:\ComputerVision\face_detector'
facemaskDetectorPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
faceDetectionThreshold = 0.5
```

Then click `Run All`
