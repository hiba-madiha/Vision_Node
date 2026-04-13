## 📚 Documentation

<details open>
<summary>Install</summary>

Clone the repository and install dependencies in a [**Python>=3.8.0**](https://www.python.org/) environment. Ensure you have [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) installed.

```bash
# Install the ultralytics package
pip install ultralytics
```

```bash
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5

# Navigate to the cloned directory
cd yolov5

# Install required packages
pip install -r requirements.txt
```

```bash
# activate the environment
source my_env/bin/activate

# Run the Program
python test_app6.py --weights yolov5n-int8.tflite --source (IP of camera) -- classes 0 1 2 3
```

</details>

<details open>
<summary>Inference with PyTorch Hub</summary>

Use YOLOv5 via [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/) for inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) are automatically downloaded from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
img = "https://ultralytics.com/images/zidane.jpg"  # Example image

# Perform inference (handles batching, resizing, normalization automatically)
results = model(img)

# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
results.print()  # Print results to console
results.show()  # Display results in a window
results.save()  # Save results to runs/detect/exp
```

</details>

<details>
<summary>Inference with detect.py</summary>

The `detect.py` script runs inference on various sources. It automatically downloads [models](https://github.com/ultralytics/yolov5/tree/master/models) from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saves the results to the `runs/detect` directory.

```bash
# Run inference using a webcam
python detect.py --weights yolov5s.pt --source 0

# Run inference on a local image file
python detect.py --weights yolov5s.pt --source img.jpg

# Run inference on a local video file
python detect.py --weights yolov5s.pt --source vid.mp4

# Run inference on a screen capture
python detect.py --weights yolov5s.pt --source screen

# Run inference on a directory of images
python detect.py --weights yolov5s.pt --source path/to/images/

# Run inference on a text file listing image paths
python detect.py --weights yolov5s.pt --source list.txt

# Run inference on a text file listing stream URLs
python detect.py --weights yolov5s.pt --source list.streams

# Run inference using a glob pattern for images
python detect.py --weights yolov5s.pt --source 'path/to/*.jpg'

# Run inference on a YouTube video URL
python detect.py --weights yolov5s.pt --source 'https://youtu.be/LNwODJXcvt4'

# Run inference on an RTSP, RTMP, or HTTP stream
python detect.py --weights yolov5s.pt --source 'rtsp://example.com/media.mp4'
```
