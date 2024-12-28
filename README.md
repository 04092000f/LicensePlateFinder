# LicensePlateFinder

LicensePlateFinder implements **YOLOv5** to build a high-precision **License Plate Detection Model**. The model was fine-tuned on a custom dataset and achieved impressive performance:

- **mAP\@0.5:0.95**: `0.762`
- **mAP\@0.5**: `0.831` (VOC metric)

---

## Table of Contents

1. [About the Dataset](#about-the-dataset)
2. [Model and Preprocessing](#model-and-preprocessing)
3. [Training Configuration](#training-configuration)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Inference Results](#inference-results)
6. [Real-Time Inference with OCR](#real-time-inference-with-ocr)

---

## About the Dataset

The dataset comprises **5,694 images** of license plates with bounding box annotations provided in `[xmin, ymin, xmax, ymax]` format. It is organized as follows:

```
Dataset
├── train
│  └── Vehicle registration plate
│      └── Label
└── validation
    └── Vehicle registration plate
        └── Label
```

- **Training Samples**: 5,308
- **Validation Samples**: 386

### Sample Images

Below are some images with `ground truth` bounding boxes:



---

## Model and Preprocessing

The **YOLOv5** model was chosen for its balance between speed and accuracy. Key steps in preprocessing included:

1. **Annotation Conversion**: Converted `[xmin, ymin, xmax, ymax]` **Pascal VOC** bounding box annotations into `[x, y, width, height]` **YOLO** format.
2. **Data Configuration File**: Prepared a `data.yaml` file with the following details:

```yaml
train: /content/License_Plate_Detection/train/images
val: /content/License_Plate_Detection/validation/images
nc: 1
names: ['Reg-plate']
```

---

## Training Configuration

The model was fine-tuned with the following hyperparameters:

- **Batch Size**: `32`
- **Epochs**: `25`
- **Iterations**: `4,146`
  - calculated as `int(epochs * train_img_count / batch_size)`
- **Initial Learning Rate**: `1e-3`

---

## Evaluation Metrics

The final COCO primary metric `(mAP@0.5:0.95)` was **0.762**, while the `mAP@0.5` (VOC metric) was **0.831**. The overall metric statistics is:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.762
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.831
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.830
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.862
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.895
```

The WandB logs can be found [here](https://wandb.ai/furqansa344-na/opencv_od_project/reports/License-Plate-Detection--Vmlldzo5MjA5NDcx?accessToken=axc7exli81c4oe8ykmppbw6hpz3k95bzn7w9ir8g7tepvi1vvghhokhdoo9d53le).

---

## Inference Results



---

## Real-Time Inference with OCR

To further enhance the functionality of the license plate detection system, **PaddleOCR** was integrated to extract and display license plate text in real time. This allows for seamless detection and recognition of license plates.

### Workflow

1. **Detection**: YOLOv5 detects license plates in images or video frames.
2. **Cropping**: The detected bounding box regions are extracted.
3. **Text Recognition**: PaddleOCR processes the cropped regions and extracts text.
4. **Display**: The extracted text is overlaid on the original frames for visualization.

### Key Features

- **Multi-Language Support**: PaddleOCR supports various languages for text recognition.
- **Real-Time Capability**: The pipeline is optimized for real-time performance, making it suitable for applications such as traffic monitoring and automated toll systems.

### Algorithm

The following steps outline the algorithm for real-time inference:

1. **Input Frame**:
    - Capture a video frame or load an image.

2. **License Plate Detection**:
    - Use YOLOv5 to detect bounding boxes for license plates.

3. **Region Cropping**:
    - Extract the regions corresponding to the detected bounding boxes.

4. **OCR Processing**:
    - Pass the cropped regions to PaddleOCR for text recognition.

5. **Visualization**:
    - Overlay the detected text on the original frame near the corresponding bounding box.

6. **Output Frame**:
    - Display or save the annotated frame with license plate numbers.

This integration showcases the power of combining advanced object detection with OCR for practical and impactful applications.
