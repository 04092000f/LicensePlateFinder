
markdown
Copy code
# License Plate Detection

![OpenCV Logo](https://opencv.org/wp-content/uploads/2021/06/OpenCV_logo_black_.png)

This project uses **YOLOv5** to build a high-precision **License Plate Detection Model**. The model was fine-tuned on a custom dataset and achieved excellent detection accuracy.

---

## Table of Contents

1. [About the Dataset](#about-the-dataset)
2. [Model and Preprocessing](#model-and-preprocessing)
3. [Training Configuration](#training-configuration)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Inference Results](#inference-results)
6. [Setup Instructions](#setup-instructions)
7. [References](#references)

---

## About the Dataset

The dataset consists of **5,694 images** of license plates with bounding box annotations in `[xmin, ymin, xmax, ymax]` format. The dataset follows the directory structure:

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

Here are a few sample images from the dataset:

![Dataset Samples](https://github.com/04092000f/License-Detection/blob/main/visuals/image.png)

---

## Model and Preprocessing

**YOLOv5** was selected for its speed and accuracy. Key preprocessing steps included:

1. **Annotation Conversion**: Converted bounding box annotations into YOLO format:


2. **Configuration File**: Created a `data.yaml` file:
```yaml
train: /content/License_Plate_Detection/train/images
val: /content/License_Plate_Detection/validation/images
nc: 1
names: ['Reg-plate']

---

### Training Hyperparameters:

* Batch size: `32`

* Epochs: `25`; Iterations: `int(epoch * train_img_count / BATCH_SIZE)`  = `4146`

* Initial LR: `1e-3`

  

### Evaluation Metrics

The final COCO primary metric (**mAP@0.5:0.95**) was **`0.762`**, while the **mAP@0.5** (VOC metric) was **0.831**. The overall metric statistics is:

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


### Video Inference

![Video](visuals/video.gif)




---
