
markdown
Copy code
# License Plate Detection

![OpenCV Logo](https://opencv.org/wp-content/uploads/2021/06/OpenCV_logo_black_.png)

This project uses **YOLOv5** to build a high-precision **License Plate Detection Model**. The model was fine-tuned on a custom dataset and achieved excellent detection accuracy.

---

## Table of Contents

1. [About the Dataset](#1-about-the-dataset)
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



---

### Detection Model used for Fine-tuning

[YOLOv5](https://github.com/ultralytics/yolov5)t) was used for fine-tuning on the dataset.

Since <b>YOLOv5</b> was used, there are two things that were needed to be done:
    - Convert the and preprocess the annotations in a proper YOLOv5 format, the format is given below:
            `class_id x y width height`

    - Create a data.yaml file which stores class labels corresponding to their class ids. This file is very essential for model training of <b>YOLOv5</b> model. The file format is given below:
               ``` 
               train: /content/License_Plate_Detection/train/images
               val: /content/License_Plate_Detection/validation/images
               nc: 1
               names: ['Reg-plate']
               ```



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
