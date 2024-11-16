# License-Detection
# <img src = "https://opencv.org/wp-content/uploads/2021/06/OpenCV_logo_black_.png">

The notebook: `Project2_License_Plate_Detection.ipynb` was used to build a License Plate Detector Model.

### About the Dataset

The dataset consisted of `5694` images samples of in **License plates**.  A few samples are shown below.
![Images](https://github.com/04092000f/License-Detection/blob/main/visuals/image.png)<br>

Out of these `5694` image samples,  `5308` samples consisted of the train data and `386` samples from the validation data.

The annotations are in the form of `[xmin, ymin, xmax, ymax]` format. The dataset shared the following hierarchy:

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

[RetinaNet ResNet50 FPN 3x](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#retinanet) was used for fine-tuning on the dataset.



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

https://github.com/04092000f/License-Detection/visuals/video.mp4


---
