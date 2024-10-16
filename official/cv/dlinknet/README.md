# DLinkNet

## dlinknet-description

D-Linknet model is constructed based on LinkNet architecture. This implementation is as described  in the original paper [D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html).
The model performed best in the 2018 DeepGlobe Road Extraction Challenge. The network uses encoder-decoder structure, cavity convolution and pre-trained encoder to extract road.

In the DeepGlobe Road Extraction Challenge, the raw size of the images and masks provided is 1024×1024, and the roads in most images span the entire image. Still, roads have some natural properties, such as connectivity, complexity, etc. With these attributes in mind, D-Linknet is designed to receive 1024×1024 images as input and retain detailed spatial information. D-linknet can be divided into A, B, C three parts, called encoder, central part and decoder respectively.

D-linknet uses ResNet34, pre-trained on the ImageNet dataset, as its encoder. ResNet34 was originally designed for the classification of 256×256 medium resolution images, but in this challenge the task was to segment roads from 1024×1024 high resolution satellite images. Considering narrowness, connectivity, complexity, and long road spans, it is important to increase the perceived range of features of the central part of the network and retain details. Pooling layer can multiply the felt range of features, but may reduce the resolution of the central feature map and reduce the spatial information. The empty convolution layer may be an ideal alternative to the pooling layer. D-linknet uses several empty convolution layers with skip-connection in the middle.

Empty convolution can be stacked in cascading mode. As shown in the figure above, if the expansion coefficients of the stacked cavity convolution layers are 1, 2, 4, 8 and 16 respectively, then the acceptance fields of each layer will be 3, 7, 15, 31 and 63. The encoder part (ResNet34) has five down-sampling layers. If the image of size 1024×1024 passes through the encoder part, the size of the output feature map will be 32×32. In this case, D-Linknet uses hollow convolution layers with expansion coefficients of 1, 2, 4 and 8 in the central part, so the feature points on the last central layer will see 31×31 points on the first central feature map, covering the main part of the first central feature map. Nevertheless, d-Linknet takes advantage of multi-resolution capabilities, and the central part of D-Linknet can be seen as parallel mode.

The decoder for D-Linknet is the same as the original LinkNet, which is computationally valid. In the decoder part, the transpose convolution layer is used for up-sampling, and the resolution of the feature map is restored from 32×32 to 1024×1024.

## environment-requirements

- [Installing Ascend AI processor software package](https://www.mindspore.cn/install/en#installing-ascend-ai-processor-software-package)
- [MindSpore r2.3.1](https://www.mindspore.cn/install/en)
- [MindSpore Lite r2.3.1](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html)

## Pretrained weights
- [Resnet34-Imagenet](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju)

## dataset

Dataset used： [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- Description: The dataset consisted of 6226 training images, 1243 validation images and 1101 test images. The resolution of each image is 1024×1024. The dataset is represented as a dichotomous segmentation problem, where roads are marked as foreground and other objects as background.
- Dataset size: 3.83 GB

    - Train: 2.79 GB, 6226 images, including the corresponding label image, the original image named 'xxx_sat.jpg', the corresponding label image named 'xxx_mask.png'.
    - Val: 552 MB, 1243 images, no corresponding label image, original image named 'xxx_sat.jpg'.
    - Test: 511 MB, 1101 images, no corresponding label image, original image named 'xxx_sat.jpg'.

- Note: since this data set is used for competition, the label images of the verification set and test set will not be disclosed. I have adopted the method of dividing the training set by one tenth as the verification set to verify the training accuracy of the model.
- The data set shown above is linked to the Kaggle community and can be downloaded directly.

- If you don't want to divide the training set by yourself, you can just download this [baiduNetDisk link](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) , which contains three folders:

    - train: file used for training script, 5604 images, including the corresponding label image, the original image is named `xxx_sat.jpg`, the corresponding label image is named `xxx_mask.png`.
    - valid: file used for the test script. 622 images, not containing the corresponding label image. The original image is named `xxx_sat.jpg`.
    - valid_mask: file used for the eval script. 622 images are the label image corresponding to the valid image named `xxx_mask.png`.

## Training
- Modify the dlinknet_config.yaml file and set the download Resnet34 pretraining weight path

  ```yaml
  pretrained_ckpt: '/xxx/resnet34_xxx.ckpt'
  ```
- Training
  ```shell
  # standalone training command
  python train.py --data_path=[DATASET] --config_path=[CONFIG_PATH] --output_path=[OUTPUT_PATH] --device_target=[DEVICE_TARGET] > train.log 2>&1 & 
   ```
  Parameter Description:

  - data_path: The training image folder path
  - config_path: The training configuration YAML file path
  - output_path: The train output checkpoint file path
  - device_target: The training device type, currently only supports Ascend
  
  ```shell
    # standalone training script
    bash scripts/run_standalone_ascend_train.sh [DATASET] [CONFIG_PATH]

    # distributed training script
    bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]
  ```
  Parameter Description:

  - WORKER_NUM: The number of cards used for training
- Evaluate
  ```shell
    # evaluation command
    python eval.py --data_path=[DATASET] --label_path=[LABEL_PATH] --trained_ckpt=[CHECKPOINT] --predict_path=[PREDICT_PATH] --config_path=[CONFIG_PATH] --device_target=[DEVICE] > eval.log 2>&1 &

    # evaluation script
    bash scripts/run_standalone_ascend_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]
  ```
  Parameter Description:

  - data_path: The image path of the valid dataset
  - label_path: The label path of the valid dataset
  - trained_ckpt: The trained checkpoint file path
  - predict_path: The storage path for predicted results
  - config_path: The training configuration path
  - device_target: The evaluating device type, currently only supports Ascend
## Mindsproe Lite Inference
- Export inference mindir model
  ```shell
   # export model
  python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[model_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
  ```
  Parameter Description:

  - config_path: The training configuration path
  - trained_ckpt: The trained checkpoint file path
  - file_name: The storage path of the exported mindir file

- Inference
  ```shell
  # Ascend310 Inference
  bash run_infer_310.sh [DATA_PATH] [LABEL_PATH] [MINDIR_PATH] [DEVICE_ID]
  ```
  Parameter Description:
  - MINDIR_PATH: The path of exported mindir model
  - DEVICE_ID The device ID used for inference
## Performance
  - Training

    | Ascend|Mindspore|image size|epochs|Speed|loss|Accuracy|
    | ----  |-------- |----------|----  |----|----|----|
    | 910B3|r2.3.1|1024*1024|214|350ms/step|0.201190|IOU: 98%|

  - Inference

    | Ascend|Mindspore-Lite|Speed(FPS)|Accuracy|
    | ----  |----          |----|----|
    | 310|r2.3.1|17|IOU: 98%|
  - Results Show

## Model weights

 - [CKPT Weight Pretrained Resnet34](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju)
 - [CKPT Weight file after training]()
 - [Mindir weight for inference]()
