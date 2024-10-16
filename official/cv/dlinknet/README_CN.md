# DLinkNet

## 模型简介

D-LinkNet模型基于LinkNet架构构建。实现方式见论文[D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html)
在2018年的DeepGlobe道路提取挑战赛中，这一模型表现最好。该网络采用编码器-解码器结构、空洞卷积和预训练编码器进行道路提取任务。

在“DeepGlobe道路提取挑战”中，提供的image和mask的原始大小为1024×1024，并且大多数图像中的道路跨越整个图像。尽管如此，道路仍具有一些自然属性，例如连通性、复杂性等。考虑到这些属性，D-LinkNet旨在接收1024×1024图像作为输入并保留详细的空间信息。D-LinkNet可分为A，B，C三个部分，分别称为编码器，中央部分和解码器。

D-LinkNet使用在ImageNet数据集上预训练的ResNet34作为其编码器。ResNet34最初是为256×256尺寸的中分辨率图像分类而设计的，但在这一挑战中，任务是从1024×1024的高分辨率卫星图像中分割道路。考虑到狭窄性、连通性、复杂性和道路跨度长等方面，重要的是增加网络中心部分的特征的感受范围，并保留详细信息。使用池化层可以成倍增加特征的感受范围，但可能会降低中心特征图的分辨率并降低空间信息。空洞卷积层可能是池化层的理想替代方案。D-LinkNet使用几个空洞卷积层，中间部分带有skip-connection。

空洞卷积可以级联模式堆叠。如果堆叠的空洞卷积层的膨胀系数分别为1、2、4、8、16，则每层的接受场将为3、7、15、31、63。编码器部分（ResNet34）具有5个下采样层，如果大小为1024×1024的图像通过编码器部分，则输出特征图的大小将为32×32。在这种情况下，D-LinkNet在中心部分使用膨胀系数为1、2、4、8的空洞卷积层，因此最后一个中心层上的特征点将在第一个中心特征图上看到31×31点，覆盖第一中心特征图的主要部分。尽管如此，D-LinkNet还是利用了多分辨率功能，D-LinkNet的中心部分可以看作是并行模式。

D-LinkNet的解码器与原始LinkNet相同，这在计算上是有效的。解码器部分使用转置卷积层进行上采样，将特征图的分辨率从32×32恢复到1024×1024。

## 环境要求

- [昇腾AI处理器配套软件包](https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85)
- [MindSpore r2.3.1](https://www.mindspore.cn/install)
- [MindSpore Lite r2.3.1](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)
- 安装requirements.txt

## 预训练权重
- [Resnet34-Imagenet](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju)

## 数据集

[DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- 说明：该数据集由6226个训练图像、1243个验证图像和1101个测试图像组成。每个图像的分辨率为1024×1024。数据集被表述为二分类分割问题，其中道路被标记为前景，而其他对象被标记为背景。
- 数据集大小：3.83 GB

    - 训练集：2.79 GB，6226张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - 验证集：552 MB，1243张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - 测试集：511 MB，1101张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。

- 注意：由于该数据集为比赛用数据集，验证集与测试集的标签图像不会公开，本人在采用了将训练集划出十分之一作为验证集验证模型训练精度的方法。
- 上面给出的数据集链接为上传到Kaggle社区中的，可以直接下载。

- 如果你不想自己划分训练集，你可以只下载 [这个百度网盘链接](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) ，其中包含了三个文件夹：

    - train：用于训练脚本的文件，5604张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - valid：用于测试脚本的文件，622张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - valid_mask：用于评估脚本的文件，622张图像，是valid中图像对应的标签图像，以`xxx_mask.png`命名。
    
## 训练
- 修改dlinknet_config.yaml文件，设置下载的Resnet34预训练权重路径

  ```yaml
  pretrained_ckpt: '/xxx/resnet34_xxx.ckpt'
  ```
- 训练
  ```shell
  # 单卡训练命令
  python train.py --data_path=[DATASET] --config_path=[CONFIG_PATH] --output_path=[OUTPUT_PATH] --device_target=[DEVICE_TARGET] > train.log 2>&1 & 
   ```
  参数说明:
  
  - data_path: 训练数据集的路径
  - config_path: 训练配置yaml文件路径
  - output_path: 训练输出checkpoint路径
  - device_target: 训练设备类型，当前仅支持Ascend
  
  ```shell
    # 训练脚本
    bash scripts/run_standalone_ascend_train.sh [DATASET] [CONFIG_PATH]
  
    # 分布式训练
    bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]
  ```
  参数说明:

  - WORKER_NUM: 用于训练的卡数
- 评估
  ```shell
    # 评估示例
    python eval.py --data_path=[DATASET] --label_path=[LABEL_PATH] --trained_ckpt=[CHECKPOINT] --predict_path=[PREDICT_PATH] --config_path=[CONFIG_PATH] --device_target=[DEVICE] > eval.log 2>&1 &
  
    # 评估脚本启动
    bash scripts/run_standalone_ascend_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]
  ```
    参数说明:

  - data_path: 验证集原始图像路径
  - label_path: 验证集标签路径
  - trained_ckpt: 训练后的checkpoint路径
  - predict_path: 评估预测的结果存放路径
  - config_path: 训练配置路径
  - device_target: 训练设备类型，当前仅支持Ascend
## Mindsproe Lite 推理  
- 导出推理mindir模型
  ```shell
   # 模型导出
  python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[model_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
  ```
  参数说明:
  - config_path: 训练配置路径
  - trained_ckpt 训练保存的checkpoint路径
  - file_name: 导出的mindir存放路径

- 推理
  ```shell
  # Ascend310 推理
  bash run_infer_310.sh [DATA_PATH] [LABEL_PATH] [MINDIR_PATH] [DEVICE_ID]
  ```
    参数说明:
  - MINDIR_PATH: 导出的mindir模型路径
  - DEVICE_ID 用于推理的设备ID
## 模型指标
  - 训练 

    | Ascend|Mindspore|image size|epochs|性能|损失|精度|
    | ----  |-------- |----------|----  |----|----|----|
    | 910B3|r2.3.1|1024*1024|214|350ms/step|0.201190|IOU: 98%|

  - 推理

    | Ascend|Mindspore-Lite|性能FPS|精度|
    | ----  |----          |----|----|
    | 310|r2.3.1|17|IOU: 98%|
  - 效果展示

## 相关模型权重

 - [预训练Resnet34 ckpt](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju)
 - [训练后ckpt]()
 - [推理mindir]()
