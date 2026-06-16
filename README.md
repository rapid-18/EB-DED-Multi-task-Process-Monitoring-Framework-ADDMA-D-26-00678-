This repository provides the official implementation of the deep learning model proposed in our paper:

**Multitask Visual Monitoring for Electron Beam Directed Energy Deposition Based on Temporal-Spatial Feature Fusion and Task Incremental Learning**  
Zhenyu Liao, Zixiang Li, Yifeng Zhou, Keming Guan, Li Wang, Baohua Chang*, and Dong Du*  
Additive Manufacturing, Under Review, (ADDMA-D-26-00678)  
## Overview
We release the source code of:
- the proposed deep learning model **TSF-DPMnet and MT-TSF-DPMnet**,
- data preprocessing and data augmentation procedures used in the paper.

Due to **ongoing, unpublished research** and **intellectual property (IP) considerations**, the full raw dataset and complete annotations used in the paper **cannot be publicly released at this time**.

To support transparency, we provide:
- **a small set of representative image samples** under [`data_samples/`](./data_samples) (no full annotations), and

**Data will be made available on reasonable request**, subject to an appropriate data-use agreement and any applicable third-party restrictions.  
Please contact: **Dong Du, dudong@tsinghua.edu.cn**.

> Note: The sample images are provided **for demonstration only** and are not intended to reproduce the full quantitative results reported in the paper.
---
## Requirements
This project is developed on top of **[OpenMMLab] MMSegmentation**, and follows its overall design (configs, datasets, training/testing APIs).

- MMSegmentation: https://github.com/open-mmlab/mmsegmentation
- OpenMMLab: https://openmmlab.com/
Other requiernments are shown in requirements.txt
**Note:** This repository contains our custom model components and the corresponding preprocessing/data augmentation pipeline implemented within the MMSegmentation framework.
