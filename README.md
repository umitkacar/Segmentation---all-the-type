<div align="center">

# 🎯 Awesome Segmentation - All Types

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&duration=2800&pause=2000&color=00ADD8&center=true&vCenter=true&width=940&lines=Medical+%7C+Industrial+%7C+Biometric;Semantic+%7C+Instance+%7C+Panoptic;3D+%7C+Video+%7C+Real-time;Complete+Segmentation+Research+Hub" alt="Typing SVG" />

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/umitkacar/Segmentation---all-the-type?style=social)](https://github.com/umitkacar/Segmentation---all-the-type)
[![Last Commit](https://img.shields.io/github/last-commit/umitkacar/Segmentation---all-the-type)](https://github.com/umitkacar/Segmentation---all-the-type/commits/main)

*The most comprehensive collection of segmentation resources across all domains - Medical, Industrial, Biometric, and more!*

[🚀 Getting Started](#-getting-started) • [📚 Categories](#-categories) • [🎓 Papers](#-papers) • [🔬 Datasets](#-datasets) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🏥 Medical Image Segmentation](#-medical-image-segmentation)
- [🎨 Semantic Segmentation](#-semantic-segmentation)
- [🔍 Instance Segmentation](#-instance-segmentation)
- [🌐 Panoptic Segmentation](#-panoptic-segmentation)
- [👁️ Biometric Segmentation](#️-biometric-segmentation)
- [🏭 Industrial & Defect Segmentation](#-industrial--defect-segmentation)
- [🎬 Video Segmentation](#-video-segmentation)
- [🧊 3D Point Cloud Segmentation](#-3d-point-cloud-segmentation)
- [✨ Interactive & Foundation Models](#-interactive--foundation-models)
- [⚡ Real-time & Mobile Segmentation](#-real-time--mobile-segmentation)
- [📊 Datasets](#-datasets)
- [🎓 Survey Papers](#-survey-papers)

---

## 🏥 Medical Image Segmentation

*State-of-the-art models for MRI, CT, X-ray, Ultrasound, and Pathology image segmentation*

### 🌟 State-of-the-Art Models (2024-2025)

| Model | Venue | Description | Stars | Code |
|-------|-------|-------------|-------|------|
| **nnU-Net** | Nature Methods 2020 | Self-configuring framework for medical image segmentation | ⭐ 7k+ | [GitHub](https://github.com/MIC-DKFZ/nnUNet) |
| **SAM4MIS** | CIBM 2024 | Segment Anything Model for Medical Images | ⭐ 500+ | [GitHub](https://github.com/YichiZhang98/SAM4MIS) |
| **SOTA-MedSeg** | - | Collection of SOTA methods across challenges | ⭐ 1k+ | [GitHub](https://github.com/JunMa11/SOTA-MedSeg) |
| **TotalSegmentator** | - | Automatic segmentation of 104 anatomical structures | ⭐ High | - |

### 📑 Key Papers

- **[U-Net in Medical Image Segmentation: A Review (Dec 2024)](https://arxiv.org/abs/2412.02242)** - Comprehensive review across modalities
- **nnU-Net: self-configuring method for deep learning-based biomedical image segmentation** - 9/10 MICCAI 2020 winners built on this
- **nnSAM: Plug-and-play SAM Improves nnUNet Performance**
- **SAM2-PATH: SAM2 for semantic segmentation in digital pathology**

### 🛠️ Frameworks & Tools

- **nnU-Net** - Automatic configuration for any dataset
- **MONAI** - PyTorch-based framework for healthcare imaging
- **MedicalSeg** - End-to-end medical image segmentation library
- **SegmentAnything Medical** - SAM adaptations for medical imaging

### 🔬 Datasets

- **Medical Segmentation Decathlon** - 10 medical imaging tasks
- **BRATS** - Brain tumor segmentation
- **ACDC** - Automated Cardiac Diagnosis Challenge
- **LiTS** - Liver Tumor Segmentation
- **KiTS** - Kidney Tumor Segmentation
- **ISIC** - Skin lesion segmentation

### 📂 Topic Collections

- [medical-image-segmentation](https://github.com/topics/medical-image-segmentation) - 200+ repositories
- [mri-segmentation](https://github.com/topics/mri-segmentation) - MRI-specific implementations

---

## 🎨 Semantic Segmentation

*Dense pixel-wise classification for scene understanding*

### 🌟 State-of-the-Art Models (2024-2025)

| Model | Architecture | Highlights | Code |
|-------|-------------|-----------|------|
| **DeepLabV3+** | Encoder-Decoder + ASPP | Industry standard, excellent performance | [GitHub](https://github.com/topics/deeplabv3plus) |
| **PSPNet** | Pyramid Pooling Module | Multi-scale context aggregation | [GitHub](https://github.com/topics/pspnet) |
| **Segformer** | Transformer-based | Efficient attention mechanism | - |
| **Mask2Former** | Transformer | Universal segmentation architecture | - |
| **SegNext** | CVPR 2022 | Efficient transformer for semantic seg | - |

### 📚 Comprehensive Repositories

#### Top Collections

1. **[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)** ⭐ 9k+
   - OpenMMLab's toolbox with 50+ models
   - Includes PSPNet, DeepLabv3, Transformers
   - Updated August 2024

2. **[Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)** ⭐ 10.3k+
   - Pretrained weights for multiple architectures
   - PSPNet, FPN, DeepLabv3, DeepLab-v3-plus
   - Updated July 2025

3. **[pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)**
   - Minimal implementation for PSPNet and Deeplabv3
   - Cityscapes dataset support

4. **[Amazing-Semantic-Segmentation](https://github.com/luyanger1799/Amazing-Semantic-Segmentation)**
   - TensorFlow & Keras implementations
   - FCN, UNet, SegNet, PSPNet, PAN, RefineNet, DeepLabV3, DeepLabV3+

5. **[Semantic-Segmentation-PyTorch](https://github.com/Charmve/Semantic-Segmentation-PyTorch)**
   - FCN, U-Net, SegNet, GCN, PSPNet, Deeplabv3, Deeplabv3+
   - Multiple dataset support

### 🏆 Benchmark Datasets

- **Cityscapes** - Urban street scenes
- **ADE20K** - Scene parsing with 150 categories
- **PASCAL VOC** - 21 object categories
- **COCO-Stuff** - 171 categories
- **Mapillary Vistas** - Street-level imagery

---

## 🔍 Instance Segmentation

*Detect and segment individual object instances*

### 🌟 State-of-the-Art Models (2024-2025)

| Model | Year | mAP (COCO) | Speed (FPS) | Code |
|-------|------|-----------|------------|------|
| **BEiT3** | 2024 | 54.8 | - | - |
| **MaskDINO** | 2024 | 52.3 (mask) / 59.0 (box) | - | - |
| **Mask2Former** | 2022 | 50.5 | - | - |
| **RF-DETR-Seg** | 2025 | 44.3 | 170+ | [Preview](https://blog.roboflow.com/rf-detr-segmentation-preview/) |
| **BorderMask** | 2025 | 44.7 | - | Scientific Reports |
| **Mask R-CNN** | 2017 | 38-40 | - | [GitHub](https://github.com/matterport/Mask_RCNN) |
| **YOLACT** | 2019 | 34.1 | 33.5 | [GitHub](https://github.com/dbolya/yolact) |

### 🚀 Recent Breakthroughs

#### RF-DETR-Seg (October 2025)
- **3x faster** than largest YOLO11
- **44.3 mAP@50:95** on COCO
- **170+ FPS** on T4 GPU
- New real-time SOTA

#### OMG-Seg (2024)
- All-in-one segmentation model
- Supports 10+ segmentation tasks
- Transformer-based encoder-decoder

### 📦 Popular Implementations

- **[Mask_RCNN](https://github.com/matterport/Mask_RCNN)** - Keras & TensorFlow
- **[YOLACT](https://github.com/dbolya/yolact)** - Real-time instance segmentation
- **[CenterMask](https://github.com/youngwanLEE/CenterMask)** - CVPR 2020, Anchor-free
- **[Detectron2](https://github.com/facebookresearch/detectron2)** - Facebook AI Research

### 🔗 Resources

- [instance-segmentation topic](https://github.com/topics/instance-segmentation) - 500+ repos
- [COCO Instance Segmentation Benchmark](https://paperswithcode.com/sota/instance-segmentation-on-coco)

---

## 🌐 Panoptic Segmentation

*Unified semantic and instance segmentation*

### 🎯 Concept

Panoptic Segmentation = **Semantic Segmentation** (Stuff) + **Instance Segmentation** (Things)

### 🌟 State-of-the-Art Models

| Model | Venue | PQ (COCO) | Highlights |
|-------|-------|-----------|-----------|
| **Panoptic-DeepLab** | CVPR 2020 | 65.5 | Fast bottom-up approach |
| **Mask2Former** | 2022 | - | Universal architecture |
| **K-Net** | NeurIPS 2021 | - | Towards unified segmentation |
| **OneFormer** | - | - | Universal image segmentation |
| **EoMT** | CVPR 2025 | - | Encoder-only Mask Transformer |

### 📚 Key Implementations

1. **[panoptic-deeplab](https://github.com/bowenc0221/panoptic-deeplab)** - PyTorch (CVPR 2020)
   - Dual-ASPP and dual-decoder architecture
   - Can train on 4x 1080Ti GPUs
   - Reproduces COCO results (35.5 PQ)

2. **[deeplab2](https://github.com/google-research/deeplab2)** - Google Research
   - Unified TensorFlow codebase
   - State-of-the-art dense labeling

3. **[Awesome-Panoptic-Segmentation](https://github.com/Angzz/awesome-panoptic-segmentation)**
   - Comprehensive paper list
   - Dataset collection

### 🏆 Benchmarks

- **Cityscapes Panoptic** - 84.2% mIoU, 39.0% AP, 65.5% PQ
- **COCO Panoptic** - Standard benchmark
- **ADE20K Panoptic** - Scene understanding

---

## 👁️ Biometric Segmentation

*Iris, fingerprint, face, palm vein, and other biometric modalities*

### 🔐 Modalities

#### 👁️ Iris Recognition
- **Deep Learning for Iris Recognition** - ACM Computing Surveys (2024)
- Covers 200+ articles, papers, and GitHub repos from last 10 years
- Iris segmentation databases and triplet loss systems
- Updated through February 2025

#### 🖐️ Palm Vein Recognition
- **Review of Palm Vein Biometric Recognition** - IJEEE 2025
- Transformers and attention-based mechanisms
- High accuracy, low susceptibility to forgery

#### 👆 Fingerprint Recognition
- Feature extraction and matching algorithms
- Minutiae-based and deep learning approaches
- [fingerprint-recognition topic](https://github.com/topics/fingerprint-recognition)

### 📑 Recent Surveys & Papers (2024-2025)

1. **Deep learning techniques for hand vein biometrics: A comprehensive review** (ScienceDirect 2024)
   - Finger vein, palm vein, dorsal hand vein (2015-2024)
   - Non-intrusive, high accuracy

2. **Feature extraction and learning approaches for cancellable biometrics** (CAAI 2024)
   - Security and privacy aspects

3. **Towards generation of synthetic palm vein patterns** (ScienceDirect 2022)
   - Data augmentation techniques

### 🛠️ GitHub Resources

- [iris-recognition topic](https://github.com/topics/iris-recognition) - Segmentation & recognition
- [biometrics topic](https://github.com/topics/biometrics) - Updated Feb 2025
- [biometric-identification](https://github.com/topics/biometric-identification) - Multi-modal systems

### 🔬 Applications

- **Multi-modal Biometrics** - Face + Iris + Palmprint + Fingerprint + Ear
- **CNN-based Deep Features** - End-to-end learning
- **Cancellable Biometrics** - Privacy-preserving approaches

---

## 🏭 Industrial & Defect Segmentation

*Quality control, defect detection, and anomaly segmentation in manufacturing*

### 🌟 State-of-the-Art Repositories (2024-2025)

#### 1. **[awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)** ⭐ Highly Active

**Latest Papers:**
- ECCV 2024, CVPR 2025, AAAI 2025, ICLR 2025
- Few-shot anomaly generation
- Glass defect detection with diffusion models
- Fabric defect detection

**Datasets:**
- **IM-IAD** - Industrial Image Anomaly Detection Benchmark

#### 2. **[Machine-Vision-and-Anomaly-Detection-Papers](https://github.com/djene-mengistu/Machine-Vision-and-Anomaly-Detection-Papers)**

SOTA deep learning models for:
- Industrial anomaly detection
- Defect segmentation & detection
- Defect classification
- Industrial machine vision applications

#### 3. **[3CAD Dataset](https://github.com/enquanyang2022/3cad)** - AAAI 2025

- **27,039 high-resolution images**
- 8 different manufactured parts
- Pixel-level anomaly labels
- First dataset for 3C product quality control

#### 4. **[GLASS](https://github.com/cqylunlun/GLASS)** - ECCV 2024

- Unified anomaly synthesis with Gradient Ascent
- Self-built datasets: WFDD, MAD-man, MAD-sys
- Addresses coverage & controllability

### 🔥 Recent Developments (2024-2025)

| Topic | Venue | Innovation |
|-------|-------|-----------|
| Multimodal LLMs for IAD | ICLR 2025 | Language-vision models |
| Enhanced Fabric Defect Detection | TIM 2025 | Advanced CNNs |
| Glass Defect with Diffusion Models | 2024 | Imbalanced dataset handling |
| Deep Learning Proactive QC | 2024 | Preventive quality control |

### 🛠️ Key Technologies

- **Anomaly Detection** - Unsupervised & semi-supervised
- **Defect Localization** - Pixel-level segmentation
- **Quality Control** - Real-time inspection
- **Anomalib** - Intel's anomaly detection library

### 📂 Topic Collections

- [defect-detection](https://github.com/topics/defect-detection)
- [defect-segmentation](https://github.com/topics/defect-segmentation)
- [anomaly-detection-in-industry-manufacturing](https://github.com/vnk8071/anomaly-detection-in-industry-manufacturing)

---

## 🎬 Video Segmentation

*Temporal consistency and tracking across video frames*

### 🌟 State-of-the-Art Models (2024-2025)

| Model | Venue | Approach | FPS | Code |
|-------|-------|----------|-----|------|
| **SSP** | CVPR 2025 | Semantic Similarity Propagation | - | [GitHub](https://github.com/fraunhoferivi/ssp) |
| **HTR** | TCSVT 2024 | Hybrid Memory for Temporal Consistency | - | [GitHub](https://github.com/bo-miao/HTR) |
| **STCN** | NeurIPS 2021 | Space-Time Correspondence Networks | 20+ | [GitHub](https://github.com/hkchengrex/STCN) |
| **XMem** | ECCV 2022 | Long-term video object segmentation | - | - |
| **XMem++** | - | Enhanced memory mechanism | - | - |
| **MemSAM** | CVPR 2024 | SAM for Echocardiography Videos | - | - |

### 📚 Comprehensive Collections

#### 1. **[Awesome-Video-Object-Segmentation](https://github.com/gaomingqi/Awesome-Video-Object-Segmentation)** 🔥
- Latest VOS papers
- Datasets: MOSEv2 (2025), SA-V (2024)
- Temporal modeling methods

#### 2. **[awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation)**
- Curated VOS paper list
- 2024 updates

### 🎯 Key Concepts

#### Space-Time Memory Networks (STM)
- Store features from previous frames
- Query matching for propagation
- Used in STCN, XMem, XMem++

#### Temporal Consistency
- Frame-to-frame coherence
- Prevents flickering
- Smooth transitions

### 📊 Applications

- **Semi-supervised VOS** - First frame annotation
- **Referring VOS** - Language-guided segmentation
- **Medical Video** - Echocardiography, endoscopy
- **Autonomous Driving** - Object tracking

### 🔗 Resources

- [video-object-segmentation topic](https://github.com/topics/video-object-segmentation)
- **DAVIS** - Densely Annotated VIdeo Segmentation
- **YouTube-VOS** - Large-scale VOS dataset

---

## 🧊 3D Point Cloud Segmentation

*Segmentation of 3D data from LiDAR, RGB-D cameras, and mesh*

### 🌟 State-of-the-Art Models (2024-2025)

| Model | Type | Innovation | Code |
|-------|------|-----------|------|
| **PointCT** | WACV 2024 | Weakly-supervised transformer | - |
| **PointNeXt** | NeurIPS 2022 | Improved PointNet++ | [PyTorch](https://github.com/guochengqian/PointNeXt) |
| **SPoTr** | CVPR 2023 | Self-positioning transformers | - |
| **AF-GCN** | CVPR 2023 | Attentive filtering | - |
| **Stratified Transformer** | CVPR 2022 | Hierarchical attention | - |
| **DRINet++** | - | Efficient voxel-as-point | - |
| **PV-RCNN++** | - | Point-voxel feature abstraction | - |

### 📚 Comprehensive Repositories

#### 1. **[3D-PointCloud](https://github.com/zhulf0804/3D-PointCloud)** 📦
- Papers and datasets
- DRINet++, PV-RCNN++, DPointNet
- Voxel-based methods

#### 2. **[awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis)** ⭐
- Original PointNet implementations
- TensorFlow & PyTorch versions
- Comprehensive paper list

#### 3. **[awesome-point-cloud-analysis-2023](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2023)**
- Updated daily since 2017
- Large-scale segmentation
- Superpoint graphs

### 🏗️ Architecture Types

#### Pure Point-based
- **PointNet** - First deep learning on point sets
- **PointNet++** - Hierarchical feature learning
- **DPointNet** - Density-oriented variant

#### Point-Voxel Hybrid
- **VoxelNet** - End-to-end 3D object detection
- **PV-RCNN** - Point-voxel feature set abstraction
- Split space into voxels + PointNet features

#### Transformer-based
- **Point Transformer** - Self-attention for points
- **Stratified Transformer** - Hierarchical approach
- **SPoTr** - Self-positioning mechanism

### 🎯 Applications

- **Autonomous Driving** - LiDAR segmentation
- **Robotics** - 3D scene understanding
- **Indoor Scenes** - S3DIS, ScanNet
- **Outdoor Scenes** - SemanticKITTI, nuScenes
- **CAD & Manufacturing** - Mesh segmentation

### 🔬 Datasets

- **S3DIS** - Stanford Large-Scale 3D Indoor Spaces
- **ScanNet** - RGB-D indoor scenes
- **SemanticKITTI** - Outdoor LiDAR sequences
- **ModelNet** - CAD models
- **ShapeNet** - 3D shape repository

### 🔗 Topic Collections

- [point-cloud-segmentation](https://github.com/topics/point-cloud-segmentation)
- [pointnet2](https://github.com/topics/pointnet2)
- [pointcloud-segmentation](https://github.com/topics/pointcloud-segmentation)

---

## ✨ Interactive & Foundation Models

*Promptable segmentation with SAM, SAM2, and beyond*

### 🚀 Meta's Segment Anything Models

#### **SAM (Segment Anything Model)** - 2023
[![GitHub](https://img.shields.io/github/stars/facebookresearch/segment-anything?style=social)](https://github.com/facebookresearch/segment-anything)

- **Zero-shot transfer** - Generalizes to new domains
- **Interactive prompts** - Points, boxes, masks
- **Foundation model** - Learned general object notion
- **1B+ mask dataset** - SA-1B dataset

#### **SAM 2 (Segment Anything in Images and Videos)** - 2024
[![GitHub](https://img.shields.io/github/stars/facebookresearch/sam2?style=social)](https://github.com/facebookresearch/sam2)

- **🏆 ICLR 2025 Best Paper Honorable Mention**
- **Real-time video** - ~44 FPS processing
- **Streaming memory** - Efficient video handling
- **Zero-shot generalization** - Unseen objects/videos
- **SAM 2.1** - Released Sep 29, 2024

#### **SAM 3** - Coming 2025 🎉
Officially announced, launching in 2025!

### 📚 Awesome SAM Collections

#### 1. **[Awesome-Segment-Anything](https://github.com/liliu-avril/Awesome-Segment-Anything)**
- First comprehensive SAM survey
- Papers, applications, extensions

#### 2. **[Awesome-SAM2](https://github.com/GuoleiSun/Awesome-SAM2)** 🔥
- Papers, codes, slides about SAM2
- Image and video segmentation
- Continuously updated

### 🏥 SAM for Medical Imaging

**[SAM4MIS](https://github.com/YichiZhang98/SAM4MIS)** - CIBM 2024

Recent medical adaptations:
- **nnSAM** - Plug-and-play improves nnUNet
- **SAM2-PATH** - Digital pathology segmentation
- **SAM for Tooth Segmentation** - Dental X-rays
- **Medical SAM Adapter** - Domain adaptation

### 🔌 Integration with Detection Models

#### YOLO + SAM
- **YOLOv8-seg** - 11.7x smaller, 1069x faster than SAM-b
- **YOLOv11-seg** - Even more efficient
- **YOLO + SAM2** - Enhanced video segmentation

### 🎯 ASAM - Adversarial Tuning (CVPR 2024)
Boosting SAM with adversarial training

### 🌐 Interactive Segmentation APIs

Official SAM 2 supports:
- **Image prompts** - Points, boxes, masks
- **Video prompts** - Add prompts and propagate
- **Multi-object tracking** - Simultaneous segmentation

### 📊 Performance Comparison

| Model | Speed | Size | Accuracy | Use Case |
|-------|-------|------|----------|----------|
| SAM-b | Baseline | Large | High | Research |
| SAM 2 | 44 FPS | Medium | Very High | Production |
| YOLOv11-seg | 170+ FPS | Small | High | Real-time |

---

## ⚡ Real-time & Mobile Segmentation

*Efficient models for edge devices, mobile phones, and embedded systems*

### 🌟 Lightweight Architectures (2024-2025)

| Model | Backbone | Speed | Platform | Code |
|-------|----------|-------|----------|------|
| **MobileNetV3-Seg** | MobileNetV3 | Real-time | Mobile | [GitHub](https://github.com/topics/mobilenetv3) |
| **EfficientNet-Seg** | EfficientNet | High | Edge | Multiple |
| **NanoDet-Plus** | EfficientNet | 97 FPS | Cellphone | Updated 2024 |
| **Fast-SCNN** | - | Real-time | Mobile | Multiple |
| **BiSeNet** | Bilateral | Real-time | Edge | Multiple |

### 📚 Comprehensive Repositories

#### 1. **[Lightweight-Segmentation](https://github.com/Tramac/Lightweight-Segmentation)** 🎯
Models included:
- MobileNetV1, V2, V3
- ShuffleNetV1, V2
- IGCv3
- EfficientNet

#### 2. **[Efficient-Segmentation-Networks](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)** 📦

Complete collection:
- **SQNet, LinkNet, SegNet**
- **UNet, ENet, ERFNet**
- **EDANet, ESPNet, ESPNetv2**
- **LEDNet, ESNet, FSSNet**
- **CGNet, DABNet, Fast-SCNN**
- **ContextNet, FPENet**

#### 3. **[real-time-network](https://github.com/wpf535236337/real-time-network)**
- Mobile device optimization
- MobileNetV3, EfficientNet, MixNet
- Comprehensive architecture references

#### 4. **[mobile-semantic-segmentation](https://github.com/akirasosa/mobile-semantic-segmentation)**
- MobileNetV2 + U-Net inspired
- Hair segmentation demo
- Mobile device deployment

### ⚙️ Optimization Techniques

#### TensorRT Acceleration
**[mobilenet](https://github.com/norbertmarko/mobilenet)**
- MobileNet V1/V2 + SkipNet
- TensorRT optimization
- Fastest inference on mobile

#### Quantization & Pruning
- INT8 quantization
- Channel pruning
- Knowledge distillation

### 🎯 Architecture Patterns

#### Encoder-Decoder
- **Lightweight encoder** - MobileNet, EfficientNet
- **Efficient decoder** - Atrous conv, pyramid pooling

#### Two-Branch Networks
- **BiSeNet** - Spatial + Context paths
- **Fast-SCNN** - Learning to Downsample

#### Hybrid Approaches
- **MobileNetV3-DeepLabV3** - Best of both worlds
- **EfficientNet-UNet** - Compound scaling

### 📱 Deployment Platforms

- **TensorFlow Lite** - Android & iOS
- **ONNX Runtime** - Cross-platform
- **Core ML** - iOS devices
- **NCNN** - Mobile-optimized framework
- **MNN** - Alibaba's mobile framework

### 🏆 Benchmark Results

| Model | Cityscapes mIoU | Speed (FPS) | Parameters |
|-------|----------------|------------|------------|
| BiSeNet V2 | 75.8 | 124 | 3.4M |
| Fast-SCNN | 68.0 | 123 | 1.1M |
| ESPNetv2 | 66.4 | 112 | 1.2M |

### 🔗 Resources

- [mobilenetv3](https://github.com/topics/mobilenetv3)
- [efficientnet](https://github.com/topics/efficientnet)
- [mobilenet-v2](https://github.com/topics/mobilenet-v2)
- [real-time-semantic-segmentation](https://github.com/topics/real-time-semantic-segmentation)

---

## 📊 Datasets

### Medical Imaging
- **Medical Segmentation Decathlon** - 10 tasks, multiple modalities
- **BRATS** - Brain tumor MRI
- **ACDC** - Cardiac MRI
- **LiTS** - Liver & tumor CT
- **KiTS** - Kidney & tumor CT
- **ISIC** - Skin lesion images

### Natural Images
- **COCO** - 330K images, 80 categories
- **Cityscapes** - 5K urban street scenes
- **ADE20K** - 25K images, 150 categories
- **PASCAL VOC** - 11K images, 21 categories
- **Mapillary Vistas** - 25K street-level images

### Industrial
- **3CAD** - 27K images, 3C products (AAAI 2025)
- **IM-IAD** - Industrial anomaly benchmark
- **MVTec AD** - Anomaly detection dataset
- **DAGM** - Defect detection
- **WFDD, MAD-man, MAD-sys** - Manufacturing defects

### 3D Point Cloud
- **S3DIS** - Stanford 3D indoor spaces
- **ScanNet** - RGB-D indoor scenes
- **SemanticKITTI** - LiDAR sequences
- **nuScenes** - Autonomous driving
- **ModelNet** - 3D CAD models
- **ShapeNet** - Large-scale 3D shapes

### Video
- **DAVIS** - Densely Annotated VIdeo Segmentation
- **YouTube-VOS** - 4,453 videos
- **MOSEv2** - 2025 release
- **SA-V** - Segment Anything Videos (2024)

### Biometric
- **CASIA-Iris** - Iris recognition
- **FVC** - Fingerprint verification
- **PolyU** - Palmprint & palm vein databases
- **NIST** - Various biometric datasets

---

## 🎓 Survey Papers

### Recent Comprehensive Surveys (2024-2025)

1. **U-Net in Medical Image Segmentation: A Review** (Dec 2024)
   - [arXiv:2412.02242](https://arxiv.org/abs/2412.02242)
   - Covers X-ray, MRI, CT, Ultrasound

2. **Deep Learning for Iris Recognition: A Survey** (ACM Computing Surveys 2024)
   - 200+ articles and repositories
   - 10 years of research

3. **Deep learning techniques for hand vein biometrics** (ScienceDirect 2024)
   - 2015-2024 comprehensive review
   - Finger, palm, dorsal hand veins

4. **A survey of deep learning for industrial visual anomaly detection** (AI Review 2025)
   - Industrial defect detection
   - Latest deep learning techniques

5. **Image Segmentation: State-of-the-Art Models in 2025**
   - Complete overview of SOTA methods
   - Benchmarks and comparisons

### Classic Foundational Papers

- **PointNet: Deep Learning on Point Sets** (CVPR 2017)
- **Mask R-CNN** (ICCV 2017)
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** (MICCAI 2015)
- **DeepLab** series (v1-v3+)
- **PSPNet: Pyramid Scene Parsing Network** (CVPR 2017)

---

## 🚀 Getting Started

### For Researchers

1. **Medical Imaging?** Start with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
2. **General Segmentation?** Try [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
3. **Interactive?** Use [SAM 2](https://github.com/facebookresearch/sam2)
4. **Real-time?** Check [Lightweight-Segmentation](https://github.com/Tramac/Lightweight-Segmentation)
5. **3D Data?** Explore [PointNet++](https://github.com/charlesq34/pointnet2)

### For Practitioners

```bash
# Quick start with PyTorch
pip install segmentation-models-pytorch
pip install albumentations
```

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit:

- 📄 **Papers** - Recent publications in segmentation
- 💻 **Code** - New implementations or improvements
- 📊 **Datasets** - New annotated datasets
- 🐛 **Issues** - Report errors or suggest enhancements

### How to Contribute

1. Fork this repository
2. Add your contribution
3. Submit a pull request
4. Follow the existing format

---

## 📜 License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=umitkacar/Segmentation---all-the-type&type=Date)](https://star-history.com/#umitkacar/Segmentation---all-the-type&Date)

---

## 🙏 Acknowledgments

This repository is maintained by the community and includes research from:
- Academic institutions worldwide
- Industry research labs (Meta AI, Google Research, OpenMMLab)
- Open-source contributors

**Special thanks to all researchers and developers who make their work publicly available!**

---

<div align="center">

### 💡 Found this useful? Give it a ⭐!

Made with ❤️ for the Computer Vision Community

[⬆ Back to Top](#-awesome-segmentation---all-types)

</div>
