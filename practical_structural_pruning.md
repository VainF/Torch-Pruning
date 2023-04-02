# Practical Structural Pruning

Welcome to the resource list for structural pruning, which is continually being updated. If you have any useful resources, papers, or repositories related to structural pruning and would like to share them with other users, please feel free to open a new pull request or issue.

<br>

### 0. Projects/Papers that use Torch-Pruning
| Ttle | URL | 
|:--    | :--: |
| Pruning-for-YOLOX | [project](https://github.com/YINYIPENG-EN/Pruning_for_YOLOX) |
| mobile-yolov5-pruning-distillation | [project](https://github.com/Syencil/mobile-yolov5-pruning-distillation) |
| YOLOX-pruning | [project](https://github.com/lizexu123/YOLOX-pruning) |
| Vision Model Compression | [project](https://github.com/ashutoshsameer/vision-model-compression) |
| Ladder Side-Tuning (LST) | [project](https://github.com/ylsung/Ladder-Side-Tuning/tree/c09bf673e6b5f9e10bac40aa144b820cf6fbdec6), [NeurIPS'22](https://openreview.net/forum?id=isPnnaTZaP5) |
| LAASP | [project](https://github.com/ghimiredhikura/LAASP-FilterPruning/tree/63e84c07031f85ef155954b02f9cb9b56ed8c6d8) |
| Magnitude and Similarity based Variable Rate Filter Pruning for Efficient Convolution Neural Networks | [project](https://github.com/ghimiredhikura/MSVFP-FilterPruning) |
| Resnet50-pruning | [projects](https://github.com/wxy1234567/Resnet50-pruning) |
| Performance-aware Approximation of Global Channel Pruning for Multitask CNNs | [project](https://github.com/HankYe/PAGCP), [TPAMI'23](https://arxiv.org/abs/1909.08174) |
| yolov5-pruning | [project](https://github.com/jylink/yolov5-pruning/tree/9b039785efa1a6f6323cceb24815fa06fbf01b89) |
| Model-Pruning | [project](https://github.com/ChaeyeonSeo/Models-pruning) |
| Melanoma detection CNN using PyTorch | [project](https://github.com/98munozpatricia/Melanoma/tree/7d9899d0a45bb2939718ffbe6c69319155dac763) |
| KamalEngine | [project](https://github.com/zju-vipa/KamalEngine/tree/0276eb062595d52472090fbcbcedcd76db8cfd44) |
| Modified-Yolov4Tiny-RaspberryPi | [project](https://github.com/benedettaliberatori/Modified-Yolov4Tiny-RaspberryPi) |
| model_pruning | [project](https://github.com/liugt123/model_prune) |

<br>

### 1. Benchmark & Survey & Awesome List

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning]()  |   Huan Wang  | Preprint | [pytorch](https://github.com/MingSun-Tse/Why-the-State-of-Pruning-so-Confusing)   |
| [Structured Pruning for Deep Convolutional Neural Networks](https://arxiv.org/abs/2303.00566) | Yang He | Preprint | [Awesome List](https://github.com/he-y/Awesome-Pruning) |

<br>

### 2. General

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) | Zhonghui You | NeurIPS'19 | [pytorch](https://github.com/youzhonghui/gate-decorator-pruning) |
| [Neural Pruning via Growing Regularization](https://arxiv.org/abs/2012.09243) | Huan Wang | ICLR'21 | [pytorch](https://github.com/mingsun-tse/regularization-pruning) |
| [Group Fisher Pruning for Practical Network Compression](https://arxiv.org/abs/2108.00708) |  Liyang Liu | ICML'21 | [pytorch](https://github.com/jshilong/FisherPruning) | 
| [Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) | Tianyi Chen | NeurIPS'21 |  [pytorch](https://github.com/tianyic/only_train_once) |
| [OTOV2: AUTOMATIC, GENERIC, USER-FRIENDLY](https://openreview.net/pdf?id=7ynoX1ojPMt) | Tianyi Chen | ICLR'23 | [pytorch](https://github.com/tianyic/only_train_once) |
| [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | Gongfan Fang | CVPR'23 | [pytorch](https://github.com/VainF/Torch-Pruning) |

<br>

### 3. YOLO

| Type | Title | Authors | Venue | Code | 
|:--:   |:--    |:--:  |:--:    |:--: |
| YOLOv5 | [Performance-aware Approximation of Global Channel Pruning for Multitask CNNs](https://arxiv.org/abs/1909.08174) | Hancheng Ye | TPAMI'23 | [pytorch](https://github.com/HankYe/PAGCP) |
| YOLOv7 | [Torch-Pruning: benchmarks/prunability/yolov7](https://github.com/VainF/Torch-Pruning/tree/master/benchmarks/prunability) | - | - | [pytorch](https://github.com/VainF/Torch-Pruning/blob/master/benchmarks/prunability/yolov7_train_pruned.py) |


<br>

### 4. GANs

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936) | Muyang Li | CVPR'20 | [pytorch](https://github.com/mit-han-lab/gan-compression-dynamic) |

<br>

### 5. Vision Transformers

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [Vision Transformer Pruning](https://arxiv.org/abs/2104.08500) | Mingjian Zhu | Preprint | N/A |
| [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://openreview.net/forum?id=w5DacXWzQ-Q) | Zheng Chuanyang | NeurIPS'22 | N/A |
| [Width & Depth Pruning for Vision Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/20222) | Fang Yu | AAAI'22 | N/A |
| [Chasing Sparsity in Vision Transformers: An End-to-End Exploration](https://arxiv.org/abs/2106.04533) | Tianlong Chen | NeurIPS'21 | [pytorch](https://github.com/VITA-Group/SViTE) |
| [CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction](https://arxiv.org/abs/2203.04570) | Zhuoran Song | Preprint  | [pytorch](https://github.com/ok858ok/CP-ViT) |
| [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | Gongfan Fang | CVPR'23 | [pytorch](https://github.com/VainF/Torch-Pruning) |

<br>

### 6. NLP

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/abs/2204.00408) | Mengzhou Xia | ACL'22 | [pytorch](https://github.com/princeton-nlp/CoFiPruning) | 

<br>

### 7. GNNs

| Title | Authors | Venue | Code | 
|:--    |:--:  |:--:    |:--: |
| [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | Gongfan Fang | CVPR'23 | [pytorch](https://github.com/VainF/Torch-Pruning) |


<br>


### 8. Misc

| Title | Source |
|:--    | :-- |
| [Issue - Does the tool support to prune the yolov5/v7/v8?](https://github.com/VainF/Torch-Pruning/issues/100) | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| [Issue - How can we transfer schedular and optimizer from original model to pruned model](https://github.com/VainF/Torch-Pruning/issues/120) | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| [Discussion - How to get the index of the pruned channel(s) when using High-level Pruners?](https://github.com/VainF/Torch-Pruning/discussions/116#discussioncomment-5426178) | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| [Issue - How to get all groups of a network](https://github.com/VainF/Torch-Pruning/issues/109) |[Torch-Pruning](https://github.com/VainF/Torch-Pruning)| 
| [Issue - 论文中图3(b)的剪枝方案是怎么推出的 (in Chinese)](https://github.com/VainF/Torch-Pruning/issues/115) | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) |
| [Model Compression - the Pruning Approaches](https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf) | Lecture Notes (Princeton) |
| [CVPR 2023 \| DepGraph：任意架构的结构化剪枝，CNN、Transformer、GNN等都适用！(In Chinese)](https://mp.weixin.qq.com/s/Ux-yuaKokGGhUkfF491ZYw) | CVer |
