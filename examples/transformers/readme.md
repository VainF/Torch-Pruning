# Transformers

This example demonstrate the minimal code to prune Transformers, including Vision Transformers (ViT), Swin Transformers, and BERT. If you need a more comprehensive example for pruning and finetuning, please refer to the [codebase for Isomorphic Pruning](https://github.com/VainF/Isomorphic-Pruning), where detailed instructions and pre-pruned models are available.


## Pruning ViT-ImageNet-21K-ft-1K from [Timm](https://github.com/huggingface/pytorch-image-models)

### Data
Please prepare the ImageNet-1K dataset as follows and modify the data root in the script.
```
./data/imagenet/
  train/
    n01440764/
      n01440764_10026.JPEG
      ...
    n01773157/
    n02051845/
    ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
    n01773157/
    n02051845/
```

### Pruning
```bash
bash scripts/prune_timm_vit_b_16_taylor_uniform.sh
```

```
...
----------------------------------------
Summary:
Base MACs: 17.59 G, Pruned MACs: 4.61 G
Base Params: 86.57 M, Pruned Params: 22.05 M
Base Loss: 0.6516, Pruned Loss: 7.2412
Base Accuracy: 0.8521, Pruned Accuracy: 0.0016
Saving the pruned model to output/pruned/vit_base_patch16_224_pruned_taylor_uniform.pth...
```

### Finetuning
```bash
bash scripts/finetune_timm_vit_b_16_taylor_uniform.sh
```
Pruning results for ImageNet-21K-ft-1K (Timm):

|  | ViT-B/16 (Timm) |	ViT_B/32 (Timm) | Group L2 (Uniform) | Group Taylor (Uniform) | Group Taylor (Bottleneck) | Group Hessian (Uniform) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| **#Params** | 86.57 M		|  	88.22 M | 22.05 M | 22.05 M | 24.83 M | 22.05 M |
| **MACs** | 17.59 G		| 4.41 G |  4.61 G	| 4.61 G | 4.62 G | 4.61 G |
| **Acc @ Epoch 300** | 85.21	| 80.68  | 78.11 | 80.19 | 80.06 | 80.15   |
| **Latency (Bs=1, A5000)** | 5.21 ms <br> +- 0.05 ms	|  3.87 ms <br> +- 0.05 ms | 3.99 ms <br> +- 0.10 ms | 3.99 ms <br> +- 0.10 ms  |  3.87 ms <br> +- 0.14 ms  |  3.99 ms <br> +- 0.10 ms    |
| **Checkpoints** | - | - | [ckpt](https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_l2_uniform.pth) | [ckpt](https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_taylor_uniform.pth) | [ckpt](https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_taylor_bottleneck.pth) | [ckpt](https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_hessian_uniform.pth) |

*Notes:*
* Uniform - We apply the same pruning ratio to all layers.
* Bottleneck - We only prune the internal dimensions of Attention & FFN, leading to bottleneck structures.
* Please adjust the learning rate accordingly if the batch size and number of GPUs are changed. Refer to [this paper](https://arxiv.org/pdf/1706.02677.pdf) for more details about linear LR scaling with large mini-batch.

<div align="center">
  <img src="https://github.com/VainF/Torch-Pruning/assets/18592211/2537a09f-6d62-4879-8300-6ceec722ebe9" width="100%"></img>
</div>

### Which pruner should be used for ViT pruning?

In short, `tp.importance.GroupTaylorImportance` + `tp.pruner.BasePruner` is a good choice for ViT pruning.

* Prune a Vision Transformer (ImageNet-1K) from HF Transformers without fine-tuning.
<div align="center">
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/6f99aa90-259d-41e8-902a-35675a9c9d90" width="45%"></img>
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/11473499-d28a-434b-a8d6-1a53c4b3b7c0" width="45%"></img>
</div>

* Prune a Vision Transformer (ImageNet-21K-ft-1K) from timm without finetuning
<div align="center">
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/8726aaf3-129a-4ff6-855e-c73573a8d3e4" width="45%"></img>
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/5ff31ebe-3d0e-417b-8020-68afaa19dc65" width="45%"></img>
</div>

### Latency

* Download our finetuned models
```bash
mkdir pretrained
cd pretrained
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_taylor_uniform.pth
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_taylor_bottleneck.pth
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_l2_uniform.pth
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.2.5/vit_b_16_pruning_hessian_uniform.pth
```

* Measure the latency of the pruned models
```bash
python measure_latency.py --model pretrained/vit_b_16_pruning_taylor_uniform.pth
```

### Pruning attention heads

```bash
python prune_timm_vit.py --prune_num_heads --head_pruning_ratio 0.5 
```

```bash
...
Head #0
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #1
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #2
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #3
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #4
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #5
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #6
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #7
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #8
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #9
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #10
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64

Head #11
[Before Pruning] Num Heads: 12, Head Dim: 64 =>
[After Pruning] Num Heads: 6, Head Dim: 64
...
```


## Pruning ViT-ImageNet-1K from [HF Transformers](https://huggingface.co/docs/transformers/index)

### Pruning
```bash
bash scripts/prune_hf_vit_b_16_taylor_uniform.sh  
```
```
...
----------------------------------------
Summary:
Base MACs: 16.85 G, Pruned MACs: 4.24 G
Base Params: 86.57 M, Pruned Params: 22.05 M
Base Loss: 0.9717, Pruned Loss: 7.0871
Base Accuracy: 0.7566, Pruned Accuracy: 0.0015
Saving the pruned model to output/pruned/hf_vit_base_patch16_224_pruned_taylor_uniform.pth...
```

### Finetuning
```bash
bash scripts/finetune_hf_vit_b_16_taylor_uniform.sh
```

Pruning results for ImageNet-1K (HF Transformers):

| | ViT-B/16 <br> (HF) | ViT-B/16 <br> (Torchvision) |	ViT_B/32 <br> (Torchvision) | Group L1 <br> (Uniform) | Group Taylor <br> (Uniform) | Group Taylor <br> (Bottleneck) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| **#Params** |  86.56 M	| 86.57 M	|  	88.22 M | 22.05 M | 22.05 M | 22.8 M |
| **MACs** | 17.59 G	| 17.59 G	|  4.41 G |  4.61 G	| 4.61 G | 4.23 G |
| **Acc @ Ep 300** | 75.66 | 81.068	|  75.91 |  79.20	| 79.61 | 79.11 |

## Pruning Swin Transformers from [HF Transformers](https://huggingface.co/docs/transformers/index)
```bash
python prune_hf_swin.py
```
```
...
Base MACs: 4.350805 G, Pruned MACs: 1.438424 G
Base Params: 28.288354 M, Pruned Params: 9.462802 M
```

## Pruning Bert from [HF Transformers](https://huggingface.co/docs/transformers/index)
```bash
python prune_hf_bert.py
```
```
...
Base MACs: 680.150784 M, Pruned MACs: 170.206464 M
Base Params: 109.482240 M, Pruned Params: 33.507840 M
```

## Ackowledgement

The training code was adpated from [Torchvision Reference](https://github.com/pytorch/vision/tree/main/references/classification).



