# Examples for Transformers

## Pruning ViT from [Timm](https://github.com/huggingface/pytorch-image-models)

### Data
Please prepare the ImageNet-1K dataset as follows and modify the data root in the script.
```
imagenet/
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
Some results:

|  | Vit-B/16 (Timm) |	ViT_B/32 (Timm) | Group L1 (Uniform) | Group Taylor (Uniform) | Group Taylor (Bottleneck) | Group Hessian (Uniform) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| **#Params** | 86.57 M		|  	88.22 M | 22.05 M | 22.05 M | 24.83 M | 22.05 M |
| **MACs** | 17.59 G		| 4.41 G |  4.61 G	| 4.61 G | 4.62 G | 4.61 G |
| **Acc @ Epoch 300** | 85.21	| 80.68  | 74.96 | 80.19 |   | 80.15   |
| **Latency (Bs=1, A5000)** | 5.21 ms <br> +- 0.05 ms	|  3.87 ms <br> +- 0.05 ms | 3.99 ms <br> +- 0.10 ms | 3.99 ms <br> +- 0.10 ms  |  3.87 ms <br> +- 0.14 ms  |  3.99 ms <br> +- 0.10 ms    |

*Notes:*
* Uniform - We apply the same pruning ratio to all layers.
* Bottleneck - We only prune the internal dimensions of Attention & FFN, leading to bottleneck structures.
* Please adjust the learning rate accordingly if the batch size and number of GPUs are changed. Refer to [this paper](https://arxiv.org/pdf/1706.02677.pdf) for more details about linear LR scaling with large mini-batch.

<div align="center">
  <img src="https://github.com/VainF/Torch-Pruning/assets/18592211/28de54dd-fd40-4889-abe0-00a49436a702" width="100%"></img>
</div>

### Which pruner should be used for ViT pruning?

* Pruning a Vision Transformer (HF Transformers) pre-trained on ImageNet-1K without fine-tuning.
<div align="center">
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/6f99aa90-259d-41e8-902a-35675a9c9d90" width="45%"></img>
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/11473499-d28a-434b-a8d6-1a53c4b3b7c0" width="45%"></img>
</div>


## Pruning Other Transformers

### ViT from [HF Transformers](https://huggingface.co/docs/transformers/index)
```bash
python prune_hf_vit.py
```
```
...
Base MACs: 16.848735 G, Pruned MACs: 4.241336 G
Base Params: 86.567656 M, Pruned Params: 22.050664 M
```

### Swin Transformers from [HF Transformers](https://huggingface.co/docs/transformers/index)
```bash
python prune_hf_swin.py
```
```
...
Base MACs: 4.350805 G, Pruned MACs: 1.438424 G
Base Params: 28.288354 M, Pruned Params: 9.462802 M
```

### Bert from [HF Transformers](https://huggingface.co/docs/transformers/index)
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



