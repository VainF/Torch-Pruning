# Examples for HuggingFace Transformers

Our post-training scripts will be released.

## Pruning HF ViT
```bash
python prune_vit.py
```
```
...
Base MACs: 16.848735 G, Pruned MACs: 4.241336 G
Base Params: 86.567656 M, Pruned Params: 22.050664 M
```

Some results:

| | Vit-B/16 - Torchvision |	Vit-B/16 - HF |	ViT_B/32 - Torchvision | Group L1 (Uniform) | Group Taylor (Uniform) | Group Taylor (Bottleneck) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| **#Params** | 86.57 M	| 86.56 M	|  	88.22 M | 22.05 M | 22.05 M | 22.8 M |
| **MACs** | 17.59 G	| 16.84 G	| 4.41 G |  4.24 G	| 4.24G | 4.30G |
| **Acc @ Epoch 300** | 81.068	| 75.66	| 75.91 |  79.20	| 79.61 | 79.11 |
| **Acc @ Epoch 50** | -	| -	| - |  69.24	| 71.93 | 71.54 |

* Uniform - The same pruning ratio for all layers.
* Bottleneck - Only prune the internal dimensions of Attention & FFN.

<div align="center">
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/24de19ff-60aa-4402-94e3-527670ffb55e" width="80%"></img>
</div>

## Pruning HF Swin
```bash
python prune_swin.py
```
```
...
Base MACs: 4.350805 G, Pruned MACs: 1.438424 G
Base Params: 28.288354 M, Pruned Params: 9.462802 M
```

## Pruning HF Bert
```bash
python prune_bert.py
```
```
...
Base MACs: 680.150784 M, Pruned MACs: 170.206464 M
Base Params: 109.482240 M, Pruned Params: 33.507840 M
```




