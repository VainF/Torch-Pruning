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



|| Vit-B/16 - Torchvision |	Vit-B/16 - HF |	ViT_B/32 - Torchvision | Vit-B/16 - 50% - Local Pruning |
| :-- | :--: | :--: | :--: | :--: |
| **#Params** | 86.57 M	| 86.56 M	|  	88.22 M | 22.05 M |
| **MACs** | 17.59 G	| 16.84 G	| 4.41 G |  4.24 G	| 
| **Top-1 Acc.** | 81.068	| 75.66	| 75.91 |  79.2	| 



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




