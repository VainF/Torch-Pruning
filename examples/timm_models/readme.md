# Pruning Models from Timm


## 0. List all models in Timm

```bash
python prune_timm_models.py --list_models
```

Output:
```
['bat_resnext26ts', 'beit_base_patch16_224', 'beit_base_patch16_384', 'beit_large_patch16_224', 'beit_large_patch16_384', 'beit_large_patch16_512', 'beitv2_base_patch16_224', ...]
```

## 1. Pruning

Some models might requires additional modifications to enable pruning. For example, we need to reimplement the forward function of `vit` to relax the constraint in structure. Refer to [examples/transformers](../transformers/) for more details.

```bash
python prune_timm_models.py --model convnext_xxlarge --ch_sparsity 0.5 # --global_pruning
```

#### Outputs:
```
========Before pruning========
...
  (norm_pre): Identity()
  (head): NormMlpClassifierHead(
    (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Identity())
    (norm): LayerNorm2d((3072,), eps=1e-05, elementwise_affine=True)
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (pre_logits): Identity()
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Linear(in_features=3072, out_features=1000, bias=True)
  )
)


========After pruning========
...
  (norm_pre): Identity()
  (head): NormMlpClassifierHead(
    (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Identity())
    (norm): LayerNorm2d((1536,), eps=1e-05, elementwise_affine=True)
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (pre_logits): Identity()
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Linear(in_features=1536, out_features=1000, bias=True)
  )
)
MACs: 197.9920 G => 49.7716 G
Params: 846.4710 M => 213.2587 M
```
