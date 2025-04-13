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

> [!NOTE] 
> Some models might require additional modifications to enable pruning. For example, we need to reimplement the forward function of `vit` to relax the constraint in structure. Refer to [examples/transformers/prune_timm_vit.py](../transformers/prune_timm_vit.py) for more details.



### Example 1: Vision Transformer Pruning

```bash
python prune_timm_models.py --model vit_base_patch32_224 --pruning_ratio 0.5
```

#### Outputs:
```
Pruning vit_base_patch32_224...

VisionTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32)) => (proj): Conv2d(3, 384, kernel_size=(32, 32), stride=(32, 32))
    (norm): Identity()
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (patch_drop): Identity()
  (norm_pre): Identity()
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True) => (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True) => (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (q_norm): Identity()
        (k_norm): Identity()
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True) => (proj): Linear(in_features=384, out_features=384, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (ls1): Identity()
      (drop_path1): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True) => (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True) => (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True) => (fc2): Linear(in_features=1536, out_features=384, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (ls2): Identity()
      (drop_path2): Identity()
    )
...
    (11): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True) => (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True) => (qkv): Linear(in_features=384, out_features=1152, bias=True)
        (q_norm): Identity()
        (k_norm): Identity()
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True) => (proj): Linear(in_features=384, out_features=384, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (ls1): Identity()
      (drop_path1): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True) => (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True) => (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True) => (fc2): Linear(in_features=1536, out_features=384, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (ls2): Identity()
      (drop_path2): Identity()
    )
  )
  (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True) => (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
  (fc_norm): Identity()
  (head_drop): Dropout(p=0.0, inplace=False)
  (head): Linear(in_features=768, out_features=1000, bias=True) => (head): Linear(in_features=384, out_features=1000, bias=True)
)

MACs: 4.4157 G => 1.1463 G
Params: 88.2242 M => 22.8790 M
```


### Example 2: ConvNext Pruning
```bash
python prune_timm_models.py --model convnext_xxlarge --pruning_ratio 0.5 # --global_pruning
```


For ConvNext, you see a warning message like this:
```
Unwrapped parameters detected: ['stages.2.blocks.5.gamma', ... 'stages.2.blocks.26.gamma'].
 Torch-Pruning will prune the last non-singleton dimension of these parameters. If you wish to change this behavior, please provide an unwrapped_parameters argument
```
This is not an error. It is caused by the `stages.xx.blocks.xx.gamma` parameters that do not belong to any standard layers such as nn.Linear or nn.Conv2d. Torch-Pruning will prune the last non-singleton dimension of these parameters by default. 


#### Outputs:
```
ConvNeXt(
  (stem): Sequential(
    (0): Conv2d(3, 384, kernel_size=(4, 4), stride=(4, 4)) => (0): Conv2d(3, 192, kernel_size=(4, 4), stride=(4, 4))
    (1): LayerNorm2d((384,), eps=1e-05, elementwise_affine=True) => (1): LayerNorm2d((192,), eps=1e-05, elementwise_affine=True)
  )
  (stages): Sequential(
    (0): ConvNeXtStage(
      (downsample): Identity()
      (blocks): Sequential(
        (0): ConvNeXtBlock(
          (conv_dw): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384) => (conv_dw): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True) => (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True) => (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (norm): Identity()
            (fc2): Linear(in_features=1536, out_features=384, bias=True) => (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (shortcut): Identity()
          (drop_path): Identity()
        )
...
        (2): ConvNeXtBlock(
          (conv_dw): Conv2d(3072, 3072, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=3072) => (conv_dw): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
          (norm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True) => (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=3072, out_features=12288, bias=True) => (fc1): Linear(in_features=1536, out_features=6144, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (norm): Identity()
            (fc2): Linear(in_features=12288, out_features=3072, bias=True) => (fc2): Linear(in_features=6144, out_features=1536, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (shortcut): Identity()
          (drop_path): Identity()
        )
      )
    )
  )
  (norm_pre): Identity()
  (head): NormMlpClassifierHead(
    (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
    (norm): LayerNorm2d((3072,), eps=1e-05, elementwise_affine=True) => (norm): LayerNorm2d((1536,), eps=1e-05, elementwise_affine=True)
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (pre_logits): Identity()
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Linear(in_features=3072, out_features=1000, bias=True) => (fc): Linear(in_features=1536, out_features=1000, bias=True)
  )
)

MACs: 197.9920 G => 49.7716 G
Params: 846.4710 M => 213.2587 M
```

