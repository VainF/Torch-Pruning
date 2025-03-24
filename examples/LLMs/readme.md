# Pruning Large Language Models

This example provides a minimal example of pruning large language models with a magnitude-based importance score. We use the `transformers` library to load the model and the `datasets` library to evaluate the Perplexity with `Wikitext2`. To be compatible with the Huggingface Transformers format, this example applies local pruning that unfiromly compress the model width. For finetuning, you may use other libraries such as [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). For more comprehensive examples of Gradient-based pruning or finetuning, please refer to [LLM-Pruner](https://github.com/horseee/LLM-Pruner).

This script has been tested with the following models:

*  [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
*  [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
*  [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
*  [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
*  [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
*  [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
*  [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
*  [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
*  [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

Other Llama-based models should work as well.


## 0. Requirements

```bash
pip install transformers datasets accelerate --upgrade

# if torch-pruning not installed
pip install torch-pruning --upgrade # >=1.5.0
```

## 1. Pruning

### Basic Usage
  
```bash
python prune_llm.py --model MODEL_CARD --pruning_ratio PRUNING_RATIO --max_seq_len MAX_SEQ_LEN --save_model SAVE_HF_MODEL
```

Arguments:
- `MODEL_CARD`: The model card of the model to be pruned, such as `meta-llama/Meta-Llama-3-8B`.
- `PRUNING_RATIO`: The ratio of the model width to be pruned, such as `0.5`. 
- `MAX_SEQ_LEN`: The maximum sequence length of the model, such as 4096. If not provided, the script will use the maximum sequence length of the model.
- `SAVE_HF_MODEL`: The path to save the pruned model. If not provided, the pruned model will not be saved. You can load a saved model using `AutoModelForCausalLM.from_pretrained`.


### :rocket: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

> [!NOTE]  
> The Qwen2.5-7B & DeepSeek-R1-Distill-Qwen-7B models have 28 heads with ``num_key_value_heads=4``. This limits the pruning ratio to be multiple of 28/4=7 such as [1/7, 2/7, 3/7, 4/7, 5/7, 6/7]. This is a hard constraint if you want to save and load the pruned model using Huggingface Transformers since HF only supports ``in_features==out_features`` in the ``q_proj`` and ``o_proj``. For other models, you need to follow the same rule to enable HF format compatibility. Otherwise, you need to save the model object directly with ``torch.save(model, PATH)``.
```bash
# 3/7 ~ 0.428571428, this script will craft a 2B model
python prune_llm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --pruning_ratio 0.428571428 --max_seq_len 4096 # --save_model pruned_model
```
<details>
<summary>Output:</summary>

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584) => (embed_tokens): Embedding(152064, 2048)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True) => (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True) => (k_proj): Linear(in_features=2048, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True) => (v_proj): Linear(in_features=2048, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False) => (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False) => (gate_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False) => (up_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False) => (down_proj): Linear(in_features=10824, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06) => (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False) => (lm_head): Linear(in_features=2048, out_features=152064, bias=False)
)

Qwen2Config {
  "_attn_implementation_autoset": true,
  "_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10824,
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 152064
}

num_params 2778732544
evaluating on wikitext2
Token indices sequence length is longer than the specified maximum sequence length for this model (2541001 > 16384). Running this sequence through the model will result in indexing errors
nsamples 73
sample 0
sample 50
wikitext perplexity 28358.30078125
```
</details>



### :rocket: Qwen/Qwen2.5-7B-Instruct



```bash
# 3/7 ~ 0.428571428, this script will craft a 2B model
python prune_llm.py --model Qwen/Qwen2.5-7B-Instruct --pruning_ratio 0.428571428 --max_seq_len 4096 # --save_model pruned_model
```

<details>
<summary>Output:</summary>

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584) => (embed_tokens): Embedding(152064, 2048)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True) => (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True) => (k_proj): Linear(in_features=2048, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True) => (v_proj): Linear(in_features=2048, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False) => (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False) => (gate_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False) => (up_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False) => (down_proj): Linear(in_features=10824, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06) => (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False) => (lm_head): Linear(in_features=2048, out_features=152064, bias=False)
)
Qwen2Config {
  "_attn_implementation_autoset": true,
  "_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10824,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

num_params 2778732544
evaluating on wikitext2
nsamples 73
sample 0
sample 50
wikitext perplexity 150926.78125
```

</details>



### :rocket: Qwen/Qwen2.5-7B
```bash
# 3/7 ~ 0.428571428, this script will craft a 2B model
python prune_llm.py --model Qwen/Qwen2.5-7B --pruning_ratio 0.428571428 --max_seq_len 4096 # --save_model pruned_model
```

<details>
<summary>Output:</summary>

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584) => (embed_tokens): Embedding(152064, 2048)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True) => (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True) => (k_proj): Linear(in_features=2048, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True) => (v_proj): Linear(in_features=2048, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False) => (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False) => (gate_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False) => (up_proj): Linear(in_features=2048, out_features=10824, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False) => (down_proj): Linear(in_features=10824, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06) => (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06) => (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False) => (lm_head): Linear(in_features=2048, out_features=152064, bias=False)
)

Qwen2Config {
  "_attn_implementation_autoset": true,
  "_name_or_path": "Qwen/Qwen2.5-7B",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10824,
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 152064
}

num_params 2778732544
evaluating on wikitext2
Token indices sequence length is longer than the specified maximum sequence length for this model (2541000 > 131072). Running this sequence through the model will result in indexing errors
nsamples 73
sample 0
sample 50
wikitext perplexity 307206.03125
```

</details>



### :rocket: Llama-3.1 8B

```bash
python prune_llm.py --model meta-llama/Llama-3.1-8B --pruning_ratio 0.5 --max_seq_len 4096 # --save_model pruned_model
```
<details>
<summary>Output:</summary>

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096) => (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False) => (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False) => (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False) => (v_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False) => (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False) => (gate_proj): Linear(in_features=2048, out_features=7168, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False) => (up_proj): Linear(in_features=2048, out_features=7168, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False) => (down_proj): Linear(in_features=7168, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05) => (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05) => (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05) => (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False) => (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
LlamaConfig {
  "_attn_implementation_autoset": true,
  "_name_or_path": "meta-llama/Llama-3.1-8B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 7168,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "vocab_size": 128256
}

num_params 2337409024
evaluating on wikitext2
nsamples 70
sample 0
sample 50
wikitext perplexity 576501.0
```

</details>


### :rocket: Llama-3.2 3B

```bash
python prune_llm.py --model meta-llama/Llama-3.2-3B --pruning_ratio 0.5 --max_seq_len 4096 # --save_model pruned_model
```
<details>
<summary>Output:</summary>

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072) => (embed_tokens): Embedding(128256, 1536)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=3072, out_features=3072, bias=False) => (q_proj): Linear(in_features=1536, out_features=3072, bias=False)
          (k_proj): Linear(in_features=3072, out_features=1024, bias=False) => (k_proj): Linear(in_features=1536, out_features=1024, bias=False)
          (v_proj): Linear(in_features=3072, out_features=1024, bias=False) => (v_proj): Linear(in_features=1536, out_features=1024, bias=False)
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False) => (o_proj): Linear(in_features=3072, out_features=1536, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=3072, out_features=8192, bias=False) => (gate_proj): Linear(in_features=1536, out_features=4096, bias=False)
          (up_proj): Linear(in_features=3072, out_features=8192, bias=False) => (up_proj): Linear(in_features=1536, out_features=4096, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False) => (down_proj): Linear(in_features=4096, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05) => (input_layernorm): LlamaRMSNorm((1536,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05) => (post_attention_layernorm): LlamaRMSNorm((1536,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((3072,), eps=1e-05) => (norm): LlamaRMSNorm((1536,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=128256, bias=False) => (lm_head): Linear(in_features=1536, out_features=128256, bias=False)
)

LlamaConfig {
  "_attn_implementation_autoset": true,
  "_name_or_path": "meta-llama/Llama-3.2-3B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 24,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "vocab_size": 128256
}

num_params 1274893824
evaluating on wikitext2
Token indices sequence length is longer than the specified maximum sequence length for this model (2458791 > 131072). Running this sequence through the model will result in indexing errors
nsamples 70
sample 0
sample 50
wikitext perplexity 58421.9375
```

</details>




### :rocket: microsoft/Phi-3-mini-4k-instruct

```bash
python prune_llm.py --model microsoft/Phi-3-mini-4k-instruct --pruning_ratio 0.5 # --save_model pruned_model
```


<details>
<summary>Output:</summary>

```
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000) => (embed_tokens): Embedding(32064, 1536, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False) => (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False) => (qkv_proj): Linear(in_features=1536, out_features=4608, bias=False)
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False) => (gate_up_proj): Linear(in_features=1536, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False) => (down_proj): Linear(in_features=4096, out_features=1536, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05) => (input_layernorm): Phi3RMSNorm((1536,), eps=1e-05)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05) => (post_attention_layernorm): Phi3RMSNorm((1536,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): Phi3RMSNorm((3072,), eps=1e-05) => (norm): Phi3RMSNorm((1536,), eps=1e-05)
    (rotary_emb): Phi3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False) => (lm_head): Linear(in_features=1536, out_features=32064, bias=False)
)

Phi3Config {
  "_attn_implementation_autoset": true,
  "_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
  "architectures": [
    "Phi3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "microsoft/Phi-3-mini-4k-instruct--configuration_phi3.Phi3Config",
    "AutoModelForCausalLM": "microsoft/Phi-3-mini-4k-instruct--modeling_phi3.Phi3ForCausalLM"
  },
  "bos_token_id": 1,
  "embd_pdrop": 0.0,
  "eos_token_id": 32000,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 4096,
  "model_type": "phi3",
  "num_attention_heads": 16,
  "num_hidden_layers": 32,
  "num_key_value_heads": 16,
  "original_max_position_embeddings": 4096,
  "pad_token_id": 32000,
  "resid_pdrop": 0.0,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sliding_window": 2047,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "vocab_size": 32064
}

num_params 1004570112
evaluating on wikitext2
Token indices sequence length is longer than the specified maximum sequence length for this model (2824490 > 4096). Running this sequence through the model will result in indexing errors
nsamples 83
sample 0
sample 50
wikitext perplexity 110115.0
```

</details>
