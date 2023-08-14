from transformers import AutoTokenizer, BertModel
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention
import torch_pruning as tp

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
#print(model)
hf_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
example_inputs = {'input_ids': hf_inputs['input_ids'], 'token_type_ids': hf_inputs['token_type_ids'], 'attention_mask': hf_inputs['attention_mask']}

#outputs = model(**example_inputs)
#last_hidden_states = outputs.last_hidden_state

imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
channel_groups = {}

# All heads should be pruned simultaneously, so we group channels by head.
for m in model.modules():
    if isinstance(m, BertSelfAttention):
        channel_groups[m.query] = m.num_attention_heads
        channel_groups[m.key] = m.num_attention_heads
        channel_groups[m.value] = m.num_attention_heads

pruner = tp.pruner.MagnitudePruner(
    model, 
    example_inputs, 
    global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=1, # the number of iterations to achieve target sparsity
    ch_sparsity=0.5,
    channel_groups=channel_groups,
    output_transform=lambda out: out.pooler_output.sum(),
    ignored_layers=[model.pooler],
)

for g in pruner.step(interactive=True):
    print(g)
    g.prune()

# Modify the attention head size and all head size aftering pruning
for m in model.modules():
    if isinstance(m, BertSelfAttention):
        m.attention_head_size = m.query.out_features // m.num_attention_heads
        m.all_head_size = m.query.out_features

print(model)
test_output = model(**example_inputs)
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %f M, Pruned MACs: %f M"%(base_macs/1e6, pruned_macs/1e6))
print("Base Params: %f M, Pruned Params: %f M"%(base_params/1e6, pruned_params/1e6))