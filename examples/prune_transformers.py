import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch_pruning as tp

# Load BertModel from https://github.com/huggingface/transformers
from transformers import BertModel
model = BertModel.from_pretrained('/data/home/mxy/pretrained_models/bert-base') #'bert-base-uncased'

# Build dependency graph
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs = torch.randint(0, 30522, (32, 128)), pruning_dim = -1) # Note to set pruning_dim to -1 to prune BertModel on hidden_states.

# get a pruning plan by pruning from word embedding
strategy = tp.strategy.L1Strategy() 
pruning_idxs = strategy(model.embeddings.word_embeddings.weight.T, amount=0.1) # Transpose the weight matrix to [num_embeddings, embedding_dim]
pruning_plan = DG.get_pruning_plan( model.embeddings.word_embeddings, tp.prune_embedding, idxs=pruning_idxs )
print(pruning_plan)

# execute the plan (prune the model) and save the model
pruning_plan.exec()
torch.save(model, 'model.pth')