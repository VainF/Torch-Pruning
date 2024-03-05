import torch
from scipy.spatial import distance
import numpy as np

# Define a sample tensor to simulate the effect of the codes
weight_torch = torch.randn(3, 4, 5, 3)

# Code 1
weight_vec1 = weight_torch.view(weight_torch.size()[0], -1)
norm1 = torch.norm(weight_vec1, 2, 1)

# Code 2
weight_vec2 = weight_torch.flatten(1)
norm2 = weight_vec2.abs().pow(2).sum(1).sqrt()  # Adding sqrt() to match the L2 norm calculation in Code 1

# Check if the results are the same
result_same = torch.allclose(norm1, norm2)

# Code 3
norm3 = weight_vec2.abs().pow(2)

local_imp_np = norm3.cpu().numpy()
# calculate euclidean distance
similar_matrix = distance.cdist(local_imp_np, local_imp_np, 'euclidean')

similar_matrix_2 = torch.cdist(norm3.unsqueeze(0), norm3.unsqueeze(0), p=2).squeeze(0)

similar_sum = np.sum(np.abs(similar_matrix), axis=0)

similar_sum_2 = torch.sum(torch.abs(similar_matrix_2), dim=0)

print(result_same, norm1, norm2, similar_matrix, similar_matrix_2, similar_sum, similar_sum_2)




# weight_torch = torch.randn(5)
# local_imp_np = weight_torch.cpu().numpy()
# # calculate euclidean distance
# similar_matrix = distance.cdist(local_imp_np, local_imp_np, 'euclidean')


weight_torch = torch.randn(5)
similar_matrix = torch.cdist(weight_torch.unsqueeze(1), weight_torch.unsqueeze(1), p=2).squeeze(0)
similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)



weight_torch_many_dim = torch.randn(5,4)
similar_matrix_2 = torch.cdist(weight_torch_many_dim.unsqueeze(0), weight_torch_many_dim.unsqueeze(0), p=2).squeeze(0)
similar_sum_2 = torch.sum(torch.abs(similar_matrix_2), dim=0)


print(similar_sum, similar_sum_2)