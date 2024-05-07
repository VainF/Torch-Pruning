import torch

class SoftTargetCrossEntropyNoneSoftmax(torch.nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropyNoneSoftmax, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * torch.nn.functional.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
    
    