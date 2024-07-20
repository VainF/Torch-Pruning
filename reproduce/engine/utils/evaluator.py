from tqdm import tqdm
import torch.nn.functional as F 
import torch
from . import metrics

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)

def segmentation_evaluator(dataloader, num_classes, ignore_idx=255):
    cm = metrics.ConfusionMatrix(num_classes, ignore_idx=ignore_idx)
    metric = metrics.MetricCompose({
        'mIoU': metrics.mIoU(cm),
        'Acc': metrics.Accuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)