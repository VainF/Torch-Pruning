import abc
import torch
import torch.nn as nn
from . import functional
import random

class Importance:
    pass

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

class MagnitudeImportance(Importance):
    def __init__(self, p=1, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance_mat = []
        non_importance = True
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(this_importance)
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance_mat[0].shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance_mat[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance_mat[0].shape[0],
                        w.shape[0] // importance_mat[0].shape[0],
                        w.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(this_importance)
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    this_importance = torch.norm(w, dim=1, p=self.p)
                    importance_mat.append(this_importance)
            if self.local:
                break
        importance_mat = torch.stack(importance_mat, dim=0)
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]
        elif self.reduction == "prod":
            return importance_mat.prod(dim=0)


class RandomImportance(Importance):

    @torch.no_grad()
    def __call__(self, plan):
        _, idxs = plan[0]
        return torch.randn( (len(idxs), ) )


class SensitivityImportance(Importance):
    def __init__(self, local=False, reduction="mean") -> None:
        self.local = local
        self.reduction = reduction

    def __call__(self, loss, plan):
        loss.backward()
        with torch.no_grad():
            importance = 0
            n_layers = 0
            for dep, idxs in plan:
                layer = dep.target.module
                prune_fn = dep.handler
                n_layers += 1
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_in_channel,
                ]:
                    w_dw = (layer.weight * layer.weight.grad)[idxs]
                    importance += torch.norm(torch.flatten(w_dw, 1), dim=1)
                    if layer.bias:
                        w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    w_dw = (layer.weight * layer.weight.grad)[:, idxs].transpose(0, 1)
                    importance += torch.norm(torch.flatten(w_dw, 1), dim=1)
                elif prune_fn == functional.prune_batchnorm:
                    if layer.affine:
                        w_dw = (layer.weight * layer.weight.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                        w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                else:
                    n_layers -= 1

                if self.local:
                    break
            if self.reduction == "sum":
                return importance
            elif self.reduction == "mean":
                return importance / n_layers
            


class HessianImportance(Importance):
    def __init__(self) -> None:
        pass

class BNScaleImportance(Importance):
    def __init__(self, group_level=False, reduction='mean'):
        self.group_level = group_level
        self.reduction = reduction

    def __call__(self, plan):
        importance_mat = []

        for dep, idxs in plan:
            # Conv-BN
            module = dep.target.module
            if isinstance(module, nn.BatchNorm2d) and module.affine:
                imp = torch.abs(module.weight.data)
                importance_mat.append( imp )
                if not self.group_level:
                    return imp
        importance_mat = torch.stack(importance_mat, dim=0)
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]


        
class StrcuturalImportance(Importance):
    def __init__(self, p=1, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance_mat = []
        non_importance = True
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(rescale(this_importance))
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance_mat[0].shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance_mat[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance_mat[0].shape[0],
                        w.shape[0] // importance_mat[0].shape[0],
                        w.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(rescale(this_importance))
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    this_importance = torch.norm(w, dim=1, p=self.p)
                    importance_mat.append(rescale(this_importance))
            if self.local:
                break
        importance_mat = torch.stack(importance_mat, dim=0)
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]
        elif self.reduction == "prod":
            return importance_mat.prod(dim=0)


class LAMPImportance(Importance):
    def __init__(self, p=2, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance = 0
        n_layers = 0
        non_importance = True
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            n_layers += 1
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                this_importance = rescale(this_importance)
                importance+=this_importance
                #if layer.bias is not None:
                #    w = (layer.bias)[idxs].view(-1, 1)
                #    importance += torch.norm(w, dim=1, p=self.p)
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance.shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance.shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance.shape[0],
                        w.shape[0] // importance.shape[0],
                        w.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                this_importance = rescale(this_importance)
                importance += this_importance
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                continue
                if layer.affine is not None:
                    #scale = layer.weight / sqrt_rv
                    #bias = layer.bias - rm / sqrt_rv * layer.weight
                    w = (layer.weight)[idxs].view(-1, 1)
                    importance += rescale(torch.norm(w, dim=1, p=self.p))
                    #w = (bias)[idxs].view(-1, 1)
                    #importance *= torch.norm(w, dim=1, p=self.p)
            #        non_importance = False
            else:
                n_layers -= 1
            if self.local:
                break
        argsort_idx = torch.argsort(importance).tolist()[::-1] # [7, 5, 2, 3, 1, ...]
        sorted_importance = importance[argsort_idx]
        cumsum_importance = torch.cumsum(sorted_importance, dim=0 )
        sorted_importance = sorted_importance / cumsum_importance 
        inversed_idx = torch.arange(len(sorted_importance))[argsort_idx].tolist() # [0, 1, 2, 3, ..., ]
        importance = sorted_importance[inversed_idx]
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance
        elif self.reduction == "mean":
            return importance / n_layers
