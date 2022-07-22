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
    def __init__(self, p=1, dep_aware=False, reduction="mean"):
        self.p = p
        self.dep_aware = dep_aware
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        global_imp = []
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                local_imp = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                global_imp.append(local_imp)

            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != global_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % global_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp[0].shape[0],
                        w.shape[0] // global_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                global_imp.append(local_imp)

            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    global_imp.append(local_imp)

            if not self.dep_aware:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]


class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, plan):
        _, idxs = plan[0]
        return torch.randn((len(idxs),))


class SensitivityImportance(Importance):
    def __init__(self, dep_aware=False, reduction="mean") -> None:
        self.dep_aware = dep_aware
        self.reduction = reduction

    def __call__(self, loss, plan):
        loss.backward()
        with torch.no_grad():
            global_imp = 0
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

                if self.dep_aware:
                    break
            if self.reduction == "sum":
                return importance
            elif self.reduction == "mean":
                return importance / n_layers


class SensitivityImportance(Importance):
    def __init__(self, p=1, dep_aware=False, reduction="mean"):
        self.p = p
        self.dep_aware = dep_aware
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        global_imp = []
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w_dw = (layer.weight * layer.weight.grad)[idxs]
                local_imp = torch.norm(torch.flatten(w_dw, 1), dim=1)
                if layer.bias:
                    w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                    local_imp += torch.norm(w_dw, dim=1)
                global_imp.append(local_imp)
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w_dw = (layer.weight * layer.weight.grad)[:, idxs].transpose(0, 1)
                local_imp = torch.norm(torch.flatten(w_dw, 1), dim=1)
                global_imp.append(local_imp)
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w_dw = (layer.weight * layer.weight.grad)[idxs].view(-1, 1)
                    local_imp = torch.norm(w_dw, dim=1)
                    w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                    local_imp += torch.norm(w_dw, dim=1)
                    global_imp.append(local_imp)

            if not self.dep_aware:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]


class HessianImportance(Importance):
    def __init__(self) -> None:
        pass


class BNScaleImportance(Importance):
    def __init__(self, dep_aware=False, reduction="mean"):
        self.dep_aware = dep_aware
        self.reduction = reduction

    def __call__(self, plan):
        global_imp = []
        for dep, idxs in plan:
            # Conv-BN
            module = dep.target.module
            if isinstance(module, nn.BatchNorm2d) and module.affine:
                imp = torch.abs(module.weight.data)
                global_imp.append(imp)
                if not self.dep_aware:
                    return imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]


class LAMPImportance(Importance):
    def __init__(self, p=1, dep_aware=False, reduction="mean"):
        self.p = p
        self.dep_aware = dep_aware
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        global_imp = []
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                local_imp = self.lamp(torch.norm(torch.flatten(w, 1), dim=1, p=self.p))
                global_imp.append(local_imp)

            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != global_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % global_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp[0].shape[0],
                        w.shape[0] // global_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))

                global_imp.append(local_imp)

            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))
                    global_imp.append(local_imp)

            if not self.dep_aware:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]

    def lamp(self, imp):
        argsort_idx = torch.argsort(imp, largest=False).tolist()[
            ::-1
        ]  # [7, 5, 2, 3, 1, ...]
        sorted_imp = imp[argsort_idx]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.arange(len(sorted_imp))[
            argsort_idx
        ].tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]


class GroupLassoImportance(Importance):
    def __init__(self, p=2, dep_aware=False, reduction="mean"):
        self.p = p
        self.dep_aware = dep_aware
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        global_imp = 0
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs].flatten(1)
                global_imp += w.pow(2).sum(1)
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if w.shape[0] != global_imp.shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % global_imp.shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp.shape[0],
                        w.shape[0] // global_imp.shape[0],
                        w.shape[1],
                    )
                    w = torch.flatten(w, 1)
                global_imp += w.pow(2).sum(1)
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs]
                    global_imp += w.pow(2)

            if self.dep_aware:
                break

        global_imp = global_imp.sqrt()
        return global_imp
