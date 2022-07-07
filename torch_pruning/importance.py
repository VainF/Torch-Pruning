import abc
import torch
from . import functional


class Importance:
    def __init__(self, metric_fn) -> None:
        super().__init__(metric_fn)

    def __call__(self, plan):
        return self.metric_fn(plan[0])


class MagnitudeImportance(Importance):
    def __init__(self, p=2, local=False, reduction="sum"):
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
            # print(layer)
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w_dw = (layer.weight)[idxs]
                importance += torch.norm(torch.flatten(w_dw, 1), dim=1, p=self.p)
                if layer.bias is not None:
                    w_dw = (layer.bias)[idxs].view(-1, 1)
                    importance += torch.norm(w_dw, dim=1, p=self.p)
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w_dw = (layer.weight)[:, idxs].transpose(0, 1)
                w_dw = torch.flatten(w_dw, 1)
                if w_dw.shape[0] != importance.shape[0]:  # for conv-flatten-linear
                    if (
                        w_dw.shape[0] % importance.shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w_dw = w_dw.view(
                        importance.shape[0],
                        w_dw.shape[0] // importance.shape[0],
                        w_dw.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w_dw, 1), dim=1, p=self.p)
                importance += this_importance
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w_dw = (layer.weight)[idxs].view(-1, 1)
                    importance += torch.norm(w_dw, dim=1, p=self.p)
                    w_dw = (layer.bias)[idxs].view(-1, 1)
                    importance += torch.norm(w_dw, dim=1, p=self.p)
                    non_importance = False
            else:
                n_layers -= 1
            if self.local:
                break
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance
        elif self.reduction == "mean":
            return importance / n_layers


class SensitivityImportance(Importance):
    def __init__(self, local=False, reduction="sum") -> None:
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
                    functional.prune_conv_out_channel,
                ]:
                    w_dw = (layer.weight * layer.weight.grad)[idxs, :, :, :]
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
