from .metapruner import MetaPruner
import torch

class SaliencyPruner(MetaPruner):

    def reset(self):
        self.accum_imp = {}
        self.groups = list(super().get_all_groups())

    def update_importance(self, loss):
        loss.backward()
        for group in self.groups:
            imp = self.importance(group)
            if group not in self.accum_imp:
                self.accum_imp[group] = imp
            else:
                self.accum_imp[group]+=imp

    def estimate_importance(self, group):
        return self.accum_imp[group]

    def prune_local(self):
        if self.current_step == self.pruning_steps:
            return
        for group in self.groups:
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                imp = self.estimate_importance(group)
                current_channels = self.DG.get_out_channels(module) #utils.count_prunable_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )
                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned].tolist()
                
                # Prune the score vector
                keep_idxs = imp_argsort[n_pruned:].tolist()
                keep_idxs.sort()
                self.accum_imp[group] = self.accum_imp[group][keep_idxs]

                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs)
                if self.DG.check_pruning_group(group):
                    group.exec()

    def prune_global(self):
        if self.current_step == self.pruning_steps:
            return

        global_importance = []
        for group in self.groups:
            imp = self.estimate_importance(group)
            global_importance.append((group, imp))

        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self._total_channels *
            (1 - target_sparsity)
        )
        #print(len(imp), self._total_channels, n_pruned, target_sparsity)
        if n_pruned<0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        # prune by threhold
        thres = topk_imp[-1]
        for group, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1).tolist()
            # Prune the score vector
            keep_idxs = (imp > thres).nonzero().view(-1).tolist()
            keep_idxs.sort()
            self.accum_imp[group] = self.accum_imp[group][keep_idxs]

            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self._check_sparsity(group) and self.DG.check_pruning_group(group):
                group.exec()