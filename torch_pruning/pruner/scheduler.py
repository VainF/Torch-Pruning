
def linear_scheduler(layer_ch_sparsity, steps):
    return [((i + 1) / float(steps)) * layer_ch_sparsity for i in range(steps)]
