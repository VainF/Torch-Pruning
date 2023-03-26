
def linear_scheduler(ch_sparsity_dict, steps):
    return [((i) / float(steps)) * ch_sparsity_dict for i in range(steps+1)]