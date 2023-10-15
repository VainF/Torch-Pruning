
def linear_scheduler(pruning_ratio_dict, steps):
    return [((i) / float(steps)) * pruning_ratio_dict for i in range(steps+1)]