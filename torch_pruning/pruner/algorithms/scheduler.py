from typing import List

def linear_scheduler(pruning_ratio: float, steps: int) -> List[float]:
    return [((i) / float(steps)) * pruning_ratio for i in range(steps + 1)]
