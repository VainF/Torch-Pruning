# Compatibility branch for LoRa-Transfer-Pruning

This branch is based on upstream tag `v1.6.1` (e80127d7c6935c319a7dd6719b8a72e3c1fcbff9).
The package version is `1.6.1+glazkov.1`. Consumers should pin the Git commit,
not the branch tip, so installation does not change when this branch advances.

The patch addresses [upstream issue #532](https://github.com/VainF/Torch-Pruning/issues/532):
PyTorch can expose negative `torch.cat` dimensions as unsigned 64-bit integers.
Decode the signed dimension and normalize it using output rank when available,
then apply the existing concat mapping policy. This prevents spurious offsets
in attention/RoPE graphs while preserving channel concatenation for 2-D Linear
outputs. `MAX_VALID_DIM` remains at the upstream value of 100.

Run the focused regression suite with:

```sh
python -m pytest -q tests/test_unsigned_concat_dim.py
```

The tests cover actual autograd metadata, the fallback without output metadata,
and structural pruning through channel concatenation. The downstream project
also checks its complete CPU suite, including attention dependency graphs.
This patch does not redesign the handling of split, unbind, or arbitrary axes.

Once an upstream release includes an equivalent fix and passes these tests and
the downstream suite, switch consumers to that release and retire this branch.
