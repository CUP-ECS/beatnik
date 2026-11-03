Python command run to generate gold files (M1-G1, `tasks/milestone1.md`), from
the reference repository root:
```
python examples/run_adaptive_mesh_bubble.py --steps 600 \
  --source-quadrature vertex --br-approximation direct \
  --no-dynamic-remesh --refine-every 0 \
  --checkpoint-every-steps 10 --no-video --checkpoint-dir results
```

That is the task entry's command. The run differed from it only in
`--checkpoint-dir`, which named `/p/lustre5/stewartj/beatnik/M1-G1` in place of
`results`; the files were then copied here and verified byte for byte. The
interpreter was `/usr/tce/bin/python3` (Python 3.13.2, numpy 2.1.2, scipy
1.15.3), launched as `flux run -N 1 -n 1` inside
[scripts/tuolumne/m1g1_gold.flux](../../../scripts/tuolumne/m1g1_gold.flux),
job `f3hyB1FBHAym`. Its log, `beatnik_m1g1_gold.f3hyB1FBHAym.log` in the
Beatnik repository root, has the per-step stdout and the checks below.

## Reference version

Reference repository `zmodel3d-amr` at `ec7d7bf4aaf9e84c6ac999dce3193ca1650677f1`,
with one uncommitted change in `examples/run_adaptive_mesh_bubble.py`
(`:221`): the `--source-quadrature` default is `vertex` where the commit has
`face`. The command passes `--source-quadrature vertex` explicitly, so the change
does not affect this set. The `zmodel3d/` package was clean.

## What is here

61 numbered checkpoints in `gold/`, `_step0000000` through `_step0000600` every
10 steps, plus a `checkpoint_latest.npz` that duplicates the last of them
(verified byte-identical with `cmp`). `checkpoint_latest.npz` carries no
`_step%07d.npz` suffix, so it is inert to `goldForStep` and does not count toward
the 61.

**The run did NOT stop early.** The last progress line is `step=  600`, the log
has `saved final finite checkpoint` and no `stopping at step=`, and every field is
finite in every file. Final `time = 1.3141321331131375`.

**Counts are constant** at `vertices (162, 3)` / `faces (320, 3)` in all 61
files: the level-2 icosphere default, with connectivity held fixed by
`--no-dynamic-remesh --refine-every 0`. The progress lines report
`refine_events=0 flips=0 remesh_events=0` throughout.

**Key set:** the same nine keys as `initial_conditions/gold.npz` —
`faces`, `initial_min_edge`, `initial_volume`, `potential`,
`remesh_material_position`, `state_model`, `step`, `time`, `vertices` — in every
file, so `compare_output.py`'s `FIELD_MAP` needs no edit.

**Step 0** is bitwise identical to `initial_conditions/gold.npz`: `np.array_equal`
holds on all nine keys, and `compare_output.py <step0> initial_conditions/gold.npz
--rtol 1e-12 --atol 1e-14` exits 0.

**Vertex pairing at step 600.** A self-compare of the last numbered file,

```
python tests/regression_tests/compare_output.py \
  tests/regression_tests/direct-solve-600-steps/gold/checkpoint_t00001p314132_step0000600.npz \
  tests/regression_tests/direct-solve-600-steps/gold/checkpoint_t00001p314132_step0000600.npz \
  --rtol 1e-12 --atol 1e-14
```

exits 0 and reports `matching (eps=1e-09): 162/162 unambiguous, ambiguous cpp=0
gold=0`.
