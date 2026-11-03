Python command run to generate gold files (M1-G2, `tasks/milestone1.md`), from
the reference repository root:
```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --dynamic-remesh --isotropic-cleanup \
  --source-quadrature vertex \
  --steps 600 --checkpoint-every-steps 10 --no-video --checkpoint-dir results
```

That is the task entry's command. The run differed from it only in
`--checkpoint-dir`, which named `/p/lustre5/stewartj/beatnik/M1-G2` in place of
`results`; the files were then copied here and verified byte for byte. Every
other option is at its `parse_args` default. The interpreter was
`/usr/tce/bin/python3` (Python 3.13.2, numpy 2.1.2, scipy 1.15.3), launched as
`flux run -N 1 -n 1` inside
[scripts/tuolumne/m1g2_gold.flux](../../../scripts/tuolumne/m1g2_gold.flux),
job `f3hyB1NchZz7`. Its log, `beatnik_m1g2_gold.f3hyB1NchZz7.log` in the
Beatnik repository root, has the full per-step stdout (a progress line at step 1
and every 60 steps, with `minQ`, `minEdge`, `minGap` and the remesh counters) and
the checks below. Generation took 707 s, 1.18 s/step.

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
finite in every file. The compare-depth ceiling is therefore the full 600 steps.
Final `time = 0.55793412422180344`.

**Key set:** the same nine keys as `initial_conditions/gold.npz` —
`faces`, `initial_min_edge`, `initial_volume`, `potential`,
`remesh_material_position`, `state_model`, `step`, `time`, `vertices` — in every
file, so `compare_output.py`'s `FIELD_MAP` needs no edit.

**Step 0** is bitwise identical to `initial_conditions/gold.npz`: `np.array_equal`
holds on all nine keys, and `compare_output.py <step0> initial_conditions/gold.npz
--rtol 1e-12 --atol 1e-14` exits 0 with `162/162 unambiguous`.

**Vertex pairing at step 600.** A self-compare of the last numbered file,

```
python tests/regression_tests/compare_output.py \
  tests/regression_tests/milestone1-bubble/gold/checkpoint_t00000p557934_step0000600.npz \
  tests/regression_tests/milestone1-bubble/gold/checkpoint_t00000p557934_step0000600.npz \
  --rtol 1e-12 --atol 1e-14
```

exits 0 and reports `matching (eps=1e-09): 538/538 unambiguous, ambiguous cpp=0
gold=0`.

## Vertex and face counts

The counts are **not** constant. Step 0 is the level-2 icosphere, `162`/`320`.
The first remesh pass refines it heavily: the step-1 progress line already reads
`V=462 F=920` with `remesh_events=390`, and the step-10 checkpoint holds the
peak, `547`/`1090`. The mesh then coarsens slightly and settles: `539`/`1074` by
step 60, and it stays between `538`/`1072` and `540`/`1076` from there to step
600, ending at `538`/`1072`. The cumulative `remesh_events` counter reads 631 at
step 60 and 639 at step 600, so nearly all mesh editing happens in the first 60
steps. `refine_events=0` and `flips=0` throughout. Every row satisfies
`V = F/2 + 2`, the closed genus-0 surface relation.

## Per-checkpoint series

Read from the committed `.npz` files: `step`, `time` (to 6 decimals), and the
row counts of `vertices` and `faces`.

| step | time | V | F |
| --- | --- | --- | --- |
| 0 | 0.000000 | 162 | 320 |
| 10 | 0.011017 | 547 | 1090 |
| 20 | 0.019425 | 544 | 1084 |
| 30 | 0.027647 | 542 | 1080 |
| 40 | 0.035732 | 542 | 1080 |
| 50 | 0.043733 | 542 | 1080 |
| 60 | 0.052516 | 539 | 1074 |
| 70 | 0.062211 | 539 | 1074 |
| 80 | 0.071830 | 539 | 1074 |
| 90 | 0.081439 | 539 | 1074 |
| 100 | 0.091043 | 539 | 1074 |
| 110 | 0.100643 | 539 | 1074 |
| 120 | 0.110238 | 539 | 1074 |
| 130 | 0.119828 | 539 | 1074 |
| 140 | 0.129411 | 539 | 1074 |
| 150 | 0.138987 | 539 | 1074 |
| 160 | 0.148557 | 539 | 1074 |
| 170 | 0.158119 | 539 | 1074 |
| 180 | 0.167673 | 539 | 1074 |
| 190 | 0.177219 | 539 | 1074 |
| 200 | 0.186757 | 539 | 1074 |
| 210 | 0.196285 | 539 | 1074 |
| 220 | 0.205803 | 539 | 1074 |
| 230 | 0.215311 | 539 | 1074 |
| 240 | 0.224808 | 539 | 1074 |
| 250 | 0.234297 | 538 | 1072 |
| 260 | 0.243799 | 538 | 1072 |
| 270 | 0.253316 | 538 | 1072 |
| 280 | 0.262837 | 538 | 1072 |
| 290 | 0.272356 | 538 | 1072 |
| 300 | 0.281868 | 538 | 1072 |
| 310 | 0.291371 | 538 | 1072 |
| 320 | 0.300862 | 539 | 1074 |
| 330 | 0.310341 | 538 | 1072 |
| 340 | 0.319792 | 538 | 1072 |
| 350 | 0.329205 | 538 | 1072 |
| 360 | 0.338589 | 538 | 1072 |
| 370 | 0.347951 | 538 | 1072 |
| 380 | 0.357291 | 538 | 1072 |
| 390 | 0.366610 | 538 | 1072 |
| 400 | 0.375907 | 540 | 1076 |
| 410 | 0.385183 | 540 | 1076 |
| 420 | 0.394438 | 540 | 1076 |
| 430 | 0.403672 | 540 | 1076 |
| 440 | 0.412883 | 540 | 1076 |
| 450 | 0.422073 | 540 | 1076 |
| 460 | 0.431240 | 540 | 1076 |
| 470 | 0.440383 | 540 | 1076 |
| 480 | 0.449502 | 540 | 1076 |
| 490 | 0.458598 | 540 | 1076 |
| 500 | 0.467668 | 540 | 1076 |
| 510 | 0.476714 | 540 | 1076 |
| 520 | 0.485735 | 540 | 1076 |
| 530 | 0.494731 | 540 | 1076 |
| 540 | 0.503701 | 540 | 1076 |
| 550 | 0.512644 | 540 | 1076 |
| 560 | 0.521561 | 540 | 1076 |
| 570 | 0.530452 | 540 | 1076 |
| 580 | 0.539608 | 539 | 1074 |
| 590 | 0.548847 | 538 | 1072 |
| 600 | 0.557934 | 538 | 1072 |
