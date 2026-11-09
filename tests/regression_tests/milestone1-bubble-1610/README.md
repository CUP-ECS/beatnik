# Milestone 1 — the steps 0–1610 rising-bubble gold set

Python command run to generate the gold files (M1-G3j, `tasks/milestone1.md`),
from the staged copy of the reference repository:
```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --dynamic-remesh --isotropic-cleanup \
  --source-quadrature vertex \
  --steps 5000 --checkpoint-every-steps 10 --no-video \
  --checkpoint-dir /p/lustre5/stewartj/beatnik/M1-G3j/out
```

That is M1-G2's command ([../milestone1-bubble/README.md](../milestone1-bubble/README.md))
with `--steps 5000` in place of `--steps 600` and `--checkpoint-dir` pointing at
the Lustre output directory. Every other option is at its `parse_args` default.
The interpreter is `/usr/tce/bin/python3` (Python 3.13.2, numpy 2.1.2, scipy
1.15.3), launched as `flux run -N 1 -n 1` with `cd` into the stage
`/p/lustre5/stewartj/beatnik/M1-G3j/stage/` by
[scripts/tuolumne/m1g3_gold.flux](../../../scripts/tuolumne/m1g3_gold.flux),
job `f3iZLsmfWYw1` on `tuolumne1116`. Its log,
`beatnik_m1g3_gold.f3iZLsmfWYw1.log` in the Beatnik repository root, has the
full per-step stdout and the in-job checks. Generation wall: **1782 s** for
1625 steps (1.10 s/step).

## Reference version

From `stage/COMMIT`, written when the stage was copied from the live reference
checkout (2026-11-06T14:23:32-08:00): reference repository `zmodel3d-amr` at
`ec7d7bf4aaf9e84c6ac999dce3193ca1650677f1`, with one uncommitted change in
`examples/run_adaptive_mesh_bubble.py`: the `--source-quadrature` default is
`vertex` where the commit has `face`. The command passes `--source-quadrature
vertex` explicitly, so the change does not affect this set. The `zmodel3d/`
package was clean. The stage was taken at Beatnik `2a38355`.

## Where the files are

The `.npz` files are **not** in this repository. They are on Lustre in
`/p/lustre5/stewartj/beatnik/M1-G3j/out/`, which holds 164 numbered
checkpoints, steps 0, 10, …, 1620 and 1624, plus a `checkpoint_latest.npz`
byte-identical to the step-1624 file. The milestone test consumes only the
statistics of steps 0–1610, compiled in by M1-A1j.

## The run stopped by itself at step 1625

The reference does not reach step 5000. Its last lines are

```
stopping at step=1625 t=1.2540: nonfinite mesh-RHS state
saved final finite checkpoint /p/lustre5/stewartj/beatnik/M1-G3j/out/checkpoint_t00001p251003_step0001624.npz
```

preceded by numpy overflow warnings in `zmodel3d/mesh_solver.py`. The driver
exits 0 after the stop.

## The gold set is steps 0–1610

**The gold set is the 162 checkpoints of steps 0, 10, …, 1610**, final `time =
1.2381221993625053` (1.238122). Step 1610 is the last checkpoint before the
mesh surges. The counts rise smoothly, from `538`/`1072` (V/F) at step 1000 to
`610`/`1216` at step 1610, then jump:

| step | time | V | F |
| --- | --- | --- | --- |
| 1620 | 1.241753 | 793 | 1590 |
| 1624 | 1.251003 | 1993 | 4002 |

Those two files are **excluded** from the gold set, and stay on Lustre
untouched. The step-1624 file is degenerate: its self-compare at `--rtol 1e-12
--atol 1e-14` exits 1 with `matching (eps=1e-09): 1/1993 unambiguous, ambiguous
cpp=1992 gold=1992`, so no vertex pairing is possible against it.

## Checks

From the generation job's in-job checks (`beatnik_m1g3_gold.f3iZLsmfWYw1.log`):

- **Step 0** is `np.array_equal` to `initial_conditions/gold.npz` on all nine
  keys, and `compare_output.py` of the two at `--rtol 1e-12 --atol 1e-14` exits
  0 with `162/162 unambiguous`.
- **Step 600** is `np.array_equal` on all nine keys to M1-G2's
  `milestone1-bubble/gold/checkpoint_t00000p557934_step0000600.npz`. The
  reference is serial and deterministic, so this ties the staged reference to
  the one M1-G2 ran.
- Every numeric field of every one of the 164 files is finite. The only
  failures the job reports are the step-1624 self-compare above.

From the verification job `f3j6vKt4EQS3`,
[scripts/tuolumne/m1g3_check.flux](../../../scripts/tuolumne/m1g3_check.flux),
log `beatnik_m1g3_check.f3j6vKt4EQS3.log` in the Beatnik repository root
(`ALL CHECKS PASS`, rc 0):

- A self-compare of the step-1610 file with the staged `compare_output.py`,

  ```
  python compare_output.py \
    /p/lustre5/stewartj/beatnik/M1-G3j/out/checkpoint_t00001p238122_step0001610.npz \
    /p/lustre5/stewartj/beatnik/M1-G3j/out/checkpoint_t00001p238122_step0001610.npz \
    --rtol 1e-12 --atol 1e-14
  ```

  exits 0 and reports `matching (eps=1e-09): 610/610 unambiguous, ambiguous
  cpp=0 gold=0`. **Ambiguity count: 0.**
- Steps 0..1610 are exactly 162 numbered files, steps 0, 10, …, 1610, each with
  the nine keys `faces`, `initial_min_edge`, `initial_volume`, `potential`,
  `remesh_material_position`, `state_model`, `step`, `time`, `vertices`, the
  stored `step` equal to the file's, and every numeric field finite.

## Vertex and face counts

Steps 0–600 have M1-G2's step / `time` / V / F rows exactly: `162`/`320` at step 0, the peak
`547`/`1090` at step 10, and `538`–`540`/`1072`–`1076` from step 60 to 600.
The mesh then coarsens slightly, to a minimum of `534`/`1064` at steps 870–890,
returns to `538`/`1072` by step 1000, and grows steadily from step 1100 on:
`541`/`1078` at step 1150, `576`/`1148` at step 1400 and
`610`/`1216` at step 1610, the largest mesh in the gold set. Every row satisfies
`V = F/2 + 2`, the closed genus-0 surface relation.

## Per-checkpoint series

Read from the `.npz` files by the verification job: `step`, `time` (to 6
decimals), and the row counts of `vertices` and `faces`. The generation job's
table for the same steps is identical.

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
| 610 | 0.566942 | 538 | 1072 |
| 620 | 0.575890 | 538 | 1072 |
| 630 | 0.584782 | 538 | 1072 |
| 640 | 0.593623 | 537 | 1070 |
| 650 | 0.602621 | 536 | 1068 |
| 660 | 0.611647 | 536 | 1068 |
| 670 | 0.620596 | 536 | 1068 |
| 680 | 0.629417 | 535 | 1066 |
| 690 | 0.637067 | 535 | 1066 |
| 700 | 0.644664 | 535 | 1066 |
| 710 | 0.652243 | 535 | 1066 |
| 720 | 0.659799 | 535 | 1066 |
| 730 | 0.667329 | 535 | 1066 |
| 740 | 0.674833 | 535 | 1066 |
| 750 | 0.682309 | 535 | 1066 |
| 760 | 0.689758 | 535 | 1066 |
| 770 | 0.697180 | 535 | 1066 |
| 780 | 0.704573 | 535 | 1066 |
| 790 | 0.711938 | 535 | 1066 |
| 800 | 0.719275 | 535 | 1066 |
| 810 | 0.726584 | 535 | 1066 |
| 820 | 0.733864 | 535 | 1066 |
| 830 | 0.741116 | 535 | 1066 |
| 840 | 0.748340 | 535 | 1066 |
| 850 | 0.755536 | 535 | 1066 |
| 860 | 0.762703 | 535 | 1066 |
| 870 | 0.769853 | 534 | 1064 |
| 880 | 0.777015 | 534 | 1064 |
| 890 | 0.784142 | 534 | 1064 |
| 900 | 0.791236 | 535 | 1066 |
| 910 | 0.798300 | 535 | 1066 |
| 920 | 0.805333 | 535 | 1066 |
| 930 | 0.812337 | 535 | 1066 |
| 940 | 0.819313 | 535 | 1066 |
| 950 | 0.826261 | 535 | 1066 |
| 960 | 0.833180 | 535 | 1066 |
| 970 | 0.840071 | 535 | 1066 |
| 980 | 0.846935 | 535 | 1066 |
| 990 | 0.853771 | 537 | 1070 |
| 1000 | 0.860580 | 538 | 1072 |
| 1010 | 0.867363 | 538 | 1072 |
| 1020 | 0.874122 | 538 | 1072 |
| 1030 | 0.880858 | 538 | 1072 |
| 1040 | 0.887570 | 538 | 1072 |
| 1050 | 0.894259 | 538 | 1072 |
| 1060 | 0.900925 | 538 | 1072 |
| 1070 | 0.907568 | 538 | 1072 |
| 1080 | 0.914186 | 538 | 1072 |
| 1090 | 0.920782 | 538 | 1072 |
| 1100 | 0.927354 | 539 | 1074 |
| 1110 | 0.933903 | 539 | 1074 |
| 1120 | 0.940429 | 540 | 1076 |
| 1130 | 0.946935 | 540 | 1076 |
| 1140 | 0.953428 | 540 | 1076 |
| 1150 | 0.959911 | 541 | 1078 |
| 1160 | 0.966380 | 541 | 1078 |
| 1170 | 0.972831 | 541 | 1078 |
| 1180 | 0.979265 | 541 | 1078 |
| 1190 | 0.985681 | 541 | 1078 |
| 1200 | 0.992079 | 546 | 1088 |
| 1210 | 0.998459 | 547 | 1090 |
| 1220 | 1.004821 | 547 | 1090 |
| 1230 | 1.011167 | 547 | 1090 |
| 1240 | 1.017497 | 547 | 1090 |
| 1250 | 1.023811 | 547 | 1090 |
| 1260 | 1.030109 | 549 | 1094 |
| 1270 | 1.036393 | 554 | 1104 |
| 1280 | 1.042662 | 557 | 1110 |
| 1290 | 1.048917 | 560 | 1116 |
| 1300 | 1.055162 | 561 | 1118 |
| 1310 | 1.061391 | 562 | 1120 |
| 1320 | 1.067594 | 562 | 1120 |
| 1330 | 1.073770 | 566 | 1128 |
| 1340 | 1.079923 | 566 | 1128 |
| 1350 | 1.086058 | 568 | 1132 |
| 1360 | 1.092178 | 569 | 1134 |
| 1370 | 1.098283 | 574 | 1144 |
| 1380 | 1.104366 | 576 | 1148 |
| 1390 | 1.110416 | 576 | 1148 |
| 1400 | 1.116400 | 576 | 1148 |
| 1410 | 1.122349 | 579 | 1154 |
| 1420 | 1.128273 | 580 | 1156 |
| 1430 | 1.134167 | 581 | 1158 |
| 1440 | 1.140041 | 581 | 1158 |
| 1450 | 1.145899 | 581 | 1158 |
| 1460 | 1.151747 | 581 | 1158 |
| 1470 | 1.157586 | 583 | 1162 |
| 1480 | 1.163417 | 584 | 1164 |
| 1490 | 1.169244 | 587 | 1170 |
| 1500 | 1.175065 | 591 | 1178 |
| 1510 | 1.180898 | 595 | 1186 |
| 1520 | 1.186739 | 595 | 1186 |
| 1530 | 1.192582 | 596 | 1188 |
| 1540 | 1.198408 | 596 | 1188 |
| 1550 | 1.204209 | 596 | 1188 |
| 1560 | 1.209995 | 601 | 1198 |
| 1570 | 1.215773 | 602 | 1200 |
| 1580 | 1.221534 | 603 | 1202 |
| 1590 | 1.227127 | 606 | 1208 |
| 1600 | 1.232645 | 609 | 1214 |
| 1610 | 1.238122 | 610 | 1216 |
