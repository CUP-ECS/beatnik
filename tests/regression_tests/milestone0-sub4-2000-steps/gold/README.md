Python command run to generate gold files:
```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --no-dynamic-remesh --refine-every 0 \
  --source-quadrature vertex \
  --icosphere-subdivisions 4 --steps 2000 \
  --checkpoint-every-steps 25 --no-video --checkpoint-dir results4
```

## What is here

81 numbered checkpoints, `_step0000000` through `_step0002000` every 25 steps
(`--checkpoint-every-steps 25`), plus a `checkpoint_latest.npz` that duplicates
the last of them.

**The run did NOT stop early.** Step 2000 is present, every field is finite in
every file, and the checkpoint for step 2000 exists — the two observable
signatures of an early `stopping at step=... nonfinite ...` are both absent.
Risk M0-R2 (frozen mesh gives out) did not fire here, and neither did M0-R6 (`dt0` on a finer mesh).

`checkpoint_latest.npz` is **not** installed. It carries no `_step%07d.npz`
suffix, so it is inert to `goldForStep`
([Beatnik_Test_DirectSolve10Steps.cpp:263](../../Beatnik_Test_DirectSolve10Steps.cpp#L263)),
and shipping it would put a second copy of the largest file in every install.
M0-T1's install glob in [tests/CMakeLists.txt](../../../CMakeLists.txt)
therefore excludes it; the 81 numbered files and this `README.md` are what get
installed. Note that this `README.md` lives *inside* `gold/`, unlike
`direct-solve-10-steps/README.md`, which sits one level up — the install rules
handle both layouts rather than unifying them.

**Counts are constant** at `vertices (2562, 3)` / `faces (5120, 3)` in all 81 files, as
they must be with `--no-dynamic-remesh --refine-every 0`: any change would mean
adaptivity leaked in.

**Key set:** the same nine keys as `initial_conditions/gold.npz` —
`faces`, `initial_min_edge`, `initial_volume`, `potential`,
`remesh_material_position`, `state_model`, `step`, `time`, `vertices` —
verified identical in every one of the 81 files. `compare_output.py`'s
`FIELD_MAP` ([:111-125](../../compare_output.py#L111)) therefore needs no edit.

**Vertex pairing after 2000 steps of roll-up (risk M0-R4).** A self-compare of
the last numbered file against itself,

```
python tests/regression_tests/compare_output.py \
  tests/regression_tests/milestone0-sub4-2000-steps/gold/checkpoint_t00001p964304_step0002000.npz \
  tests/regression_tests/milestone0-sub4-2000-steps/gold/checkpoint_t00001p964304_step0002000.npz \
  --rtol 1e-12 --atol 1e-14
```

exits 0 and reports `matching (eps=1e-09): 2562/2562 unambiguous, ambiguous cpp=0 gold=0`,
with `max|e| = 0` on every field. The default `--match-eps 1e-9` still resolves
this mesh at step 2000; the pairing has not degraded.

## Per-step series

Computed from the committed `.npz` files by pure numpy over `vertices` and
`faces` — no MPI, no Beatnik binary. Definitions:

- **min quality** is `min_f 4*sqrt(3)*A_f / sum(l^2)` over the faces, the
  project's triangle-quality convention
  ([src/Beatnik_Params.hpp:220](../../../../src/Beatnik_Params.hpp#L220)).
- **`V/V0 - 1`** is the enclosed-volume drift, `V = (1/6) sum_f a.(b x c)` over
  `faces`, as T2d computes it. `V0` is the step-0 value of that same sum, which
  matches the file's own `initial_volume` to the last bit
  (`0.065308421062416244`).

**Headline numbers (level 4).** Quality falls, near-monotonically (the series
is not *strictly* monotone; it wobbles by a few percent sample to sample), from
`9.743550e-01` at step 0 to `1.242421e-01` at step 2000, which is also the
global minimum of the sampled series — it is still falling at the end of the
run rather than having bottomed out. It first drops below `0.5` at step 800,
below `0.3` at step 1000 and below `0.18` at step 1800, and **never reaches
`0.1`**. Volume drift grows monotonically to `+4.741414e-09` at step 2000.

**Comparison with M0-G1's minimum-quality series — the number M0-A1 needs.**
The frozen **level-3** mesh degrades *faster*, not slower:

| step | L3 min quality | L4 min quality | L3/L4 |
| ---: | ---: | ---: | ---: |
| 0 | 0.97495 | 0.97436 | 1.001 |
| 500 | 0.71388 | 0.89185 | 0.800 |
| 1000 | 0.19077 | 0.29999 | 0.636 |
| 1500 | 0.08022 | 0.22981 | 0.349 |
| 1800 | 0.04214 | 0.17303 | 0.244 |
| 2000 | 0.06304 | 0.12424 | 0.507 |

At step 2000 the level-4 minimum is **~2x** the level-3 one (`1.24e-01` vs
`6.30e-02`), and at the level-3 worst point (step 1700) the gap is wider still.
Level 4 also never goes below `0.1` where level 3 spends the last ~500 steps
there. So the finer frozen mesh is the *healthier* one over this horizon: the
level-3 set is the one closest to the M0-R2 regime, and M0-A1 cannot justify
picking level 3 as the primary member on mesh-health grounds.

Note the two runs do not cover the same physical time: 2000 steps of the
reference's adaptive dt reach `t = 1.998284` at level 3 but only `t = 1.964304`
at level 4, because the adaptive dt is relative to each run's own initial
minimum edge (`3.457079e-02` vs `1.729575e-02`). Risk M0-R6 (`dt0 = 0.003`
unstable on a finer mesh) did **not** fire: the level-4 run completed all 2000
steps with everything finite, so this is not a `--dt` decision for M0-A1.

| step | `time` | min quality | `V/V0 - 1` |
| ---: | ---: | ---: | ---: |
| 0 | 0.000000000000 | 9.743550e-01 | +0.000000e+00 |
| 25 | 0.074935439792 | 9.743335e-01 | +1.592704e-10 |
| 50 | 0.149469707414 | 9.742700e-01 | +3.123195e-10 |
| 75 | 0.223203897581 | 9.741684e-01 | +4.568357e-10 |
| 100 | 0.295758521898 | 9.740345e-01 | +5.918261e-10 |
| 125 | 0.366781856404 | 9.738767e-01 | +7.176699e-10 |
| 150 | 0.435957772145 | 9.737060e-01 | +8.358823e-10 |
| 175 | 0.503013028700 | 9.730506e-01 | +9.487178e-10 |
| 200 | 0.567724127741 | 9.721565e-01 | +1.058766e-09 |
| 225 | 0.629923981336 | 9.709268e-01 | +1.168645e-09 |
| 250 | 0.689508865823 | 9.691455e-01 | +1.280818e-09 |
| 275 | 0.746446437033 | 9.667744e-01 | +1.397556e-09 |
| 300 | 0.800786021579 | 9.636762e-01 | +1.521008e-09 |
| 325 | 0.852673024873 | 9.597117e-01 | +1.653443e-09 |
| 350 | 0.902271880500 | 9.547832e-01 | +1.796610e-09 |
| 375 | 0.949436052094 | 9.485920e-01 | +1.947508e-09 |
| 400 | 0.994437812567 | 9.402414e-01 | +2.107791e-09 |
| 425 | 1.037722027641 | 9.296326e-01 | +2.284259e-09 |
| 450 | 1.078886658157 | 9.168453e-01 | +2.472686e-09 |
| 475 | 1.117731347297 | 9.048733e-01 | +2.670172e-09 |
| 500 | 1.154592845294 | 8.918473e-01 | +2.880765e-09 |
| 525 | 1.189705331081 | 8.783924e-01 | +3.103733e-09 |
| 550 | 1.223175699615 | 8.628696e-01 | +3.332708e-09 |
| 575 | 1.255014308771 | 8.435163e-01 | +3.557018e-09 |
| 600 | 1.285189749407 | 8.191386e-01 | +3.765180e-09 |
| 625 | 1.313674471479 | 7.893769e-01 | +3.948562e-09 |
| 650 | 1.340468707044 | 7.546120e-01 | +4.103116e-09 |
| 675 | 1.365606445721 | 7.157236e-01 | +4.228971e-09 |
| 700 | 1.389151573701 | 6.738462e-01 | +4.328962e-09 |
| 725 | 1.411190390514 | 6.301834e-01 | +4.407145e-09 |
| 750 | 1.431823892379 | 5.858913e-01 | +4.467742e-09 |
| 775 | 1.451161252272 | 5.420239e-01 | +4.514576e-09 |
| 800 | 1.469314875644 | 4.995239e-01 | +4.550841e-09 |
| 825 | 1.486253709503 | 4.595799e-01 | +4.578200e-09 |
| 850 | 1.501868473328 | 4.234525e-01 | +4.597793e-09 |
| 875 | 1.516289703593 | 3.916775e-01 | +4.611910e-09 |
| 900 | 1.529648903239 | 3.646400e-01 | +4.622200e-09 |
| 925 | 1.542073373816 | 3.425986e-01 | +4.629819e-09 |
| 950 | 1.553685141119 | 3.256860e-01 | +4.635573e-09 |
| 975 | 1.564600401791 | 3.139016e-01 | +4.640023e-09 |
| 1000 | 1.574929320397 | 2.999890e-01 | +4.643558e-09 |
| 1025 | 1.584776044838 | 2.821708e-01 | +4.646453e-09 |
| 1050 | 1.594238837091 | 2.680696e-01 | +4.648904e-09 |
| 1075 | 1.603410236775 | 2.578133e-01 | +4.651051e-09 |
| 1100 | 1.612377188611 | 2.514750e-01 | +4.653001e-09 |
| 1125 | 1.621221073525 | 2.490411e-01 | +4.654835e-09 |
| 1150 | 1.629967271567 | 2.429245e-01 | +4.656581e-09 |
| 1175 | 1.638648522934 | 2.378359e-01 | +4.658268e-09 |
| 1200 | 1.647348480404 | 2.361040e-01 | +4.659965e-09 |
| 1225 | 1.656142283873 | 2.364426e-01 | +4.661732e-09 |
| 1250 | 1.665095635702 | 2.296638e-01 | +4.663627e-09 |
| 1275 | 1.674263531995 | 2.242722e-01 | +4.665707e-09 |
| 1300 | 1.683688718319 | 2.203423e-01 | +4.668030e-09 |
| 1325 | 1.693400027598 | 2.179170e-01 | +4.670645e-09 |
| 1350 | 1.703397442645 | 2.169781e-01 | +4.673578e-09 |
| 1375 | 1.713584567930 | 2.174052e-01 | +4.676731e-09 |
| 1400 | 1.723939765442 | 2.189625e-01 | +4.680083e-09 |
| 1425 | 1.734378072409 | 2.193159e-01 | +4.683518e-09 |
| 1450 | 1.744845619130 | 2.216626e-01 | +4.686953e-09 |
| 1475 | 1.755354955369 | 2.260057e-01 | +4.690390e-09 |
| 1500 | 1.765935938180 | 2.298089e-01 | +4.693852e-09 |
| 1525 | 1.776607237216 | 2.320857e-01 | +4.697351e-09 |
| 1550 | 1.787373082183 | 2.330105e-01 | +4.700882e-09 |
| 1575 | 1.798219912126 | 2.299267e-01 | +4.704421e-09 |
| 1600 | 1.809044137772 | 2.259583e-01 | +4.707829e-09 |
| 1625 | 1.819789804204 | 2.211256e-01 | +4.711052e-09 |
| 1650 | 1.830398248446 | 2.155162e-01 | +4.714035e-09 |
| 1675 | 1.840832412183 | 2.092594e-01 | +4.716764e-09 |
| 1700 | 1.851093231123 | 2.024920e-01 | +4.719271e-09 |
| 1725 | 1.861189078665 | 1.953559e-01 | +4.721590e-09 |
| 1750 | 1.871134342418 | 1.879868e-01 | +4.723758e-09 |
| 1775 | 1.880947956689 | 1.805083e-01 | +4.725811e-09 |
| 1800 | 1.890652005050 | 1.730293e-01 | +4.727782e-09 |
| 1825 | 1.900270459107 | 1.656451e-01 | +4.729699e-09 |
| 1850 | 1.909828080027 | 1.584393e-01 | +4.731591e-09 |
| 1875 | 1.919301558534 | 1.515219e-01 | +4.733446e-09 |
| 1900 | 1.928596980009 | 1.450373e-01 | +4.735192e-09 |
| 1925 | 1.937725113922 | 1.390370e-01 | +4.736845e-09 |
| 1950 | 1.946704624855 | 1.335557e-01 | +4.738421e-09 |
| 1975 | 1.955556683923 | 1.286184e-01 | +4.739938e-09 |
| 2000 | 1.964304144657 | 1.242421e-01 | +4.741414e-09 |
