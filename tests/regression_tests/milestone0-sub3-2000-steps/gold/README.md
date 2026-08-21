Python command run to generate gold files:
```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --no-dynamic-remesh --refine-every 0 \
  --source-quadrature vertex \
  --icosphere-subdivisions 3 --steps 2000 \
  --checkpoint-every-steps 25 --no-video --checkpoint-dir results3
```

## What is here

81 numbered checkpoints, `_step0000000` through `_step0002000` every 25 steps
(`--checkpoint-every-steps 25`), plus a `checkpoint_latest.npz` that duplicates
the last of them.

**The run did NOT stop early.** Step 2000 is present, every field is finite in
every file, and the checkpoint for step 2000 exists — the two observable
signatures of an early `stopping at step=... nonfinite ...` are both absent.
Risk M0-R2 (frozen mesh gives out) did not fire here.

`checkpoint_latest.npz` is **not** installed. It carries no `_step%07d.npz`
suffix, so it is inert to `goldForStep`
([Beatnik_Test_DirectSolve10Steps.cpp:263](../../Beatnik_Test_DirectSolve10Steps.cpp#L263)),
and shipping it would put a second copy of the largest file in every install.
M0-T1's install glob in [tests/CMakeLists.txt](../../../CMakeLists.txt)
therefore excludes it; the 81 numbered files and this `README.md` are what get
installed. Note that this `README.md` lives *inside* `gold/`, unlike
`direct-solve-10-steps/README.md`, which sits one level up — the install rules
handle both layouts rather than unifying them.

**Counts are constant** at `vertices (642, 3)` / `faces (1280, 3)` in all 81 files, as
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
  tests/regression_tests/milestone0-sub3-2000-steps/gold/checkpoint_t00001p998284_step0002000.npz \
  tests/regression_tests/milestone0-sub3-2000-steps/gold/checkpoint_t00001p998284_step0002000.npz \
  --rtol 1e-12 --atol 1e-14
```

exits 0 and reports `matching (eps=1e-09): 642/642 unambiguous, ambiguous cpp=0 gold=0`,
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
  (`0.064886575267079027`).

**Headline numbers (level 3).** Quality falls, near-monotonically, from
`9.749529e-01` at step 0 to a global minimum of `3.826563e-02` at **step 1700**,
then recovers slightly to `6.303626e-02` at step 2000. It first drops below
`0.5` at step 700, below `0.3` at step 850, below `0.18` (the reference's
`--remesh-min-quality` default) at step 1050, and below `0.1` at step 1475.
Volume drift grows monotonically and sub-linearly to `+3.352894e-09` at step
2000 — of order `1e-9`, above T2d's `kVolumeDriftAbsCap = 1e-9`, exactly as
risk M0-R3 predicted; a milestone-0 test must re-derive that cap and must not
reuse T2d's.

| step | `time` | min quality | `V/V0 - 1` |
| ---: | ---: | ---: | ---: |
| 0 | 0.000000000000 | 9.749529e-01 | +0.000000e+00 |
| 25 | 0.074936383845 | 9.749147e-01 | +1.543745e-10 |
| 50 | 0.149477433402 | 9.748033e-01 | +3.027640e-10 |
| 75 | 0.223229912315 | 9.746278e-01 | +4.429577e-10 |
| 100 | 0.295819464641 | 9.744025e-01 | +5.740000e-10 |
| 125 | 0.366898752272 | 9.741472e-01 | +6.962519e-10 |
| 150 | 0.436155125275 | 9.738871e-01 | +8.111736e-10 |
| 175 | 0.503317820526 | 9.735630e-01 | +9.209582e-10 |
| 200 | 0.568164798530 | 9.715467e-01 | +1.028162e-09 |
| 225 | 0.630529480460 | 9.688312e-01 | +1.135410e-09 |
| 250 | 0.690307854700 | 9.652339e-01 | +1.245236e-09 |
| 275 | 0.747466713944 | 9.605318e-01 | +1.360045e-09 |
| 300 | 0.802007218281 | 9.539163e-01 | +1.481770e-09 |
| 325 | 0.853800238374 | 9.404320e-01 | +1.610082e-09 |
| 350 | 0.902945447228 | 9.204137e-01 | +1.745553e-09 |
| 375 | 0.948883021520 | 8.935743e-01 | +1.879855e-09 |
| 400 | 0.991050927083 | 8.635379e-01 | +2.002722e-09 |
| 425 | 1.029737323382 | 8.270333e-01 | +2.114200e-09 |
| 450 | 1.065364653654 | 7.866210e-01 | +2.216618e-09 |
| 475 | 1.098409437235 | 7.473157e-01 | +2.312817e-09 |
| 500 | 1.129315327391 | 7.138781e-01 | +2.404718e-09 |
| 525 | 1.158439074003 | 6.902219e-01 | +2.492934e-09 |
| 550 | 1.186054175243 | 6.768854e-01 | +2.577337e-09 |
| 575 | 1.212408285794 | 6.546391e-01 | +2.658060e-09 |
| 600 | 1.237805877414 | 6.410514e-01 | +2.736492e-09 |
| 625 | 1.261773458359 | 6.369415e-01 | +2.805258e-09 |
| 650 | 1.283596955381 | 5.920805e-01 | +2.856430e-09 |
| 675 | 1.303493457733 | 5.415878e-01 | +2.894137e-09 |
| 700 | 1.321744599192 | 4.960724e-01 | +2.922289e-09 |
| 725 | 1.338672351168 | 4.565072e-01 | +2.944042e-09 |
| 750 | 1.354619330743 | 4.166098e-01 | +2.961779e-09 |
| 775 | 1.369934120683 | 3.724791e-01 | +2.977281e-09 |
| 800 | 1.384433802219 | 3.331591e-01 | +2.990027e-09 |
| 825 | 1.397978783112 | 3.000709e-01 | +2.999864e-09 |
| 850 | 1.410854622758 | 2.733391e-01 | +3.007963e-09 |
| 875 | 1.423336498217 | 2.531758e-01 | +3.015144e-09 |
| 900 | 1.435688717584 | 2.398985e-01 | +3.022035e-09 |
| 925 | 1.448165134273 | 2.338328e-01 | +3.029192e-09 |
| 950 | 1.461009364851 | 2.171690e-01 | +3.037192e-09 |
| 975 | 1.474453820263 | 2.024252e-01 | +3.046720e-09 |
| 1000 | 1.488716548046 | 1.907700e-01 | +3.058622e-09 |
| 1025 | 1.503994810952 | 1.826938e-01 | +3.073973e-09 |
| 1050 | 1.520454374982 | 1.786545e-01 | +3.094114e-09 |
| 1075 | 1.537831169905 | 1.788171e-01 | +3.118438e-09 |
| 1100 | 1.554140476978 | 1.821462e-01 | +3.136958e-09 |
| 1125 | 1.569348009639 | 1.871643e-01 | +3.150729e-09 |
| 1150 | 1.583746433729 | 1.929519e-01 | +3.161655e-09 |
| 1175 | 1.597606906256 | 1.989907e-01 | +3.170936e-09 |
| 1200 | 1.611179285531 | 2.050158e-01 | +3.179385e-09 |
| 1225 | 1.624692449106 | 2.109211e-01 | +3.187617e-09 |
| 1250 | 1.638353801133 | 2.167039e-01 | +3.196155e-09 |
| 1275 | 1.652347325468 | 2.224386e-01 | +3.205511e-09 |
| 1300 | 1.666829781601 | 2.163940e-01 | +3.216226e-09 |
| 1325 | 1.681681919683 | 1.969769e-01 | +3.228085e-09 |
| 1350 | 1.695852636824 | 1.782046e-01 | +3.237941e-09 |
| 1375 | 1.709382496533 | 1.564319e-01 | +3.246135e-09 |
| 1400 | 1.722454483906 | 1.373242e-01 | +3.253257e-09 |
| 1425 | 1.735230872288 | 1.204442e-01 | +3.259721e-09 |
| 1450 | 1.747855927981 | 1.054533e-01 | +3.265831e-09 |
| 1475 | 1.760457731201 | 9.210187e-02 | +3.271828e-09 |
| 1500 | 1.773149203815 | 8.022036e-02 | +3.277909e-09 |
| 1525 | 1.786021907815 | 6.971853e-02 | +3.284239e-09 |
| 1550 | 1.798741746445 | 6.083731e-02 | +3.290169e-09 |
| 1575 | 1.811104617558 | 5.361703e-02 | +3.295368e-09 |
| 1600 | 1.823141645218 | 4.793012e-02 | +3.299965e-09 |
| 1625 | 1.834884259321 | 4.367054e-02 | +3.304081e-09 |
| 1650 | 1.846364649012 | 4.073129e-02 | +3.307830e-09 |
| 1675 | 1.857615833114 | 3.898316e-02 | +3.311310e-09 |
| 1700 | 1.868671498621 | 3.826563e-02 | +3.314579e-09 |
| 1725 | 1.879565734376 | 3.839556e-02 | +3.317681e-09 |
| 1750 | 1.890332748390 | 3.918829e-02 | +3.320662e-09 |
| 1775 | 1.901006620261 | 4.047900e-02 | +3.323564e-09 |
| 1800 | 1.911621111303 | 4.213578e-02 | +3.326426e-09 |
| 1825 | 1.922209535881 | 4.406333e-02 | +3.329289e-09 |
| 1850 | 1.932804686757 | 4.620054e-02 | +3.332191e-09 |
| 1875 | 1.943438802480 | 4.851537e-02 | +3.335171e-09 |
| 1900 | 1.954143563696 | 5.099940e-02 | +3.338272e-09 |
| 1925 | 1.964950105931 | 5.366297e-02 | +3.341539e-09 |
| 1950 | 1.975889037873 | 5.653107e-02 | +3.345023e-09 |
| 1975 | 1.986990455899 | 5.964026e-02 | +3.348785e-09 |
| 2000 | 1.998283947143 | 6.303626e-02 | +3.352894e-09 |
