# Canopy as Beatnik's far-field Birkhoff-Rott solver — progress log

Session record for add-canopy. Companion to `add-canopy.md`, which holds the
design, the task sequence and the risks; this file holds what actually happened,
in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `add-canopy.md` can cite it by
ID. No dates: the order of the sections is the chronology. If a session covers
more than one task, name them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

Every accuracy number written here must carry the qualification list the
design's conventions table requires: the source distribution, the rank counts,
`P_ORDER`, `ncrit`, `max_depth`, `mac_theta`, `softening` and
`near_softening_factor` it was measured at. A bare tolerance is not a
measurement.

## Topic: read-only survey — the softening-bias floor, measured

No task was started, nothing in either repository was changed, and nothing was
built or submitted. A read-only pass established the two facts the design is
built on.

### What was read

Beatnik: `src/Beatnik_BRSolverBase.hpp`, `Beatnik_BRSolverDirect.hpp`,
`Beatnik_BRSolverFMM.hpp`, `Beatnik_FarFieldInterface.hpp`,
`Beatnik_SourceQuadrature.hpp`, `Beatnik_CreateBRSolver.hpp`,
`Beatnik_Params.hpp`, the BR call sites in `Beatnik_ZModelSolver.hpp`,
`Beatnik_Solver.hpp::requireSupportedConfiguration`, both existing milestone-0
test sources, `tests/CMakeLists.txt`, `tests/unit_tests/CMakeLists.txt`,
`scripts/tuolumne/run_milestone.flux`, and
`examples/02_adaptive_mesh_bubble/InputFile.hpp`'s BR option block.

Canopy: `src/Canopy_Solver.hpp` (the public API and `FmmConfig`), and targeted
reads of `Canopy_P2P.hpp`, `Canopy_CommunicationPlan.hpp` and `README.md`'s
Known Issues. `Canopy_LaplaceKernel.hpp`, `Canopy_UpwardSweep.hpp`,
`Canopy_DownwardSweep.hpp`, `Canopy_TreeBuilder.hpp` and
`Canopy_TreePartitioner.hpp` were **not** re-read; every statement about them in
the design is carried from `tasks/canopy0.md`, which read them in full, and is
cited to that document. **T2** is the task that first opens them.

The `origin/develop-canopy` branch, which is the structured-mesh predecessor of
this work and the only place a working Canopy integration exists:
`src/FmmBRSolver.hpp` in full (686 lines), `tasks/integrate_canopy.md`, and
`tasks/fmm_premature_nan.md`.

### Measurement 1 — the softening-bias floor on the real problem

**What was measured.** The best relative velocity error any near/far split of
Canopy's current shape can achieve, with **zero** FMM truncation error: the exact
softened Birkhoff-Rott sum against a hybrid sum that uses the softened kernel for
$r < R_0$ and the **bare** kernel for $r \ge R_0$, where
$R_0 = \texttt{near\_softening\_factor}\cdot\varepsilon$. That is exactly what
`Canopy_CommunicationPlan.hpp:347-361` produces in the limit of a perfect far
field, so it is a floor on Canopy's achievable error and Canopy's own number
must be at least as large.

**Qualification list.** Source distribution: the committed milestone-0 gold
states themselves — `tests/regression_tests/milestone0-sub3-2000-steps/gold` (642
vertices) and `milestone0-sub4-2000-steps/gold` (2562), radius 0.25, centre
$z = 0.25$. Sheet strength reconstructed from each file's own `potential` as
$S_v = -\hat n_v\times\nabla_s\phi$ with area-weighted vertex averaging of
per-face linear gradients, weighted by the lumped vertex area $A_v$ — the
`Vertex` quadrature's own construction. $\varepsilon = 0.025$,
$b = \varepsilon^2 = 6.25\times10^{-4}$, `--kernel-blob-mode length`. Serial,
one rank; no `P_ORDER`, `ncrit`, `max_depth` or `mac_theta` applies, because no
FMM ran — that is the point. Reported quantity:
$\max_t|u_{\rm hybrid}-u_{\rm soft}| / \max_t|u_{\rm soft}|$.

| factor | step 25 | 100 | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- | --- | --- |
| **L3, 4** | 2.190e-02 | 2.066e-02 | 2.240e-02 | 2.108e-02 | 1.987e-02 | 2.017e-02 |
| **L3, 8** | 5.732e-03 | 5.866e-03 | 5.031e-03 | 6.713e-03 | 7.356e-03 | 7.911e-03 |
| **L3, 12** | 1.808e-03 | 1.711e-03 | 8.289e-04 | 1.562e-03 | 1.557e-03 | 1.696e-03 |
| **L4, 4** | 2.014e-02 | 2.027e-02 | 2.106e-02 | 2.659e-02 | 1.934e-02 | 1.799e-02 |
| **L4, 8** | 5.827e-03 | 5.769e-03 | 5.268e-03 | 7.119e-03 | 6.430e-03 | 5.747e-03 |
| **L4, 12** | 1.767e-03 | 1.673e-03 | 8.243e-04 | 1.697e-03 | 1.188e-03 | 1.345e-03 |

At step 2000, extended over more factor values, with the fraction of the
$N^2$ pairs falling inside $R_0$ (the P2P share, which is what caps the
speedup):

| factor | $R_0$ | $R_0/h$ (L3) | L3 relmax | L4 relmax | near-pair fraction (L3 / L4) |
| --- | --- | --- | --- | --- | --- |
| 0 (no floor) | 0 | 0 | 6.108e-01 | 1.151e+00 | 0.0000 / 0.0000 |
| 2 | 0.05 | 1.4 | 5.420e-02 | 6.429e-02 | 0.0139 / 0.0148 |
| 4 | 0.10 | 2.9 | 2.017e-02 | 1.799e-02 | 0.0488 / 0.0526 |
| 8 | 0.20 | 5.7 | 7.911e-03 | 5.747e-03 | 0.2315 / 0.2349 |
| 12 | 0.30 | 8.6 | 1.696e-03 | 1.345e-03 | 0.5372 / 0.5577 |
| 20 | 0.50 | 14.3 | 1.700e-04 | 1.316e-04 | 0.9692 / 0.9746 |
| 40 | 1.00 | 28.6 | 0 (exact) | 0 (exact) | 1.0000 / 1.0000 |

Six things to read off these, all of which the design states as facts:

- **It is a kernel error.** Flat in step across six checkpoints spanning the
  whole run, and flat in resolution: 2.0e-02 at 642 vertices against 1.8e-02 at
  2562, at the same factor. Refining the mesh does not refine the far field.
  This is the same plateau-in-$N$ signature `tasks/treecode.md` §1 measures on
  the reference treecode, from a different cause.
- **Canopy's default is a $2\times10^{-2}$ method on this problem** — worse
  than the reference treecode's $\sim\!10^{-3}$ at its own defaults, not better.
- **The floor cannot be tuned away.** $1.7\times10^{-4}$ costs 97% of pairs in
  P2P. There is no setting that is both accurate and fast, because the geometry
  fixes the ratio: $\varepsilon$ is 5% of the bubble diameter, so the softening
  perturbs the kernel by $\tfrac32\varepsilon^2/r^2 = 3.8\times10^{-3}$ even at
  maximum separation. The softening is not a short-range regularization here.
- **The near-pair fraction is essentially independent of $N$** (4.9% against
  5.3% over a 4x range in vertex count), so the P2P share is a fixed *fraction*
  of the domain and the speedup ceiling is a bounded constant — ~20x at factor 4
  — not an asymptotic win. Weak evidence over one factor of four in $N$, and
  worth re-measuring at production resolution in **T8**.
- **`canopy0.md` F1's $\tfrac32\varepsilon^2/R^2$ bound is pessimistic by about
  5x** — 9.4% predicted against 2.0e-02 measured at factor 4 — which is the
  expected direction, since the bound is the worst-case *per-pair* kernel error
  at the floor boundary and the field error averages over all separations with
  partial cancellation. The measurement supersedes the formula.
- **Disabling the floor is catastrophic, not merely inaccurate**: 61% error at
  level 3 and 115% at level 4. That is the mechanism behind develop-canopy's
  full-roll-up NaN, seen statically.

**Method, for redoing it.** A throwaway NumPy script, not committed (the same
convention `tasks/treecode.md` used for its sweep), run on a login node: load a
gold `.npz`, reconstruct $A_v$ and $S_v$ as above, evaluate the two $O(N^2)$
sums, report the max-normalized difference and the pair fraction. Roughly 60
lines; the reconstruction of $S_v$ is the only part with any subtlety, and its
sign is irrelevant to a relative-error measurement. **T5** replaces it with an
in-tree measurement against Canopy's own MAC and its own truncation error.

**A caveat that matters for reading T5 against this.** The hybrid split uses an
exact **pairwise** cutoff; Canopy's floor is on **cell-centre** separation and
rejects a whole leaf-leaf pair, so Canopy's effective near field at a given
factor is somewhat larger than modelled here (conservative, i.e. more accurate)
while its far field carries truncation error this measurement has none of (less
accurate). The two do not cancel in a knowable direction. Treat this table as
the order of magnitude and the *shape* — flat in step, flat in $N$, quadratic-ish
in factor — not as a prediction of T5's digits.

### Measurement 2 — carried, not reproduced: Canopy's bare-kernel accuracy

From `origin/develop-canopy`'s `tasks/fmm_premature_nan.md` Resolution section,
recorded there as a decisive diagnostic: with `softening = 0`, Canopy's FMM
matched a brute-force **unsoftened** all-pairs reference to machine precision —
both `1.59762e21`, relative difference `~2e-16` — at `P_ORDER = 10`,
`fmm_max_depth = 19`, `mac_theta = 0.4`, on a 256x256 structured rocketrig deck
at 16 ranks. And on a real Beatnik BR evaluation in a configuration that keeps
the far-field separations well above $\varepsilon$, `tstFmmVsExact` measured
`max_rel = 7.6e-8` / `max_abs = 8e-12` after one RK3 step and `1.5e-6` after
five (`tasks/integrate_canopy.md` rows 8a and 8b, 1 rank, verified at 4).

**Neither number was reproduced in this pass** and both are on the structured
predecessor, not this mesh. They are the evidence that the FMM *machinery* is
sound and that the kernel is the only problem — which is what makes **X1** worth
stating as a precondition rather than a dead end. `P_ORDER` 10→16 giving
byte-identical results in the same investigation is the cleanest available proof
that the error is a wrong kernel and not an under-resolved expansion.

### The trajectory-comparison impossibility

`tasks/milestone0-progress-log.md:320-332` measures the direct path's own
divergence from a one-ulp seed: `vertices` `max|e|` goes from
`5.55111512312578270e-17` at step 0 to `8.53317416726895317e-13` at step 2000
(level 3, SERIAL, np1, against the Python), i.e. about $10^4$ amplification,
**power-law rather than exponential**, and at level 4 not even monotone (peak
`3.17634807345257286e-13` at step 1400). So a per-evaluation perturbation of
size $\delta$ lands the step-2000 trajectory roughly $10^4\delta$ away before
any accumulation over the 6000 evaluations is counted, and passing the existing
`--rtol 1e-10 --atol 1e-12` needs $\delta \lesssim 10^{-15}$ — the direct sum.

That is why **T6** asserts two separate claims instead of running the existing
member with `--br-approximation fmm`, and why claim B's rung is derived from a
measurement (**T5** step 5) rather than chosen. The extrapolation from a
one-off seed to a perturbation injected at every evaluation is *not* measured and
is the reason step 5 exists.

**Affects:** **T1** — `FmmParams` must reach `near_softening_factor`, which is
the single most consequential knob on this path and has no CLI counterpart.
**T5** — its scan must include factor values well above Canopy's default 4, and
must be read against this table and the caveat above rather than in isolation.
**T6** — both claims, and claim B's rung, follow from the two findings here;
do not start it from a reading in which the existing member plus a flag would
have worked. **T8** — the near-pair fraction, not just the wall clock, is what
explains the measured speedup. **X1** — measurement 2 is why the precondition is
believed achievable.
