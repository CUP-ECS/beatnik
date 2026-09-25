# Canopy as Beatnik's far-field Birkhoff-Rott solver — progress log

Session record for add-canopy. Companion to
[add-canopy.md](add-canopy.md), which holds the design, the task sequence and
the risks; this file holds what actually happened, in order.

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
**the basis**, `order`, `ncrit`, `max_depth`, `mac_theta`, `softening` and
`near_softening_factor` it was measured at. A bare tolerance is not a
measurement.

Four things this project in particular will want back later, so record them
where they arise:

- **$\tau_A$ and the production parameter set** (T5), with the full
  qualification list and the reasoning for the chosen `order` over the one above
  and the one below it. Every compiled tolerance in T4, T6 and T7 traces here.
- **The divergence-horizon envelope and its run-to-run spread** (T5 step 7,
  R8). An envelope with no spread beside it is not usable, and T6 will trip on
  it intermittently.
- **The realized operator-key count and `total_fallback_pair_count()`** at each
  scanned `max_depth` and `order` (T5 step 5, R6). A non-zero fallback count
  means the accuracy number is a mixture of two code paths.
- **The `MaintenanceAction` histogram and the operator-construction share of
  `solve`** (T8). Together they decide whether `Rebalance`-every-stage is
  affordable, which is the question the whole cost model turns on.

## Topic: read-only survey

No task was started, nothing in either repository was changed, and nothing was
built or submitted. A read-only pass established the facts the design is built
on and fixed its fidelity target.

### What was read

Beatnik: `src/Beatnik_BRSolverBase.hpp`, `Beatnik_BRSolverDirect.hpp`,
`Beatnik_BRSolverFMM.hpp`, `Beatnik_FarFieldInterface.hpp`,
`Beatnik_SourceQuadrature.hpp`, `Beatnik_CreateBRSolver.hpp`,
`Beatnik_Params.hpp`, `Beatnik_Types.hpp`, the BR call sites in
`Beatnik_ZModelSolver.hpp`, `Beatnik_Solver.hpp::requireSupportedConfiguration`,
both existing milestone-0 test sources, `tests/CMakeLists.txt`,
`tests/unit_tests/CMakeLists.txt`, `scripts/tuolumne/run_milestone.flux`, and
`examples/02_adaptive_mesh_bubble/InputFile.hpp`'s BR option block.

Canopy: `src/Canopy_Solver.hpp` (the public API and `FmmConfig`), and targeted
reads of `Canopy_P2P.hpp`, `Canopy_CommunicationPlan.hpp` and `README.md`'s
Known Issues. The far-field contract and the two bases were taken from
`../../../canopy/tasks/abstract-solver-backend.md` rather than from the headers, and the
basis-independent findings — maintenance cost, the knob semantics, the open
defects — from `../../../canopy/tasks/canopy0.md`, which read those headers in full. **T2** is
the task that first opens them directly.

The `origin/develop-canopy` branch, which is the structured-mesh predecessor of
this work and the only place a working Canopy integration exists:
`src/FmmBRSolver.hpp` in full (686 lines), `tasks/integrate_canopy.md`, and
`tasks/fmm_premature_nan.md`.

### Why $\tau_A$ is $10^{-3}$ and not tighter

The target is the reference implementation's own default far field, not a round
number. `tasks/treecode.md` §1 measured the Python's Barnes-Hut treecode against
its own direct sum on the reference's benchmark configuration (icosphere radius
0.5, `potential = 0.5x + 0.3y`, `A=0.3 g=1.0 eps=0.025 mu=0.002`,
`use_matlab_blob=False` so $b = \varepsilon^2 = 6.25\times10^{-4}$, `vertex`
quadrature, serial): at the reference's defaults ($\theta=0.3$, order 2,
`ncrit` 64) it sits at $8.1\times10^{-4}$ at 10242 sources, $4.8\times10^{-4}$ at
2562 and $1.6\times10^{-3}$ at 642 — the last two being milestone-0's own two
levels. The reference's README makes the same claim independently. That path is
the reference's **default**, so the physics this port reproduces was produced at
$\sim\!10^{-3}$, and a Beatnik far field at $\tau_A \le 10^{-3}$ is parity rather
than a concession.

Tightening it was considered and rejected on two independent grounds, both
recorded because the argument will otherwise be had again:

- **It would not buy a trajectory comparison.** See the next subsection. Nothing
  reachable by an affordable expansion order gets a 2000-step FMM-driven run
  inside any rung the existing ladder uses.
- **It is the wrong side of the basis's cost curve.** A Cartesian-Taylor
  truncation buys 0.24-0.48 decades per order at standard admissibility, 0.82-1.06
  at Beatnik's `mac_theta = 0.3`, while the DOF count grows as
  $\binom{p+3}{3}\sim p^3/6$ (`../../../canopy/tasks/canopy-kernel-rec.md`, "Convergence
  per DOF"). $10^{-3}$ is estimated at $p = 2$-$4$ and 10-35 DOF per cell;
  $10^{-6}$ wants $p \approx 11$-$24$ and 364-2925. The target sits where the
  basis is cheap, one decade before it stops being.

**Neither figure was reproduced in this pass**; both are read from
`tasks/treecode.md`, whose sweep was produced by a throwaway NumPy script on a
login node against the reference tree, read-only. **T5** replaces the estimate
with an in-tree measurement against Canopy's own MAC and its own truncation
error, at both bases.

### The trajectory-comparison impossibility

`tasks/milestone0-progress-log.md:320-332` measures the direct path's own
divergence from a one-ulp seed: `vertices` `max|e|` goes from
`5.55111512312578270e-17` at step 0 to `8.53317416726895317e-13` at step 2000
(level 3, SERIAL, np1, against the Python), i.e. about $10^4$ amplification,
**power-law rather than exponential**, and at level 4 not even monotone (peak
`3.17634807345257286e-13` at step 1400).

A $10^{-3}$ perturbation injected at **every** evaluation is thirteen orders
above that seed. Passing the existing `--rtol 1e-10 --atol 1e-12` would need
$\delta \lesssim 10^{-15}$ — the direct sum. `tasks/treecode.md` §1 reaches the
same conclusion for the reference treecode and states it flatly: a $10^{-3}$
far-field path can never be trajectory-compared against a direct gold set, not at
$10^{-10}$ and not at $10^{-6}$.

That is why **T6** asserts two separate claims instead of running the existing
member with `--br-approximation fmm`, and why claim B is a stability and
divergence-horizon measurement rather than a loosened gold rung. The
extrapolation from a one-off seed to a perturbation injected at every evaluation
is *not* measured, which is what **T5** step 6 exists to fix.

### Carried, not reproduced: the FMM machinery is sound

From `origin/develop-canopy`'s `tasks/fmm_premature_nan.md` Resolution section:
with `softening = 0`, Canopy's FMM matched a brute-force all-pairs reference to
machine precision — both `1.59762e21`, relative difference `~2e-16` — at
`P_ORDER = 10`, `fmm_max_depth = 19`, `mac_theta = 0.4`, on a 256x256 structured
rocketrig deck at 16 ranks, solid-harmonic basis. And on a real Beatnik BR
evaluation in a configuration keeping far-field separations well above
$\varepsilon$, `tstFmmVsExact` measured `max_rel = 7.6e-8` / `max_abs = 8e-12`
after one RK3 step and `1.5e-6` after five (`tasks/integrate_canopy.md` rows 8a
and 8b, 1 rank, verified at 4).

Neither number was reproduced in this pass and both are on the structured
predecessor, not this mesh. They are evidence about the parts of Canopy the basis
selection does not touch — the tree, the partition, the MAC, the dual-tree
traversal, the communication plan and P2P, all of which are basis-blind — so they
carry to `CartesianTaylorBasis` even though they were measured under the
solid-harmonic one. They are the reason **T5** is expected to find a clean
truncation curve rather than a diagnosis problem.

**Affects:** **T1** — the fidelity target above is what makes
`near_softening_factor = 0` the default rather than a tuning choice, and what
makes `order`'s default of 2 plausible rather than arbitrary. **T5** — its scan
must be read against the decades-per-order estimate and the $p = 2$-$4$
expectation, and step 6 exists because the every-evaluation extrapolation is
unmeasured. **T6** — both claims, and claim B's shape, follow from the
trajectory-comparison impossibility; do not start it from a reading in which the
existing member plus a flag would have worked. **X1** — the machinery evidence is
why a disappointing **T5** would point at the basis rather than at Canopy.

## Canopy's Cartesian-Taylor basis lands (no Beatnik task)

Canopy built `CartesianTaylorBasis` and measured it. Nothing in Beatnik changed;
the design document was reconciled against the new material and several of its
estimates are now measurements. Recorded here because the estimates this log
carries above are among the things superseded, and this log's own rule is that a
number here outranks an estimate in the design.

**Read as measured, not estimated**, and all from
`../../../canopy/tasks/cartesian-taylor-basis.md` T1-T5 (DONE) and its progress
log. Every figure below is on 8640 particles in a **volumetric cube**,
`ncrit = 8`, `max_depth = 6`, `replication_depth = 2`, `softening = 0.025`,
`near_softening_factor = 0`, four solves with `migrate / rebalance / migrate`
between them, ranks 1-6, against a direct softened sum — not on a sheet, which
is why **T5** still has to measure Beatnik's geometry.

| arm | potential | gradient |
| --- | --- | --- |
| $\theta=0.3$, $p=3$ | $1.9263\times10^{-5}$ | $7.0718\times10^{-4}$ |
| $\theta=0.3$, $p=2$ | $2.655\times10^{-4}$ | $8.996\times10^{-3}$ |
| $\theta=0.5$, $p=2$ | $9.9667\times10^{-4}$ | $1.8652\times10^{-2}$ |
| $\theta=0.3$, `LaplaceKernel`, `near_softening_factor = 0` | $1.894\times10^{-2}$ | $5.168\times10^{-2}$ |

**The estimate this log recorded above is superseded.** "$10^{-3}$ is estimated
at $p = 2$-$4$ and 10-35 DOF per cell" was on the potential and on a
decades-per-order model with an unmeasured constant. Two things corrected it.
The constant is now measured, $c\approx1$ in half-widths ($1.062$, $0.987$,
$0.939$ at $R/w=8,16,32$). And the quantity is wrong: Beatnik reads the
**gradient**, which truncates one order before the potential because
$\nabla$ of a degree-$p$ Taylor local is degree $p-1$, so the model for what
Beatnik needs is $(\theta/2\sqrt3)^{p}$ and not $(\theta/2\sqrt3)^{p+1}$. That
is measured rather than argued: the ratio of absolute errors came out $29.47$
and $218.5$ against $1/W = 29.88$ and $223.7$ on two domains 12-fold apart in
scale, where a source-side loss would have given $1/R$ ($2.59$ and $19.4$). The
reference treecode has no target-side expansion at all
(`_expansion_batch` is evaluated at `rv = target - node.center`,
`treecode.py:121-126`), so its documented order-2 **velocity** figure is the
same truncation order as Canopy's order-2 **potential**. Beatnik's production
order is therefore **3**, not 2, and T1 compiles that default.

**Three things the design now states that a session should not re-derive.**

- **The operator cache retains nothing on a moving distribution.** Zero keys
  retained at every one of 336 builds, 3.9x the cached key count constructed
  over four solves, drift a smooth 0.18-0.40% per build. Mechanism:
  `key_needs_level = true` plus a root box recomputed from the particles on
  every maintenance path, including `migrate`. There is no Beatnik-side knob —
  the six bounding-box tolerances pad by fractions of the box's own width, so a
  padded box drifts too.
- **The far field is not automatically live.** At $\theta=0.3$ the near field
  covers about 105 occupied leaves of a 2-manifold, so a meaningful far field
  needs $N \gg 105\cdot\texttt{ncrit}$ — $N \gg 6720$ at `ncrit = 64`. Neither
  milestone-0 level clears that at the default, and 642 vertices clears it at no
  `ncrit` at all. An all-P2P solve agrees with `BRSolverDirect` to round-off and
  reads as success, which is why every accuracy figure from here on carries its
  P2P pair fraction.
- **Realized key counts are close to the cap.** 25438 unique M2L keys at
  $\theta=0.3$ against `M2L_OP_COUNT_CAP = 32768`, 7374 at $\theta=0.5$, 246 and
  0 fallback pairs, on 2941 cells. Canopy saw the $\theta=0.3$ table saturate
  the cap outright once the trajectory was amplified.

**One upstream item is now a hard gate.** Canopy's derivative ladder has
closed-form oracles only to $|k|=3$ and a finite-difference oracle run at
$|k|=4$ and nowhere else, while the M2L evaluates $b_{p+q}$ to $|p+q|=2p$. At
$p=3$ that is $|k|=6$. Canopy's R8 is marked **LIVE** and its **T6** is NOT
STARTED. **T5** and **T6** here are gated on it at $p\ge3$. Canopy's own $p=3$
arm passing is not evidence — that arm is precisely what the oracle does not
cover.

**Also settled, in passing:** Canopy's runtime API is unchanged by the far-field
abstraction (`setup`, `solve`, `auto_maintain`, `num_local_particles`,
`potential`, `gradient` all as the design describes); `FmmConfig::softening`
defaults to $-1.0$, which selects distribution-based auto-softening rather than
failing, and additionally disables `near_softening_factor`; and the np=4 defect
is narrower than recorded — `CartesianTaylorSolve` drives `NComps = 3`, compares
the gradient and passes at ranks 1-6, though at a different basis, order,
softening and distribution than `SingleSolve`.

**The copies of Canopy's design documents under `tasks/canopy/` are gone.** They
had drifted — the local `abstract-solver-backend.md` was 955 lines against
canopy's 2537 — and a stale copy of an upstream document is worse than no copy.
Cite `../../../canopy/tasks/<file>` instead.

**Affects:** **T1** — `order` defaults to 3 and its comment must say why the
reference's 2 does not transfer; `ncrit` stays 64 but carries the
$N\gg105\cdot\texttt{ncrit}$ constraint; `softening` must be set explicitly
positive. **T2** — the dispatch set starts at the production order 3, and the
config builder throws on a non-positive `softening` because Canopy's sentinel
is not an error there. **T4** — must run at an `ncrit` that makes the far field
live and assert the P2P fraction; `order = 2` becomes a third, intermediate
point in the order-knob negative case. **T5** — gated on Canopy's T6 at
$p\ge3$; `ncrit` is a scan axis rather than a background; potential and
gradient are reported separately at every point. **T6** — inherits that gate,
and its L3 member's claim A is largely a P2P comparison and must say so.
**T8** — the operator-cache question is answered and step 2 measures the cost of
the rebuild rather than whether one happens.

## Canopy's ladder validation lands (no Beatnik task)

Canopy's T6 is **DONE** and its R8 is closed for $p = 3$. The hard gate the
previous entry put on **T5** and **T6** is lifted; nothing in this document now
waits on upstream work. Recorded because the previous entry's `Affects:` line
said those two tasks were gated, and a session reading only that line would
still believe it.

**What was measured** (`../../../canopy/tasks/cartesian-taylor-basis.md` T6,
job `f3Ze8yUzNVSo`, all 12 bodies green). The finite-difference oracle now runs
at $|k| = 4 \ldots 6$, each degree with its own Richardson step divisor located
by an eleven-point scan from $L/2$ to $L/64$ — the floors turned out to sit at
*different* $h$ per degree, $L/24$ at 4 and 5 and $L/16$ at 6, so carrying
$L/32$ over would have been wrong:

| $\vert k\vert$ | divisor | tolerance | achieved worst | margin |
| --- | --- | --- | --- | --- |
| 4 | $L/24$ | $4\times10^{-6}$ | $3.080\times10^{-7}$ | 13.0x |
| 5 | $L/24$ | $3\times10^{-4}$ | $2.531\times10^{-5}$ | 11.9x |
| 6 | $L/16$ | $2\times10^{-2}$ | $1.184\times10^{-3}$ | 16.9x |

No tolerance was widened. Perturbing one recurrence coefficient by 0.1% put all
three degrees over their own bounds — 6500x, 913x, 142x — so 5 and 6 are
exercised and not merely enumerated. $|k| = 4$ came out *sharper* than it
shipped ($3.080\times10^{-7}$ against $7.321\times10^{-7}$), because the
eleven-point scan found a floor the original four-point scan straddled.

**Two details that bear on Beatnik specifically.** The oracle's sample set now
permanently carries $b = 6.25\times10^{-4}$ — which is Beatnik's
$\varepsilon^2$ at $\varepsilon = 0.025$ — and the interior scales
$|r|/\sqrt b \in \{9.276, 15.46, 74.2\}$, so the band is sampled from the
inside rather than bracketed by 1.0 and 100.0. The worst deviation at both
$|k| = 5$ and $|k| = 6$ falls at that $b$. And the growth in achieved deviation,
about 40x per degree, tracks the roundoff floor of a $|k|$-th difference rather
than the ladder's conditioning — so it is the oracle losing resolution, not the
recurrence losing accuracy. R8's predicted failure mode is visible and bounded
rather than merely absent.

**What did not close.** R8 remains open above $p = 3$: $p = 4$ reaches
$|k| = 8$ and the oracle stops at 6. Canopy's 1-D stencils now **abort loudly**
on an order they do not carry, replacing a `default:` branch that silently
reused the order-4 stencil — so the gap fails visibly if someone raises $p$,
which is why this is a constraint on adoption rather than a hazard. Beatnik's
**T2** dispatch set reaches $p = 5$ and **T5** scans it, so the scan will
produce points at $|k| = 8$ and $|k| = 10$ that nothing covers. Those are
reportable measurements; none may become the production order without another
degree of upstream oracle.

**Affects:** **T5** — the hard gate is gone; its **Depends on** is T4 alone,
and the residual constraint is that a scan point above $p=3$ may be measured
but not adopted (step 6). **T6** — no upstream gate; the production order is
validated. **T2** — the dispatch set's orders 4 and 5 are marked scan-only on
the dispatch. **R11** — retargeted from "Beatnik ships on unchecked arithmetic"
to "a scan point above $p=3$ is promoted to production"; **R10** — the contrast
it draws is now between one conditional that never fired and one that landed.
Nothing else changes: $\tau_A$, the production order 3, the `ncrit` constraint
and the operator-cache findings all stand.

## T1

`FmmParams` went from three members to sixteen, `FarFieldBasis` landed beside the
other mode enums, and `order`'s default rose from 2 to 3. `spack install` is
green — 29 CXX objects, including `adaptive_mesh_bubble.cpp.o`, after touching
that driver, because `Beatnik` is an INTERFACE library that would otherwise have
reported a header-only no-op. Nothing was run: no binary was invoked, and the
`--help` claims were checked by `grep` on the static string literal `printSchema`
emits.

**Decisions taken as given by the task, recorded so they are not reopened.**
`max_depth` defaults to **10**. README's new documentation went into the
`02_adaptive_mesh_bubble` section as an `###### FMM tunables` subsection under
`##### Birkhoff-Rott approximation`; the older `fmm_*` block at `README.md:119-128`
belongs to `01_rising_bubble` (`rocketrig`), whose parameter struct no longer
exists in `src/`, and was left untouched including its solid-harmonic-only
`fmm_near_softening_factor` paragraph — the new subsection ends with one sentence
saying those keys are a different, older set that does not reach `FmmParams`, so a
reader who finds both does not have to guess. `InputFile.hpp` reduced to comment
corrections: **no already-parsed key maps to any new member**, and in particular
`--br-near-factor` was **not** wired to `near_softening_factor`. Those are
different quantities — `--br-near-factor` is the Python's local/clustered
near-field *radius* and `near_softening_factor` is a Plummer softening floor in
multiples of the softening length — so wiring them would be exactly the silent
semantic substitution the Conventions table's "Runtime dispatch" row forbids, and
it stays IGNORED.

**`max_depth = 10`, as the reasoning was finally stated on the declaration.** Two
forces, and the comment names both. Downward: a 2-manifold at leaf occupancy
`ncrit` reaches `N/ncrit` occupied leaves in log₄ rather than log₈ levels, so at
`ncrit = 64` the sheet needs about 3 levels at 2562 vertices and about 7 at a
million — 10 never binds on a well-behaved sheet, and the declaration says so
explicitly to keep a future reader from treating it as a target. Upward: what it
*does* bound is a self-contacting roll-up, where occupancy stops falling off and
the occupied-depth count climbs; under `CartesianTaylor` the M2L keys carry the
tree level, so every occupied depth multiplies the realized key count against
Canopy's 32768-key cap (**R6**; Canopy measured 25438 keys — 78% of the cap — at
θ=0.3 on a volumetric cloud at `max_depth` 6, and saw the table saturate it
outright on an amplified trajectory). The comment states that overflow is not an
error but a route to a slower, bitwise-different per-pair translate, so an
overflow yields an accuracy number mixing two code paths. It also states that the
number is **reasoned, not measured**, that **T5** owns revising it, and — per
**R12** — that it may be lowered only on evidence of realized overflow at the
production configuration, never pre-emptively to shrink the table. develop-canopy's
19 and its depth-driven finite-difference blow-up are noted as *not* transferring:
that mechanism lived in a finite-difference L2P the analytic Taylor L2P removes.

**Defaults chosen for the knobs the document did not fix.** develop-canopy's
`src/Solver.hpp:74-104` was the precedent for all of these; Canopy's own
`FmmConfig` defaults differ from it on four of them, and where they do this is the
reasoning for which was taken.

- **`ncrit_tol = 0.1`** — agrees with both Canopy's `FmmConfig` and
  develop-canopy, so no judgment was needed. Canopy uses it as
  `coarsen_threshold = ncrit * (1 - ncrit_tol)`, i.e. 57 at `ncrit = 64`, so a
  just-split cell does not re-merge when a few particles leave. Beatnik's surface
  deforms on every RK stage, which is precisely the thrashing the band damps, so
  it was kept rather than tightened.
- **`replication_depth = 3`** — develop-canopy's value, *not* `FmmConfig`'s **1**.
  `FmmConfig`'s 1 sits below the 2-4 range Canopy's own `TreePartitioner`
  documents as typical, and `TreePartitioner`'s constructor default is itself 3.
  Cells at depth ≤ this are replicated on every rank; at depth 1 that is 9 cells,
  fewer coarse cells than ranks at the top of the 1-6 range this path runs at.
  Canopy bounds the cost of 3 directly: at most 585 cells (1+8+64+512). This is
  ownership, not occupancy, so it does not interact with `max_depth`'s key-count
  bound, and the declaration says that so the two are not traded against each
  other by mistake.
- **`imbalance_tolerance = 0.10`** — develop-canopy's value, *not* `FmmConfig`'s
  **0.05**, and looser on purpose. A deforming surface picks Canopy's `Rebalance`
  path essentially every RK stage (canopy0 F3(c)), so the partitioner runs at that
  frequency and a tighter tolerance buys balance at a cost paid every stage; at
  1-6 ranks a 10% imbalance is small in absolute terms.
- **The six bounding-box factors = 0.10 each** — develop-canopy's uniform value,
  *not* `FmmConfig`'s **0.0**. Semantics, from develop-canopy's comment block and
  confirmed against `TreeBuilder::build`: each is a fraction of that axis's width
  applied as padding to the global root box on that face. Zero padding makes the
  box hug the particles exactly, so any outward motion invalidates it immediately
  and forces the heavier maintenance path; 0.10 buys a stage or two of room.
  Uniform rather than asymmetric because the bubble is not confined on any face.
  **The declaration states what padding does *not* buy**, because this is the
  obvious wrong lever to reach for: it does not stabilize the box and raising it
  will not make it, since Canopy recomputes the box from the particles on every
  maintenance path including `migrate`, and a fraction of a moving box moves with
  it — the measured 0.18-0.40% per-build drift that clears the entire M2L cache
  (zero keys retained across 336 builds). The one real cost is noted too: at 0.10
  the box is 20% wider per axis, so every cell at a given depth is 20% wider,
  which shifts where `ncrit` occupancy is reached without changing the number of
  occupied depths.
- **`m2l_op_table_byte_budget` = 2 GiB** — Canopy's own default, kept because it
  is the *non-binding* half of Canopy's `min(count cap, budget/bytes_per_key)`.
  At order ≤ 4 a column is ≤ ~9.8 KB, so the full 32768 keys occupy roughly
  0.3 GiB and the count cap binds first at every order this path supports.
  Lowering it is the only way to make the budget bind, and per **R6** that is the
  wrong lever — the response to realized overflow is a lower `max_depth` or
  `order`. The declaration says so, so nobody tunes the budget hoping to change
  the cap.

**What turned out to differ from what the document says.**

- **The member name.** The document's T1 step 3 said `m2l_operator_byte_budget`.
  Canopy's member is **`m2l_op_table_byte_budget`**. The Beatnik member is named
  to match Canopy's and `add-canopy.md:706` was corrected in this change.
- **Four `FmmConfig` defaults are not develop-canopy's**, and the document, which
  cites develop-canopy's set as the precedent, does not flag the divergence:
  `replication_depth` is **1** in `FmmConfig` (3 in develop-canopy and in
  `TreePartitioner`'s own constructor), `imbalance_tolerance` is **0.05** (0.10),
  and all six bounding-box tolerances are **0.0** (0.10). Only `ncrit_tol` (0.1)
  and the byte budget (2 GiB) agree. Every one of the four is resolved above.
- **`basis` reaches no `FmmConfig` member.** The document's framing — "every knob
  `FmmConfig` needs" — does not fit it: Canopy's far field is a *template*
  parameter on `Solver`/`createSolver` (defaulted to `LaplaceKernel`), not a
  config field. Same for `order`, which is `P_ORDER`. Both declarations say this
  outright, because it is why **T2** must name the basis at every instantiation
  rather than set it in a struct — the **R2** failure mode is an omitted template
  argument compiling cleanly into a bare-1/r far field.
- **The conventions table's Canopy-isolation check is not a usable invariant, and
  was already failing before this task.** It states that
  `grep -l Canopy src/*.hpp` must name exactly `Beatnik_FarFieldInterface.hpp` and
  `Beatnik_Config.hpp.in`. In fact 16 headers match, because the pattern hits the
  word "Canopy" in prose comments — `Beatnik_Params.hpp`'s `FmmParams` doc comment
  has said "Canopy fast-multipole tunables" since before this change. The
  enforceable invariant is the one the task constraints state and this change was
  checked against: `grep -n "include.*Canopy_\|Canopy::" src/*.hpp src/*.in`
  returns nothing. A later task should fix the table rather than trust the grep.
- **A `toString( FarFieldBasis )` was added**, which T1's steps do not ask for.
  All ten enums in `Beatnik_Types.hpp` have one and the file's own section comment
  states the one-table-per-enum rule, so omitting it would have been the anomaly;
  and the Accuracy-claims convention requires every figure from **T5** on to name
  the basis, which needs a spelling. No `fromString` and no CLI parse path was
  added — there is no `--basis` option and none is wanted.

**Affects:** **T2** — consumes every default above in its `FmmConfig` builder, and
the mapping is mechanical for fourteen of the sixteen members: `basis` and `order`
are **not** config fields but template parameters (`FarField` and `P_ORDER`), so
the builder cannot set them and the dispatch must name them. The builder is also
where `softening` comes from — there is deliberately no `FmmParams::softening`,
so it must pass `sqrt(ZModelParams::blob())` (i.e. `eps` under `Length`,
`sqrt(eps)` under `Matlab`) and must **throw** on a non-positive result, since
Canopy's −1 sentinel is auto-softening rather than an error and additionally
disables `near_softening_factor`. Validation of `ncrit` and `max_depth` lands
there too (T1 step 9). **T3** — owns the `@note` blocks on the softened kernel and
README's accuracy statements about the FMM path; the new `###### FMM tunables`
subsection states defaults and mappings only, and makes no accuracy claim beyond
quoting Canopy's two measured gradient errors at p=2 and p=3 as the provenance of
`order = 3`. **T4** — the deferred half of T1's own exit criterion is its test's:
every `FmmConfig` member initialized, `softening` positive rather than the
sentinel, `near_softening_factor` 0. **T5** — owns revising `max_depth` from
measurement, and the byte budget is not a scan axis. Nothing about `mac_theta`,
`ncrit`'s liveness inequality, τ_A or the production order changed.

## T2

The adapter is real. `Beatnik_FarFieldInterface.hpp` went from 200 lines of
stubs to ~1300 lines holding the six-arm type-erased dispatch, the `FmmConfig`
builder, the tag-reverse round trip and the diagnostics. `spack install` is
green on `+canopy` (29 CXX objects from a cleaned build directory) and green
again on `~canopy`. **Nothing was run** — no binary, no job, no test — and no
accuracy, tolerance or speedup number is claimed. The only measurement here is
a compile cost.

**Decisions taken as given by the task, recorded so they are not reopened.**
The three signatures are `evaluateVelocity`, `evaluateRieszScalar` and
`diagnostics()`, with no `targets` argument and concrete (not templated) views;
the `TODO(types)` comments are gone rather than carried forward.
`FarFieldDiagnostics` is the third signature and `BRSolverFMM` gained
`const far_field_type& farField() const` in the same change. The P2P pair
fraction is computed in the adapter — Canopy has no such counter, and its
public `P2P` surface is `num_ghost_particles()`, `ghost_positions()` and
`ghost_charges()` and nothing else. The dispatch is a type-erased pimpl over an
abstract `Impl` with one `ImplFor<FarField, P_ORDER>` per arm: `CartesianTaylor`
at 0, 2, 3, 4, 5 and one `SolidHarmonic` at 3. A non-`Vertex`
`source_quadrature` is rejected in the adapter with `std::runtime_error` naming
the quadrature in force.

### The exit criterion was not reachable as written: `~canopy` did not disable Canopy

This is the finding of the task. `spack install` with the environment's Beatnik
spec flipped to `~canopy` produced a binary with **2263 `Canopy::` symbols** and
an installed `Beatnik_Config.hpp` carrying `#define BEATNIK_ENABLE_CANOPY`.

The mechanism, from the archived `CMakeCache.txt`. The spack package maps the
variant to `-DBeatnik_REQUIRE_CANOPY=OFF` and to dropping the `depends_on`, and
both did happen — the cache holds `Beatnik_REQUIRE_CANOPY:BOOL=OFF` and no
Canopy path appears anywhere in `CMAKE_PREFIX_PATH`. But `Canopy_FOUND` was
still true, because the *environment view* is searched and canopy is a root
spec of this environment, so `find_package(Canopy QUIET)` resolved
`Canopy_DIR` to `<env>/.spack-env/view/share/cmake/Canopy`. And
`Beatnik_add_dependency` ended with

```cmake
set(Beatnik_ENABLE_${OPT} ${${PACKAGE}_FOUND})
```

so `Beatnik_REQUIRE_CANOPY=OFF` meant "do not insist on finding it" and never
"do not use it". There was **no way at all** to build `~canopy` on a machine
that has Canopy installed — not via the spack variant, and not via
`-DBeatnik_ENABLE_CANOPY=OFF` either, since that line is a normal `set()` that
overrides any cache entry in the same scope. Since that is exactly the build
half of T2's exit criterion, the criterion was untestable as written on this
system and, in an environment shaped like this one, so is README's claim that a
`~canopy` build refuses `fmm` at run time.

**Fixed in [CMakeLists.txt](../../CMakeLists.txt), in this change**, by keying
`Beatnik_ENABLE_*` off `Beatnik_REQUIRE_*`:

```cmake
if(Beatnik_REQUIRE_${OPT})
  find_package(... REQUIRED ...)
  set(Beatnik_ENABLE_${OPT} ON)
else()
  set(Beatnik_ENABLE_${OPT} OFF)
endif()
```

Behavior is unchanged in every case except the broken one: `Beatnik_REQUIRE_*`
still defaults to `${PACKAGE}_FOUND`, so a plain `cmake ..` with Canopy present
still enables it and one without it still disables it; `+canopy` with Canopy
missing still fails at configure. Only an *explicit* OFF changed meaning, and it
changed to what it says. This is a top-level-CMakeLists edit that T2's "Fill in"
list does not name, so it is called out here and in T2's **Met.** paragraph
rather than buried: it is one line of logic and trivially revertible, and the
alternative was shipping T2 with half its exit criterion unverified. README's
Canopy dependency bullet was corrected in the same change. Left alone
deliberately: the `option()` default is written `${PACKAGE}_FOUND` rather than
`${${PACKAGE}_FOUND}`, so the cache holds the literal string `Canopy_FOUND` and
works only because `if()` re-dereferences it. It is self-correcting and was not
touched.

### "Compiles under `~canopy`" needed more than a green build

A `~canopy` build instantiates `FarFieldSolver` **nowhere**:
`Beatnik_CreateBRSolver.hpp` preprocesses out the `new BRSolverFMM<...>` line,
which is the class's only construction site, so the template is parsed and
definition-checked but never instantiated — `nm` found zero `FarFieldSolver`
symbols. That is weaker than the criterion's "compiling and throwing". Closed by
temporarily adding

```cpp
template class Beatnik::FarFieldSolver<Kokkos::DefaultExecutionSpace,
    Kokkos::DefaultExecutionSpace::memory_space>;
```

to the example driver and rebuilding: every member instantiated (constructor,
both evaluations, `diagnostics()`, `params()`, `throwNoCanopy`), zero `Canopy::`
symbols, and the `std::runtime_error` text naming
`Beatnik_ENABLE_CANOPY=ON, spack '+canopy'` present in the binary. The
instantiation was removed and `+canopy` reinstalled. **This gap reopens the
moment someone edits the guarded half of the header**, because nothing in the
tree instantiates it; a one-line instantiation in a `~canopy`-guarded unit test
would close it permanently and is left for **T4**.

### Where the implementation departed from T2's Do steps

- **The arm is selected in the constructor, not at the first evaluation**
  (step 5 says "constructed once on the first evaluation"). Two reasons, and the
  first is the important one. Lazily, `makeImpl` is instantiated only when an
  evaluation is instantiated — and `BRSolverFMM`'s virtuals still throw, so
  nothing calls one until **T3**. The six arms would not have been compiled by
  this task at all, and the `+canopy` half of the exit criterion would have
  proved almost nothing. Eagerly, the switch is instantiated from
  `BRSolverFMM`'s constructor and all six arms are emitted, which `nm` confirms.
  Second, an unsupported `(basis, order)` now fails at solver construction
  rather than at the first Runge-Kutta stage. The Canopy `Solver` itself is
  still constructed lazily, at the first evaluation, because `FmmConfig` needs
  `sqrt(ZModelParams::blob())` and `ZModelParams` does not reach the
  constructor — so "constructed once and persistent" still holds of the thing
  that matters, the tree and the communication plan.
- **The forward distributor validates itself, and a stale round trip falls back
  to `setup()`.** Step 3 describes the reuse branch unconditionally, which is
  right for develop-canopy's fixed grid and wrong here: refinement, coarsening
  and mesh load balancing all change how many owned vertices a rank has and what
  a given index means, and `--dynamic-remesh` is on by default. A stale tag map
  does not fail loudly — an unclaimed row gets `-1`, which `Cabana::Distributor`
  documents as "drop this element", so the source would vanish from the global
  sum silently. `buildForwardDistributor` therefore reports whether every row
  was claimed exactly once with no out-of-range claim; that flag is
  `MPI_LAND`-reduced and the sequence falls back to a full `setup()` when any
  rank says no. Cost is one small allreduce and one reduction over work already
  being done. The branch is on the *reduced* value, so step 8's rule — never
  branch the collective sequence on a local count — holds.
- **A round-trip completeness check on the reverse leg**, likewise not in the
  steps: the scatter counts out-of-range tags and the evaluation compares the
  returned tuple count against the owned source count, both folded into the one
  allreduce below, and throws on any mismatch across the communicator. The
  conventions table's "never return a truncated or best-effort field" is what
  motivates it; **R3** is what it is aimed at.
- **One `MPI_Allreduce` of five `long long` values, not five allreduces.** The
  global particle count, the P2P pair count, the two M2L pair counts and the
  round-trip error count are all sums over the same communicator at the same
  point, so they travel together.
- **One tuple layout serves both contractions.** The `Output` member is
  `Real[3]` and the Riesz scalar uses component 0 only, wasting two doubles per
  particle. The alternative was a second AoSoA type, a second tag definition and
  a second round trip, which is the more expensive kind of cost.
- **A softening-stability guard.** `ZModelParams` arrives at every evaluation,
  but Canopy fixes the softening at `Solver` construction and pushes it into the
  M2L operator tables, so a later evaluation presenting a different `blob()`
  would silently run a different kernel in the far field than in the near field.
  The adapter holds the value and throws on a change. The comparison is exact
  and safe to be: the value is `sqrt(eps*eps)` recomputed from the same struct.
- **A `toString( FarFieldDiagnostics::Maintenance )`**, which no step asks for,
  on the same one-table-per-enum rule T1 followed for `FarFieldBasis`.
- **`mac_theta` is not validated** even though Canopy documents `0 < theta < 1`.
  Step 4 names `ncrit`, `max_depth` and `softening` and the option surface is
  closed, so the scope was not widened.
- **Validation is split across two places** and deliberately: `ncrit` and
  `max_depth` are checked in the constructor, because they need no
  `ZModelParams` and the earliest failure is the best one, while `softening` is
  checked in `buildConfig` at the first evaluation, because that is the first
  moment `blob()` exists.

### What the Canopy headers turned out to say

- **`Solver::gradient()`'s type is arm-independent.** `gradient_view_type` is
  `Kokkos::View<Scalar*[NComps][3], MemorySpace>`, which at `Scalar = double`,
  `NComps = 3` is the same type for every basis and order. So are
  `TreeBuilder<MS,ES>` and `CommunicationPlan<MS,ES>`. Only `DownwardSweep`
  carries the basis. `Impl` therefore exposes `gradient()`, `builder()` and
  `commPlan()` as concrete types and only `readDiagnostics` is genuinely
  arm-dependent — which is what lets the pack, both contractions, the whole
  round trip and the P2P pair count be written **once** in the outer class
  instead of six times. This is the single biggest simplification available on
  this path and it is not obvious from the design document.
- **`FmmConfig` has no `order` and no `basis`, as T1 recorded**, and confirmed
  here: `P_ORDER` and the `template <class, int, int> class FarField` parameter
  are both on `Solver` and on `createSolver`. `LaplaceKernel<Scalar, P, NComps>`
  and `CartesianTaylorBasis<Scalar, P_ORDER, NComps>` both match that
  template-template signature, so one `ImplFor` template covers both bases.
- **The R2 guard that is actually available is weaker than "assert no arm relies
  on the default".** `ImplFor` asserts
  `is_same<solver_type::kernel_type, FarField<Real, P_ORDER, NCOMPS>>`, which
  fires if the `FarField` argument is ever dropped from the `Solver`
  instantiation — for five of the six arms. For the `SolidHarmonic` arm it is
  satisfiable by an omission, because that arm *is* Canopy's default. No
  compile-time construct can distinguish an explicitly written `LaplaceKernel`
  from a defaulted one. What bounds the risk is that `Canopy::Solver` is named
  in exactly one place in Beatnik, inside `ImplFor`, so there is one site where
  the argument could be dropped and the assert covers it for every arm whose
  basis is not the default. Recorded on the declaration.
- **`Canopy::MortonKey` is `uint64_t`** and `CellInfo` carries `global_count`
  per cell, globally replicated, so the P2P pair count is a host-side
  `unordered_map<MortonKey, long long>` over `builder().cells()` and a walk of
  `commPlan().p2p_plan().neighbor_lists`, which is keyed on this rank's **owned**
  leaves only — hence no double counting across ranks.
- **Nothing in the Canopy headers contradicted the design document.** The
  gradient's sign and shape, `setup`'s "count BEFORE migration" contract,
  `auto_maintain`'s three-way return, the `-1` auto-softening sentinel, the
  `r2 < 1e-24` P2P skip and the accessor set are all as described.

### The compile-time cost of the six arms

Measured as a controlled pair: two full builds of the **same 29 translation
units**, each from a `spack clean beatnik` so CMake reconfigured and no object
was reused, on tuolumne's login node.

| build | wall | **CPU (user)** |
| --- | --- | --- |
| `~canopy` (no Canopy at all) | 3m43 | **30m25** |
| `+canopy` (six arms) | 8m29 | **77m59** |

**+47.5 CPU-minutes, a factor of 2.56.** CPU time is the figure to quote; the
login node is shared and wall time is correspondingly noisy — an earlier run of
the identical `+canopy` work came in at 8m35 / 78m31, which is a useful
cross-check on the CPU number and a warning about the wall one. Each arm
instantiates `TreeBuilder`, `TreePartitioner`, `CommunicationPlan`,
`UpwardSweep`, `DownwardSweep` and `P2P`, in every translation unit that creates
a BR solver, and `nm -C` finds **289** defined symbols per `CartesianTaylor` arm
and **290** for the `SolidHarmonic` one in the installed
`adaptive_mesh_bubble`. So a seventh arm costs roughly 8 CPU-minutes of build
per translation-unit set, which is the number to weigh when **T5** wants another
scan point.

**Affects:** **T3** — the surface to call is `evaluateVelocity( sources,
strengths, params, velocity )` after `quadrature.generate`, with no `targets`
argument, no separate `setSources`, no `blob` argument and no `br_sign` to
apply afterwards; the adapter also rejects a non-`Vertex` quadrature, so T3's
own error handling need not. **T4** — `BRSolverFMM::farField().diagnostics()`
is the read path, and the fields are named in `FarFieldDiagnostics`; note which
are `global_` (reduced) and which `local_` (this rank only), because they are
not interchangeable. T4 is also the first execution of *anything* here: the
round trip, both contractions, both prefactors and the `~canopy` instantiation
gap above are all unexercised, and T4's rank sweep at 1-6 is what makes the
self-validating forward distributor and the round-trip completeness throw
worth anything. Its negative case at `order = 0` is built, and so is the
`SolidHarmonic` arm at 3. **T5** — the scan axes it may set are
`(basis, order)` over the six built arms and nothing else; a seventh arm is one
line plus roughly 8 CPU-minutes of build, and the dispatch says so.
`p2p_pair_fraction`, `global_m2l_fallback_pair_count` and
`local_m2l_unique_op_count` are already computed at every evaluation, so no
scan point needs new plumbing to report its qualification list. **T7** — the
Riesz path is `evaluateRieszScalar`, which applies `-1/4pi^2` and deliberately
does **not** apply `br_sign`; it is a second `solve()` on the same tree, not a
re-read. **T8** — the `MaintenanceAction` histogram it wants is one field per
evaluation (`FarFieldDiagnostics::maintenance`) and `local_m2l_op_keys_built`
is the per-build rebuild signature R12 asks for; the adapter adds two
`Cabana::Distributor` builds plus a claim migrate, one `MPI_LAND` and one
five-value `MPI_SUM` per evaluation on top of Canopy's own, and the fallback to
`setup()` on a changed source set is a cost that lands on every remeshing step.
Also for **T8**: the six arms' 2.56x compile cost is measured above and is the
maintenance-policy input that section wants.

## T3

`BRSolverFMM::computeInterfaceVelocity` is live and `--br-approximation fmm`
runs end to end. The body is eleven lines of code and the rest of the change is
comments and documentation, which is the point: T2 put every hard part in the
adapter, and T3's job was to not re-do any of it.

```cpp
point_view points;
strength_view strengths;
quadrature.generate( mesh, geometry, state, points, strengths );
_far_field->evaluateVelocity( points, strengths, params, velocity );
```

Two typedefs (`point_view`, `strength_view`, both pulled from
`quadrature_type` exactly as `BRSolverDirect.hpp:60-61` does) were added to the
class; the three view types involved are all
`Kokkos::View<Real*[3], Kokkos::Device<ES,MS>>` and needed no conversion.

**Decisions taken as given by the task, recorded so they are not reopened.**

- **`BRSolverFMM` applies neither prefactor and does not size the output.**
  `FarFieldSolver::evaluateVelocity` reallocates `velocity` to the source count
  and zeroes it, and applies `br_sign/4pi` itself. Duplicating either is a
  silent double application, not a redundant safety net. `BRSolverDirect`'s
  realloc/zero/coefficient lines are what the *adapter* mirrors, not what
  `computeInterfaceVelocity` reimplements — the comment in the body says so, so
  the next reader does not "fix" the apparent omission.
- **`BRSolverFMM` adds no quadrature guard.** The adapter already rejects any
  `source_quadrature` other than `Vertex` with a `std::runtime_error` naming
  the rule in force. A second check duplicates the error surface.
- **Level 3 only, and the far field is not live there.** 642 vertices at
  `ncrit = 64` is two orders below the README's liveness inequality
  (`N >> 105*ncrit = 6720`), so the solve is expected to be all or nearly all
  P2P. **A passing T3 is not far-field evidence** — not of the expansion, the
  basis selector, the M2L path or the acceptance criterion. It is a
  completes-without-throwing check. The fmm-vs-direct agreement at this
  configuration was deliberately not computed as a number, so that no later
  reader can quote one.
- **`tasks/framework.md` R6 is out of scope.** The `@note` previously pointed
  there; it now points at this document instead. framework.md was not touched.
  **Its "second-order concern" paragraph (`tasks/framework.md:2350-2354`) is
  superseded** in exactly the way the `@note` was — both were written against a
  bare-kernel far field, and under `CartesianTaylorBasis` the blob is inside
  the expansion at every order. A framework.md session can retire it.

### The exit criterion's direct half could not be checked the way it was written

The task says to `h5diff` the pre- and post-change direct checkpoints, "same
binary configuration, same flags, so any difference is a regression." `h5diff`
reported differences — and they are not a regression. **The run is not bitwise
reproducible.** Two submissions of the identical command line against the
identical binary (jobs `f3Zr8Ew4dXR9` and `f3Zr8F53BfLj`, labels `repeatA` and
`repeatB`) differ by **1.15e-15 (np1)** and **1.88e-15 (np4)** of field RMS,
worst at `/vertices/u1`. Step 0 — the initial condition, written before any
solve — is bitwise identical every time, which localizes the nondeterminism to
the timestep rather than to I/O or mesh generation.

So the bitwise compare fails on an *unmodified* binary and cannot distinguish a
regression from a rerun. Replaced with the right yardstick: max|diff|
normalized by the field's own RMS, compared against that measured noise floor.

| comparison | np1 | np4 |
| --- | --- | --- |
| same binary, two runs (the noise floor) | 1.15e-15 | 1.88e-15 |
| baseline vs. after, `direct` | **1.54e-15** | **1.03e-15** |

Baseline vs. after is the same size as the noise at np1 and *smaller* than it
at np4, so there is no detectable change to the direct path. Worth stating
plainly because the naive reading is available and wrong: **`h5diff` exiting 1
on these trees is the expected outcome, not a finding.** Anything downstream
that wants a bitwise checkpoint comparison needs to know this first.

One caveat on the baseline, stated rather than hidden. The baseline binary was
the installed one from before this task (configure stamp `82add00`, T1's
commit) and the after binary is stamped `c8fe19f` + T3's edits, so the two
differ by T2's source as well as T3's. That was left alone deliberately: T2
touched `Beatnik_FarFieldInterface.hpp`, `CMakeLists.txt` and README, none of
which is on the direct path, and a *clean* comparison across both commits is a
stronger result than one across T3 alone. Had the comparison come back dirty,
the tighter baseline (rebuild unmodified HEAD, re-run) would have been the next
step; it did not, so it was not spent. The checkpoints carry no version or SHA
attribute, so the rebuild itself cannot show up as a difference.

### R3: the round trip's first execution neither dropped nor duplicated a source

The task notes that the round trip, both `Cabana::Distributor` legs and the
contraction all execute for the first time here. They did, and they were clean
— no bug surfaced and nothing needed debugging, so **no `farField()`
diagnostics were read**; the example driver has no path that prints them and
adding one is T4's, not a side errand here.

The cheap probe that *was* available: a dropped or duplicated source presents
as a field that changes with the rank count (R3, and the same signature as the
R9 ghost-emission bug). Matching rows by `/vertices/gid` — necessary because
checkpoint row order follows the decomposition, so a raw array compare across
rank counts is meaningless and reports garbage — over all four checkpoint
files:

| np1 vs np4, gid-matched | worst normalized difference |
| --- | --- |
| `direct` (the control) | 2.224e-15 |
| `fmm` | **2.224e-15** |

Identical, and the `gid` sets match exactly at both rank counts, so every
source was present exactly once on both sides. The FMM's rank-count spread is
indistinguishable from the direct solver's. **What this does and does not
cover:** it exercises the forward distributor, the tag-reverse scatter, the
round-trip completeness check and the curl contraction — which is precisely
what R3 is about — and it exercises them at a vertex count where the M2L far
field is almost certainly not engaged. It is evidence about the round trip, not
about the expansion. T4's 1-6 sweep is still what R3 needs.

The self-validating forward distributor and the round-trip completeness throw
that T2 added on its own initiative both stayed silent, which is the correct
outcome at a frozen connectivity (`--no-dynamic-remesh --refine-every 0`) and
says nothing yet about the remeshing path that motivated them.

### Documentation

The `@note` on `computeInterfaceVelocity` now attaches the basis to the
softened-kernel claim rather than asserting it unconditionally: true under
`FarFieldBasis::CartesianTaylor`, false under `SolidHarmonic`, which is what
Canopy's template parameter defaults to. The "acceptance criterion tuned on the
bare kernel is optimistic near self-contact" claim is replaced by what actually
binds under the softened basis — Taylor truncation at the accepted `R/w`, going
as `(cw/R)^p` on the **gradient**, one order worse than on the potential, with
`FmmParams::order` as the knob and not `mac_theta`. The matching paragraph in
`Beatnik_FarFieldInterface.hpp` got the same correction plus the R2 pointer.
README gained a blockquote saying outright that `fmm` is the *default*
(`Beatnik_Params.hpp:107`, because the Python default is `treecode`) and is not
the validated path, that it runs but has no measured accuracy, and that runs
whose numbers matter should pass `--br-approximation direct` until T5. The
`basis` table row's unmeasured claim that `CartesianTaylor` "is the validated
production path" was corrected to *intended* production path.

Four stale citations in T3's own entry were corrected in `add-canopy.md`
(`BRSolverDirect.hpp:105-166`→`:105-165`, `:112-115`→`:111-115`,
`:126-127`→`:125-127`, `FarFieldInterface.hpp:36-41`→`:37-44`), the `@note`
citation was repointed from a nonexistent file-header range to `:104-109` and
the count corrected from two blocks to one, and
`src/Beatnik_FarFieldInterface.hpp` was added to the **Fill in** list, which
omitted a file step 2 requires editing.

### Cost and mechanics

`spack install` after touching the example driver (the header-only-rebuild
caveat in `systems/tuolumne/claude.md` — an INTERFACE library does not track
`HEADERS_PUBLIC`, so a header-only change can report a sub-second no-op):
**9m25s wall** on the login node, one reconfigure. The new
`scripts/tuolumne/t3_fmm_velocity.flux` is mode-parameterized
(`LABEL [MODE...]`) rather than FMM-only, which is what makes the before/after
command lines byte-identical by construction instead of by careful copying —
the log echoes each full command line, and the two submissions' `direct` lines
differ only in the checkpoint directory. `-t 10m`, `-q pdebug`, one node; four
launches fit comfortably. `flux batch --flags=waitable` is **refused on this
instance** ("only the instance owner can submit with FLUX_JOB_WAITABLE"), so
`flux job status <jobid>` is the wait mechanism here, as the fallback in the
task description anticipated.

**Affects:** **T4** — the round trip, both distributor legs and the curl
contraction now have one clean execution at 1 and 4 ranks behind them, so T4
inherits a working path rather than a first bring-up; its 1-6 sweep is still
where R3 is actually settled, since 1-vs-4 at 642 vertices exercises the tag
plumbing but almost certainly not the M2L far field. Two things T4 must plan
around: **the timestep is not bitwise reproducible** (1-2e-15 of field RMS
run-to-run on the same binary, localized to the solve rather than to I/O), so
no test may assert bitwise equality of anything downstream of a timestep, and
any tolerance must clear that floor; and **`ncrit` or the vertex count must be
chosen so the far field is live** (`N >> 105*ncrit`), or the comparison passes
at round-off no matter what the expansion does — R1's cheapest misreading, and
T3's configuration is squarely inside it. T4 is also still the place to close
T2's `~canopy` instantiation gap with a guarded explicit instantiation.
**T5** — the same liveness constraint bounds its scan configuration from below,
and the noise floor above is the resolution limit on anything it measures
through a full timestep rather than through a single evaluation. **T7** —
`computeSurfaceRieszScalar` is untouched and still throws; the call-path shape
it will need is settled by this task (generate through `generateGradient`, one
adapter call, no prefactor and no sizing on the `BRSolverFMM` side), and
`--bernoulli-scalar-mode normal-speed` is what keeps it out of the path today.
**T8** — the 9m25s rebuild is a per-change cost on this path, and the
maintenance actions the four launches took were not recorded, so T8 starts its
histogram from zero.

## T4

The FMM velocity was compared against the direct velocity on the same state and
**it agrees to 5.008e-4 of the field's own scale at every one of ranks 1-6**,
against a 2.0e-3 budget, with the far field asserted live rather than assumed.
`Beatnik_Test_FmmVsDirect` is a new `unit`-tier member (41 checks at multi-rank,
39 at one rank, job `f3aDVgMVBhpj`) and `scripts/tuolumne/t4_fmm_vs_direct.flux`
is the two-node sweep that runs it at 1-6. The tier is green at one rank too,
7/7 (job `f3aDWPshh62K`). The gate is untouched: five `regression` members, 60
launches.

**Decisions taken as given by the task, recorded so they are not reopened.** The
rank sweep runs through the new script rather than through the tier runner or
`ctest` — the tier registers every member at one rank, spack mode has no build
tree and therefore no `ctest`, and `unit_tests.flux` pins a one-node allocation
while deriving `ceil(n/4)` nodes per launch, so `BEATNIK_UNIT_RANKS=5` asks for
two nodes inside one. The script targets the **development** spack env
(`BEATNIK_USE_PROD` unset), as `t3_fmm_velocity.flux` does: this is iterative
test work, not a large queued run. `FmmConfig` is not reachable from a test by
construction, so step 8's assertions are on `diagnostics().softening` and
`params().near_softening_factor` and no accessor was added.

### The chosen pair, and the far field that resulted

**Subdivision level 4 (2562 vertices) with `ncrit = 8`**, which the design
document picks and which the measurement confirms: the realized
**`p2p_pair_fraction` is 0.2537** (0.253649-0.253651 across the six rank
counts), against the compiled bound of 0.75 and against 1.0 for a solve with no
far field at all. So **three quarters of the pairs go through M2L** and the
comparison is a measurement of the expansion rather than of two direct sums —
**R1**'s cheapest misreading, and the one T3's level-3 configuration sits
squarely inside. `global_m2l_pair_count` is **130968** and
`global_m2l_fallback_pair_count` is **0** at every rank count, so the accuracy
number below is one code path and not a mixture (**R6**); the operator table did
not overflow at this configuration.

The a-priori estimate the pair was chosen from held: 320 occupied leaves against
a 105-leaf near field predicts a P2P fraction near 1/3, and 0.2537 is that.

### The budget, with its qualification list

**2.0e-3**, on the **velocity** — i.e. the *gradient* of Canopy's potential, one
order worse than the potential at fixed `order`, which is why naming the field
is not decoration. Source distribution: the milestone-0 icosphere at subdivision
level 4, 2562 vertices, after 5 `--br-approximation direct` steps (the sheet
strength is identically zero at step 0 under
`--initial-potential-strength 0`, and the test asserts
`max |A·S| = 3.289e-6 > 0` before comparing anything). Rank counts 1, 2, 3, 4,
5, 6. Basis `CartesianTaylor`. `order = 3`. `ncrit = 8`. `max_depth = 10`.
`mac_theta = 0.3`. `softening = 0.025` (= sqrt(blob) at `--eps 0.025` under
`--kernel-blob-mode length`). `near_softening_factor = 0`. Realized P2P pair
fraction **0.2537**.

**It is not τ_A.** T5 measures that and no T5 number exists. The value is set
from Canopy's own measured gradient errors — 7.0718e-4 at p=3 and 8.996e-3 at
p=2, θ=0.3, on its volumetric cloud — which is also the provenance of
`FmmParams::order = 3`: 2.0e-3 is 2.8x above the p=3 figure (headroom for a
2-manifold rather than a cloud) and 4.5x below the p=2 figure (the room the
order-2 negative case needs). It clears T3's measured run-to-run noise floor of
1.15e-15/1.88e-15 by twelve decades, so nothing here is a bitwise claim wearing
a tolerance.

### Per-rank results

Error is `max|u_fmm − u_direct|` over owned rows, divided by
`max|u_direct| = 5.15236e-3` — a value identical to six digits at every rank
count.

| ranks | max abs error | relative | p2p fraction | particles |
| --- | --- | --- | --- | --- |
| 1 | 2.58040e-6 | **5.00819e-4** | 0.253650 | 2562 |
| 2 | 2.58001e-6 | **5.00743e-4** | 0.253651 | 2562 |
| 3 | 2.58019e-6 | **5.00778e-4** | 0.253650 | 2562 |
| 4 | 2.58041e-6 | **5.00822e-4** | 0.253650 | 2562 |
| 5 | 2.58099e-6 | **5.00934e-4** | 0.253650 | 2562 |
| 6 | 2.57993e-6 | **5.00728e-4** | 0.253651 | 2562 |

Four times inside the budget, and the **spread across the six rank counts is
2.1e-7 absolute, 4.1e-4 of the error itself**. That is not merely small, it is
*at the nondeterminism floor*: the earlier sweep of the same binary (job
`f3aDRgAiDSjh`, identical except for the test's own basis constant) measured
5.00765e-4 at np=1 against this one's 5.00819e-4, a run-to-run difference of
1.1e-4 of the error on one rank count — the same order as the whole
rank-to-rank spread. **The rank count contributes nothing detectable**, which is
the result **R3** wanted: a dropped or duplicated source produces a field wrong
by a factor that *changes with the rank count*. `global_particle_count` is
exactly 2562 at every rank count, the independent discriminator, and the
adapter's own round-trip completeness throw and self-validating forward
distributor both stayed silent.

**R4 did not fire.** np=4 is 5.00822e-4, sitting inside the same 2.1e-7 band as
every other rank count and above np=1's 5.00819e-4 by 3e-9. Canopy's np=4
three-component gradient defect is under `LaplaceKernel` at P=8 with
`softening = 0`; this path
runs `CartesianTaylorBasis` at p=3 with `softening = 0.025`, and at this
configuration the defect is not reachable. That **bounds** it further — it does
not retire it, and the sweep must keep 4 in.

### The three negative cases

All three fail as required, and each names its own reason in the log rather than
merely exiting non-zero.

1. **The order knob is live, and the error is a truncation.** `order = 0` gives
   **0.407338** relative — 204x the budget, and identical to six digits at every
   rank count — `order = 2` gives **5.9658e-3**, and `order = 3` gives 5.008e-4,
   so the three are strictly ordered and the middle one lands between the other
   two. **e2/e3 = 11.91** (11.912-11.914 across
   ranks), against Canopy's own 12.7 on its cloud: a decade, which is the
   signature of the gradient truncating at p rather than at p+1. A *bias* would
   have put e2 ≈ e3 and passed a check that only looked at order 0.
2. **The basis selector is live.** `SolidHarmonic` at
   `near_softening_factor = 0` gives **4.6582e-3 to 4.6989e-3** — it misses the
   budget by 2.3x and sits **9.30x to 9.38x above the production arm at the same
   order**,
   so the gap is the kernel and not the truncation. The two arms are not on top
   of each other, which is what **R2** needs.
3. **The blob reaches the far field.** Doubling the softening length from 0.025
   to 0.050 changes the FMM velocity by **0.139041** relative, and the perturbed
   FMM still tracks the *direct solver at that softening length* to **5.439e-4**
   — inside the same budget. The second half is the load-bearing one: a
   bare-kernel far field would move (the near field alone would see the change)
   but would not still agree with a fully softened direct sum. The adapter's
   softening-stability guard was exercised too and **threw** as designed when a
   live solver was presented the original blob after being built at the
   perturbed one.

### R2's "tens of percent" is a self-contact figure, not a property of this path

**The one place the document's prediction did not survive contact, and the one
constant this task revised.** The test first compiled the basis separation as
"the solid-harmonic arm must exceed the budget by 100x", straight from **R2**'s
statement that a bare-kernel far field with no near-field floor is wrong by tens
of percent. It failed at every rank count — and at only that one check (job
`f3aDRgAiDSjh`: 37/38 at np=1 and 39/40 at multi-rank, with everything else
green). The measurement is 4.66e-3, not 0.2.

The prediction is not wrong; it is about a different geometry. Canopy accepts a
pair only beyond R/w > 2√3/θ = 11.55 half-widths. At 2562 vertices in a root box
about 0.6 wide the occupied leaves sit at depth 4-5, so w ≈ 0.019 to 0.0094 and
the closest accepted separation is R ≈ 0.11 to 0.22 — **four to nine times the
softening length √b = 0.025**. The bare and softened kernels differ there by
(3/2)·b/R², i.e. 2% to 8% on the closest accepted pairs and less beyond, and
those pairs carry a minority of a field whose near half both bases evaluate
identically through P2P. A whole-field separation in the 1e-3..1e-2 band is
exactly what that predicts, and 4.66e-3 is it.

**"Tens of percent" therefore describes a self-contacting sheet**, where
separations approach √b — which is develop-canopy's full-roll-up NaN and not a
smooth sphere five steps off its initial condition. The constant was changed to
a ratio against the production arm (5.0, a factor of two below the observed
decade) *plus* the independently-derived requirement that the arm miss the
budget outright. **`kVelocityBudget` was not touched**, and the production arm
passes it with 4x of margin either way.

### A rank owning zero sources was not reachable from the mesh, so it was built

The mesh decomposition never produces one at this vertex count: the minimum
owned count is 2562, 1233, 807, 595, 469 and 386 at ranks 1 through 6. Rather
than leave the case uncovered, the test constructs it at the adapter's own
interface — the last rank ships its rows to rank 0 over MPI and calls
`evaluateVelocity` with an **empty** source list, so the global source set is
byte-identical and only its distribution changed.

Every rank returned, the global particle count stayed 2562, and rank 0's own
rows came back differing from the baseline field by **1.3e-18 to 4.1e-18
absolute, 2.5e-16 to 8.0e-16 relative** — round-off, six decades inside the
1e-9 tolerance and six decades below the budget. Canopy has no test for a
zero-particle rank and the adapter's file comment says so; at 2-6 ranks on this
path it handles one.

### Departures from T4's Do steps

- **Step 6 was widened, not skipped.** The step says to include a zero-source
  variant "if the decomposition can be made to produce one … if it cannot, say
  so in the log rather than leaving the case silently uncovered". It cannot, and
  this says so — and then covers the case synthetically anyway, which is
  strictly more than the step asks for and is worth more than a sentence of
  apology.
- **The `~canopy` half of step 9 does more than instantiate.** The step asks for
  one guarded explicit instantiation of `FarFieldSolver`. The file carries that,
  and in a `~canopy` build `runChecksNoCanopy` additionally asserts that
  `BRSolverFMM::computeInterfaceVelocity` throws a `std::runtime_error` whose
  text names `BEATNIK_ENABLE_CANOPY`. Without it the member would be vacuous in
  that build — it would instantiate the template and check nothing. **Not
  verified by running:** this machine builds `+canopy` and no `~canopy` build
  was made for this task, so that branch is compiled-in and unexecuted.
- **`m2l_fallback == 0` is asserted, not merely reported.** **R6** says a
  non-zero fallback count makes the accuracy figure a mixture of two code paths.
  A mixture is not the number this test claims to have measured, so it is a
  check.
- **The test writes no checkpoints**, so `BEATNIK_TEST_SCRATCH` is deliberately
  not set anywhere in the script: `makeSpinUpParams` leaves the checkpoint
  directory empty and `CheckpointIO` is a no-op without one. Nothing in this
  member touches a filesystem.

### Bugs only running revealed

**None in the code under test.** Nothing in the adapter, the round trip, the
dispatch or `BRSolverFMM` needed a change: the only edit after the first sweep
was to the test's own basis-separation constant, described above. The two
guards T2 added on its own initiative — the self-validating forward distributor
and the round-trip completeness throw — both stayed silent at every rank count,
which is correct at a frozen connectivity and still says nothing about the
remeshing path.

### Cost and mechanics

`spack install` after adding one test source: **11m17s wall, 88m40s CPU** on the
login node (the new translation unit instantiates all six Canopy arms, which is
T2's measured ~8 CPU-minutes per arm-set); a second install touching only that
source took **4m55s**. The sweep itself is cheap: the whole 1-6 job — six
launches, 21 ranks of work — ran well inside `-t 25m` on two `pdebug` nodes.
`flux batch --flags=waitable` is still refused on this instance, so
`flux job status <jobid>` remains the wait mechanism.

**Affects:** **T5** — the scan can start from a configuration that is known to
have a live far field, and the numbers it inherits are (2562, `ncrit = 8`) with
a realized P2P fraction of 0.2537, a p=3 gradient error of **5.008e-4** on
Beatnik's own sheet (against Canopy's 7.07e-4 on its volumetric cloud, so the
sheet is *slightly better*, not worse) and e2/e3 = 11.9. It also inherits a
resolution limit: the spin-up state is downstream of five timesteps, so any
figure taken this way carries a run-to-run floor of about 1e-4 **of the error**,
and a scan that wants to resolve finer differences than that must either
re-evaluate on one state in one process or pin the state some other way.
`max_depth = 10` was never approached: the fallback count is 0 and the operator
table did not overflow at this configuration, so **R6** has no evidence to act
on yet and T5's
revision of `max_depth` starts from "not binding here" rather than from nothing.
The basis question T5 must report on is reshaped: at a smooth sphere the
solid-harmonic basis costs about a decade (4.66e-3 against 5.01e-4), not the
orders of magnitude R2 predicts, and T5 should scan a *deformed* state if it
wants to see the figure R2 is actually about. **T6** — claim A now has a
measured per-evaluation bound on Beatnik's own geometry at level 4, which is the
L4 member's configuration; the L3 member's far field is still not live at any
`ncrit` and this task changed nothing about that. **T7** — the Riesz path
inherits a round trip exercised at 1-6 ranks, with a zero-source rank, and a
softening guard it must meet the same way; `computeSurfaceRieszScalar` is still
the only throwing virtual, and the test file is where its case goes (the
`ArmResult`/`evaluateArm` shape generalizes to a scalar output). **T8** — the
maintenance action was `Setup` on every evaluation here, because each arm builds
a fresh solver and evaluates once, so this task contributes **no** histogram
data; T8 still starts from zero. The 11m17s/88m40s rebuild is the per-change
cost on this path.


## T5

**τ_A is `5.01e-4` on the velocity (gradient), at `order` 3, and the far field
was live when it was measured.** The scan is 144 arms over six launches, all six
green (84/84 structural checks each), **zero fallback pairs and zero non-finite
rows at every arm**. Steps 1-6 landed as written. Step 7 did not: it uncovered
a defect in the *measurement instrument* that makes T6's stated plan
unexecutable as written, and that finding is the most consequential thing in
this entry.

**No tolerance was compiled into any test by this task.** `Beatnik_Test_FmmScan`
carries none and may not; its only assertions are structural (entity counts,
global particle count, finiteness, and that an arm's two evaluations saw the same
tree). T6 is what compiles τ_A. That is R1's discipline and it held.

### Decisions taken as given by the task, recorded so they are not reopened

- **The whole scan runs in one launch against one shared spin-up state.** The
  spin-up is downstream of five timesteps and is not bitwise reproducible, so
  cross-launch figures carry a floor of about `1e-4` *of the error*; within a
  launch every arm sees byte-identical state and arm-to-arm differences are
  exact. Implemented, not re-derived. The measured cross-launch spread at the
  production point is **`2.9e-4` of the error** across all six launches and
  **`2.1e-4`** between HIP np1 and np4 alone, so the floor is real and is the
  resolution limit on every figure this entry reads across rows.
- **The basis comparison is reported on the smooth sphere only.** R2's "tens of
  percent" is a self-contact figure and no milestone-0 configuration in this
  tree reaches self-contact. **The self-contact figure is unmeasured here** and
  no run was spent hunting for a deformed state.
- **The published horizon envelope is FMM-driven against the in-tree Python gold
  set**, with a direct-driven 2000-step run reported alongside as attribution.
  Both were run. See step 7 for why the envelope is reported under a different
  instrument than the one the task named.

### Signatures changed

Three files outside T5's stated **Fill in** list changed, each because the exit
criterion cannot be met without it.

1. **`src/Beatnik_FarFieldInterface.hpp` gained `evaluatePotential`.** Step 2
   requires a separate potential column at every scan point and **there was no
   route to one**: Beatnik's two physical far-field reads, `evaluateVelocity`
   and `evaluateRieszScalar`, are both contractions of Canopy's *gradient*
   tensor, so neither can see the potential. The new method adds
   `Contraction::Potential` (which is not a contraction at all -- it copies
   `Impl::potential()` component-wise) and **applies no prefactor**, because
   there is no Beatnik quantity it is the far field of. It is a measurement
   surface; no physics calls it. The `Contraction` enum's doc comment and the
   `evaluate()` header were corrected to say "three outputs" rather than "either
   contraction".
2. **`FarFieldDiagnostics` gained `local_m2l_bytes_per_key` and
   `local_m2l_op_cap`.** Step 5 has to report "the bytes it occupies" beside the
   key count. `bytes_per_key` is a `static constexpr` on the Canopy basis and
   `m2l_effective_op_cap()` is on the sweep, so both are read there and carried
   out rather than re-derived from `order` in a consumer -- an arithmetic
   re-derivation would be a second source of truth for a number whose whole
   purpose is to be checked against Canopy's cap.
3. **`Beatnik_Test_Milestone0Run.cpp` took `argv[4..6]`** --
   `direct`|`fmm`, `ncrit`, `order`. `:184`'s pinned
   `BRApproximation::Direct` is now the **default**, so every M0-D1 invocation
   of the driver still means exactly what it meant, and an unrecognized argv[4]
   is a recorded failure rather than a silent fall back to `direct`. Under `fmm`
   the configuration is appended to the output directory name so a direct and an
   FMM run of the same level and rank count cannot alias inside one scratch.

New files: `tests/regression_tests/Beatnik_Test_FmmScan.cpp` (registered in the
"Measurement drivers -- IN NO TIER" loop, so it gates nothing),
`scripts/tuolumne/t5_fmm_scan.flux`, `scripts/tuolumne/t5_divergence.flux`, and
`tests/regression_tests/fmm_divergence_ladder.py` -- the last of which is step
7's finding made usable and is justified below.

### The scan: what was varied, and what was not

**`(basis, order)` is the only axis limited to what T2 built** --
`CartesianTaylor` at 0, 2, 3, 4, 5 and `SolidHarmonic` at 3 only. `mac_theta`,
`ncrit` and `max_depth` are runtime `FmmParams` members and scanned freely. The
scan is **five named axes through one background** (T4's measured-live
configuration: `CartesianTaylor`, p=3, `ncrit` 8, θ=0.3, `max_depth` 10) rather
than a Cartesian product, which would be 625 arms to answer four
one-dimensional questions. 24 arms per launch; launches are
(L4, L3) x (HIP np1, HIP np4) plus Serial np1 at each level.

Job `f3aPSQYSVTpb`, `scripts/tuolumne/t5_fmm_scan.flux`, one `pdebug` node,
whole sweep **82 s** of launch wall (9-29 s per row). Commit `41297bd` plus this
task's working tree; dev spack env; `spack install` **10m33s wall / 102m42s
CPU** on the login node for the full rebuild the adapter header forces.

**Every figure below is `max|fmm - reference|` over owned rows divided by the
reference field's own max magnitude**, and carries: source distribution
(milestone-0 icosphere at the stated level, after 5 `--br-approximation direct`
steps), the rank count and backend of its row, the basis, `order`, `ncrit`,
`mac_theta`, `max_depth`, `softening = 0.025` (= sqrt(blob) at `--eps 0.025`
under `--kernel-blob-mode length`), `near_softening_factor = 0`, and the
realized P2P pair fraction. Field scales: `max|u_direct| = 5.15236e-3` and
`max|phi_direct| = 7.99705e-3` at L4; `5.10431e-3` and `7.67777e-3` at L3.

**The two references are not the same kind of object.** The gradient column is
against `BRSolverDirect` on the same mesh, geometry, state and quadrature --
T4's reference, unchanged. The potential column is against an O(N^2) sum written
in the driver over the globally gathered source set, following **Canopy's**
exclusion rule and not Birkhoff-Rott's: `Canopy_P2P.hpp` skips `pj == pi` and
any pair with `|r|^2 < 1e-24`, and a reference including the self term would
differ from the FMM by `S_t/sqrt(b)` -- a factor of 40 at this softening -- at
every target. The softening squared is taken as
`diagnostics().softening^2`, the value Canopy itself squared, rather than
re-derived from `blob()`.

#### Level 4 (2562 vertices), HIP, 1 rank -- the production row

| axis | basis | `order` | `ncrit` | `mac_theta` | `max_depth` | gradient rel | potential rel | P2P frac | M2L cell pairs | fallback | keys (rank 0) | table MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order | `cartesian-taylor` | 0 | 8 | 0.3 | 10 | 4.07338e-01 | 5.70071e-02 | 0.253650 | 130968 | 0 | 10902 | 0.08 |
| order | `cartesian-taylor` | 2 | 8 | 0.3 | 10 | 5.96723e-03 | 6.82607e-04 | 0.253650 | 130968 | 0 | 10902 | 8.32 |
| order | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| order | `cartesian-taylor` | 4 | 8 | 0.3 | 10 | 5.60924e-05 | 1.96748e-06 | 0.253650 | 130968 | 0 | 10902 | 101.89 |
| order | `cartesian-taylor` | 5 | 8 | 0.3 | 10 | 8.37613e-06 | 5.11766e-07 | 0.253650 | 130968 | 0 | 10902 | 260.84 |
| basis | `solid-harmonic` | 3 | 8 | 0.3 | 10 | 4.69909e-03 | 8.30607e-04 | 0.253650 | 130968 | 0 | 9766 | 23.84 |
| ncrit | `cartesian-taylor` | 3 | 64 | 0.3 | 10 | 1.75479e-04 | 1.33432e-05 | 0.854134 | 5736 | 0 | 864 | 2.64 |
| ncrit | `cartesian-taylor` | 3 | 32 | 0.3 | 10 | 1.75479e-04 | 1.33432e-05 | 0.854134 | 5736 | 0 | 864 | 2.64 |
| ncrit | `cartesian-taylor` | 3 | 16 | 0.3 | 10 | 4.38234e-04 | 3.64903e-05 | 0.433923 | 73544 | 0 | 10776 | 32.89 |
| ncrit | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| ncrit | `cartesian-taylor` | 3 | 4 | 0.3 | 10 | 6.39201e-04 | 5.22813e-05 | 0.107565 | 333250 | 0 | 18742 | 57.20 |
| theta | `cartesian-taylor` | 3 | 8 | 0.2 | 10 | 1.15734e-04 | 7.88889e-06 | 0.566336 | 173680 | 0 | 14966 | 45.67 |
| theta | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| theta | `cartesian-taylor` | 3 | 8 | 0.4 | 10 | 1.35576e-03 | 1.78839e-04 | 0.137649 | 83952 | 0 | 6580 | 20.08 |
| theta | `cartesian-taylor` | 3 | 8 | 0.5 | 10 | 3.51459e-03 | 5.65856e-04 | 0.090420 | 58784 | 0 | 4196 | 12.81 |
| theta | `cartesian-taylor` | 3 | 8 | 0.7 | 10 | 1.31262e-02 | 1.35718e-03 | 0.049149 | 31032 | 0 | 2246 | 6.85 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 5 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 6 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 8 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 12 | 5.00775e-04 | 3.72620e-05 | 0.253650 | 130968 | 0 | 10902 | 33.27 |
| order@0.5 | `cartesian-taylor` | 2 | 8 | 0.5 | 10 | 2.28113e-02 | 3.94263e-03 | 0.090420 | 58784 | 0 | 4196 | 3.20 |
| order@0.5 | `cartesian-taylor` | 3 | 8 | 0.5 | 10 | 3.51459e-03 | 5.65856e-04 | 0.090420 | 58784 | 0 | 4196 | 12.81 |
| order@0.5 | `cartesian-taylor` | 4 | 8 | 0.5 | 10 | 8.32256e-04 | 7.03469e-05 | 0.090420 | 58784 | 0 | 4196 | 39.22 |

#### Level 3 (642 vertices), HIP, 1 rank

| axis | basis | `order` | `ncrit` | `mac_theta` | `max_depth` | gradient rel | potential rel | P2P frac | M2L cell pairs | fallback | keys (rank 0) | table MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order | `cartesian-taylor` | 0 | 8 | 0.3 | 10 | 1.11093e-01 | 2.97110e-02 | 0.854631 | 5736 | 0 | 864 | 0.01 |
| order | `cartesian-taylor` | 2 | 8 | 0.3 | 10 | 1.66340e-03 | 1.86191e-04 | 0.854631 | 5736 | 0 | 864 | 0.66 |
| order | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| order | `cartesian-taylor` | 4 | 8 | 0.3 | 10 | 2.21279e-05 | 1.59727e-06 | 0.854631 | 5736 | 0 | 864 | 8.07 |
| order | `cartesian-taylor` | 5 | 8 | 0.3 | 10 | 2.48016e-06 | 2.12351e-07 | 0.854631 | 5736 | 0 | 864 | 20.67 |
| basis | `solid-harmonic` | 3 | 8 | 0.3 | 10 | 5.91359e-04 | 5.38641e-04 | 0.854631 | 5736 | 0 | 864 | 2.11 |
| ncrit | `cartesian-taylor` | 3 | 64 | 0.3 | 10 | 2.11941e-15 | 2.82426e-15 | 1.000000 | 0 | 0 | 0 | 0.00 |
| ncrit | `cartesian-taylor` | 3 | 32 | 0.3 | 10 | 2.11941e-15 | 2.82779e-15 | 1.000000 | 0 | 0 | 0 | 0.00 |
| ncrit | `cartesian-taylor` | 3 | 16 | 0.3 | 10 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| ncrit | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| ncrit | `cartesian-taylor` | 3 | 4 | 0.3 | 10 | 4.11035e-04 | 3.85110e-05 | 0.415373 | 60720 | 0 | 10466 | 31.94 |
| theta | `cartesian-taylor` | 3 | 8 | 0.2 | 10 | 2.25764e-15 | 2.66440e-15 | 1.000000 | 0 | 0 | 0 | 0.00 |
| theta | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| theta | `cartesian-taylor` | 3 | 8 | 0.4 | 10 | 8.03907e-04 | 1.31471e-04 | 0.554090 | 12912 | 0 | 2660 | 8.12 |
| theta | `cartesian-taylor` | 3 | 8 | 0.5 | 10 | 2.22147e-03 | 4.57457e-04 | 0.361948 | 11736 | 0 | 2208 | 6.74 |
| theta | `cartesian-taylor` | 3 | 8 | 0.7 | 10 | 7.09409e-03 | 1.02836e-03 | 0.194748 | 7056 | 0 | 1316 | 4.02 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 5 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 6 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 8 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 10 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| max_depth | `cartesian-taylor` | 3 | 8 | 0.3 | 12 | 1.70348e-04 | 1.29741e-05 | 0.854631 | 5736 | 0 | 864 | 2.64 |
| order@0.5 | `cartesian-taylor` | 2 | 8 | 0.5 | 10 | 1.34503e-02 | 3.16900e-03 | 0.361948 | 11736 | 0 | 2208 | 1.68 |
| order@0.5 | `cartesian-taylor` | 3 | 8 | 0.5 | 10 | 2.22147e-03 | 4.57457e-04 | 0.361948 | 11736 | 0 | 2208 | 6.74 |
| order@0.5 | `cartesian-taylor` | 4 | 8 | 0.5 | 10 | 3.94650e-04 | 5.42453e-05 | 0.361948 | 11736 | 0 | 2208 | 20.64 |

#### The production point across all six launches

| level | backend | ranks | gradient rel | potential rel | P2P frac | M2L cell pairs | keys (rank 0) | particles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | HIP | 1 | **5.007746e-04** | 3.726197e-05 | 0.253650 | 130968 | 10902 | 2562 |
| 4 | HIP | 4 | **5.008788e-04** | 3.726207e-05 | 0.253650 | 130968 | 8047 | 2562 |
| 4 | Serial | 1 | **5.007321e-04** | 3.726340e-05 | 0.253649 | 130968 | 10902 | 2562 |
| 3 | HIP | 1 | **1.703480e-04** | 1.297408e-05 | 0.854631 | 5736 | 864 | 642 |
| 3 | HIP | 4 | **1.705479e-04** | 1.306203e-05 | 0.854631 | 5736 | 386 | 642 |
| 3 | Serial | 1 | **1.729275e-04** | 1.247836e-05 | 0.854645 | 5736 | 864 | 642 |

### Step 6 -- τ_A and the production parameter set

**τ_A = `5.01e-4`** on the **gradient** (the velocity Beatnik actually reads),
with the full qualification list: basis `cartesian-taylor`, `order` **3**,
`ncrit` **8**, `mac_theta` **0.3**, `max_depth` **10**,
`softening` **0.025**, `near_softening_factor` **0**, source distribution the
milestone-0 icosphere at subdivision level 4 (2562 vertices) after 5 direct
steps, at **HIP ranks 1 and 4 and Serial rank 1**, realized **P2P pair fraction
0.2537** (`global_m2l_pair_count` 130968, `global_m2l_fallback_pair_count`
**0**). The three launches give `5.007746e-4`, `5.008788e-4` and `5.007321e-4`.
The **potential** at the same point is **`3.73e-5`**, a ratio of **13.4** --
close to the `11.55` that "one full order better" predicts, which is the
quantitative form of the claim that naming the field is not decoration.

**The production parameter set is unchanged from T1's compiled defaults**, and
each value now has a measurement behind it rather than a derivation:

| member | value | why this value, measured |
| --- | --- | --- |
| `basis` | `CartesianTaylor` | 9.38x better than `SolidHarmonic` at the same order on this geometry. |
| `order` | **3** | The smallest that reaches 1e-3 on the gradient. p=2 is `5.97e-3`, which misses by 6x. p=4 is `5.61e-5` and **may not be adopted** (R11). |
| `mac_theta` | 0.3 | The reference's value, kept. θ=0.2 buys 4.3x accuracy at more than double the P2P share; θ=0.5 costs 7x. |
| `ncrit` | 64 | Right at production vertex counts; **wrong at milestone-0's**, which is a property of the mesh and not a defect. The liveness table below is the evidence. |
| `max_depth` | 10 | **Measured inert.** See below. |
| `near_softening_factor` | 0 | Unchanged; meaningful only under `SolidHarmonic`. |

**`order` stays 3 and the raise is pending an upstream change, not a Beatnik
one.** p=4 measures `5.61e-5` -- 8.9x better than p=3, for 3.1x the operator
table (101.89 MB against 33.27 MB at 10902 keys) -- and p=5 measures
`8.38e-6`. Neither may be adopted: p=4 reaches derivative degree |k|=8 and
Canopy's derivative-ladder oracle is validated to |k|=6, so an arithmetic error
there would present as exactly the plateau R1 describes and be attributed to
truncation. **The raise is an upstream request for another degree of oracle**
(R11), and the figures above are what it would buy.


### Step 3-4 -- reading the scan as a scan, and the error model

**The model's constant survives on a 2-manifold; its exponent in θ does not.**
The convergence model in **Problem** is
`eps_grad ~ (theta/2sqrt3)^p`, measured on a volumetric cloud at idealized
equal-cell separations. On Beatnik's sheet, at θ=0.3:

| `order` | gradient | model | measured/model | potential | per-order gain (gradient) |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.07338e-01 | -- | -- | 5.70071e-02 | -- |
| 2 | 5.96723e-03 | 7.50e-03 | 0.80 | 6.82607e-04 | -- |
| **3** | **5.00775e-04** | 6.50e-04 | **0.77** | 3.72620e-05 | **11.92** (model 11.55) |
| 4 | 5.60924e-05 | 5.63e-05 | 1.00 | 1.96748e-06 | 8.93 |
| 5 | 8.37613e-06 | 4.87e-06 | 1.72 | 5.11766e-07 | 6.70 |

**Confirmed, with one correction.** The constant is right to within 25% through
p=4 -- and to 0.3% at p=4, which is closer than the 20% the model claims on its
own cloud. So the sheet is **not** worse than the volumetric cloud the model was
fitted on; T4 already saw this at p=3 (5.008e-4 on the sheet against Canopy's
7.07e-4 on its cloud) and the whole curve now says it.

**The correction is in the exponent, and it is in θ rather than in `order`.** A
least-squares fit over θ in {0.2, 0.3, 0.4, 0.5, 0.7} at p=3 gives a realized
exponent of **3.77** on the gradient against the model's **3**, and **4.30** on
the potential against its **4**. Every adjacent-pair estimate (3.61, 3.46, 4.27,
3.92) is above 3, so this is not fit noise. The task entry names this outcome:
"a rate differing from the model in *exponent* rather than in constant would
mean the sheet's geometry changes which term dominates, and is the finding."
**It is the finding.** The mechanism is the realized `R/w` distribution: the
model assumes every accepted pair sits at the acceptance threshold
`R/w > 2sqrt3/theta`, and on a thin sheet with depth-mismatched cell pairs the
distribution has a tail well above it, which a larger θ admits faster than the
equal-cell idealization predicts.

**The per-order gain decays above the production order** -- 11.92, then 8.93,
then 6.70, against a model that predicts a constant 11.55 per order. Same
mechanism, read along the other axis: the marginally-accepted pairs are the ones
whose Taylor series converges slowest, and they come to dominate as the
well-separated pairs' contribution falls away. **This is not R1's plateau**: the
curve is still falling by 6.7x per order at p=5 and no arm flattened. The second
θ row behaves the same way -- at θ=0.5, p=2/3/4 measure `2.28e-2`, `3.51e-3`,
`8.32e-4` against a model of `2.08e-2`, `3.01e-3`, `4.34e-4` (ratios 1.10, 1.17,
1.92) with per-order gains 6.49 then 4.22 against a model 6.93.

**The potential column flattens earlier than the gradient.** Its per-order gains
are 18.3, 18.9, then **3.84** -- so between p=4 and p=5 the potential has nearly
stopped improving while the gradient is still gaining 6.7x. At `5.12e-7` on a
field of scale `8.0e-3` that is `4e-9` absolute, which is seven decades above
double-precision round-off on a 2562-term sum, so it is **not** a
floating-point floor. It is the same marginal-pair tail reaching the potential
first, which is consistent with the potential's steeper nominal rate having less
room left.

### Step 1 -- liveness, and a correction to the design document

**The design document's level-3 claim is wrong as an absolute, and the scan
measured it rather than assuming it.** [add-canopy.md](add-canopy.md) states
that at 642 vertices the liveness inequality "has **no solution**" and that
"the 642-vertex level cannot exercise a far field at θ=0.3 under any `ncrit`".
Measured:

| vertices | `ncrit` | P2P frac | M2L share | M2L cell pairs | gradient | live by T4's 0.75 bound? |
| --- | --- | --- | --- | --- | --- | --- |
| 642 | 64, 32 | **1.000000** | **0.0%** | **0** | `2.11941e-15` | no -- **no far field at all** |
| 642 | 16, 8 | 0.854631 | 14.5% | 5736 | `1.70348e-04` | no |
| 642 | 4 | 0.415373 | 58.5% | 60720 | `4.11035e-04` | **yes** |
| 2562 | 64, 32 | 0.854134 | 14.6% | 5736 | `1.75479e-04` | no |
| 2562 | 16 | 0.433923 | 56.6% | 73544 | `4.38234e-04` | **yes** |
| 2562 | **8** | **0.253650** | **74.6%** | **130968** | **`5.00775e-04`** | **yes** |
| 2562 | 4 | 0.107565 | 89.2% | 333250 | `6.39201e-04` | **yes** |

So the precise statements are:

- **At level 3 the far field is absent entirely at `ncrit` >= 32** -- M2L cell
  pair count exactly **0**, and the "error" is `2.1e-15`. **This is R1's
  cheapest misreading with a number on it**: a solve that is entirely P2P agrees
  with `BRSolverDirect` to round-off and reads as a spectacular success.
- **At level 3 the far field is live but a minority at `ncrit` 16 and 8** (14.5%
  of pairs), and a **majority at `ncrit` 4** (58.5%). `ncrit = 4` does *not*
  degenerate the tree: 10466 keys, fallback 0, 642 particles intact.
- **The `ncrit` at which each level first has a live far field**, taking live as
  T4's compiled `p2p_pair_fraction < 0.75`: **level 4 at `ncrit` 16**, **level 3
  at `ncrit` 4**. Taking live as "carries any of the field at all": level 4 at
  every scanned `ncrit`, level 3 at `ncrit` <= 16.

The a-priori estimate was not silly, it was one-sided: 642/8 = 80 occupied
leaves against a 105-leaf near field says no *leaf-level* pair is accepted, and
that is right. What it misses is that the MAC is applied **cell-to-cell at every
level**, so coarse pairs are accepted where leaf pairs are not, and 14.5% of the
field goes through them. **Level 3 is still not where a far-field accuracy claim
belongs** and nothing here adopts it -- at 14.5% M2L share its `1.70e-4` is
mostly a direct sum -- but "cannot exercise a far field under any `ncrit`" is
too strong and T6's L3 member should not repeat it.

**Read the liveness table the right way round.** The error *rises* as the far
field takes over, because a larger M2L share means more of the field is
approximated. A row low in that table is not better code and a row high in it is
not better accuracy -- it is less measurement. The trap this scan is shaped
around is that the two are indistinguishable without the P2P fraction beside
them.

**Which regime each level is in:** level 4 at `ncrit` 8 is in the **truncation**
regime and the model applies (74.6% M2L, order curve falling 4.7 decades from
p=0 to p=5). Level 3 at `ncrit` 8 is in a **mixed** regime -- live, but with
seven eighths of the field evaluated exactly -- and its figures are not
comparable with level 4's. Level 3 at `ncrit` >= 32 is in the **no-far-field**
regime and has no accuracy figure at all, only a round-off residual.

### Step 3 -- the basis selector, and why the gap is not the discriminator

| level | `CartesianTaylor` p=3 | `SolidHarmonic` p=3 | ratio | `CartesianTaylor` p=2 |
| --- | --- | --- | --- | --- |
| 4 | `5.00775e-04` | `4.69909e-03` | **9.38x** | `5.96723e-03` |
| 3 | `1.70348e-04` | `5.91359e-04` | 3.47x | `1.66340e-03` |

**The alarm did not fire**: the two bases are not on top of each other, and
T4's 9.30-9.38x at level 4 reproduces exactly. **And the document's warning is
confirmed quantitatively**: `CartesianTaylor` at p=2 (`5.97e-3`) sits **1.27x
above** the solid-harmonic point at p=3 (`4.70e-3`), so a test keyed to the
*size* of the gap would be satisfied by an ordinary p=2 point and would read a
correctly-selected p=2 arm as a broken selector. The discriminator is the
**shape** of the order curve against that fixed point: `CartesianTaylor` falls
4.7 decades across p=0..5 while the solid-harmonic arm is a single point that no
order rescues, because its bias is in the kernel and not in the truncation.

**The solid-harmonic curve's own shape is not measured** -- T2 dispatches one
solid-harmonic arm and measuring the shape would need arms it does not build.
**The self-contact separation is not measured either**, per the task's standing
decision: R2's "tens of percent" needs accepted separations approaching
sqrt(b) and no configuration in this tree reaches that. The decade is what this
geometry has.

### Step 5 -- the operator table, and `max_depth`

**`max_depth` is measured inert, and this is the cleanest negative result in the
scan.** At level 4, `ncrit` 8, θ=0.3, the gradient error, the potential error,
the P2P fraction, the M2L cell-pair count **and** the realized key count are
identical **to all 17 printed digits** at `max_depth` 5, 6, 8, 10 and 12. Same
at level 3. The tree reaches `ncrit` occupancy well above depth 5, so the cap
never binds at these vertex counts and **lowering it changes nothing**. R6 has
no evidence to act on and R12's "do not respond by lowering `max_depth`" is not
merely a rule here -- the lever is provably disconnected.

**No arm overflowed the operator table and no arm took the fallback path.**
`global_m2l_fallback_pair_count` is **0 at all 144 arms**, so every accuracy
figure in this entry is one code path and not a mixture. Key counts, against
Canopy's per-rank cap of **32768** (`m2l_effective_op_cap()`, so the count cap
binds and not the 2 GiB byte budget -- confirming what `FmmParams` claims):

| configuration | keys (rank 0) | bytes/key | table | % of cap |
| --- | --- | --- | --- | --- |
| L4 `ncrit` 8, p=3 (production) | 10902 | 3200 | 33.27 MB | 33% |
| L4 `ncrit` 8, p=4 | 10902 | 9800 | 101.89 MB | 33% |
| L4 `ncrit` 8, p=5 | 10902 | 25088 | 260.84 MB | 33% |
| L4 `ncrit` 4, p=3 -- **the worst key count scanned** | **18742** | 3200 | 57.20 MB | **57%** |
| L4 `ncrit` 8, p=3, `SolidHarmonic` | 9766 | 2560 | 23.84 MB | 30% |
| L3 `ncrit` 8, p=3 | 864 | 3200 | 2.64 MB | 3% |

`order` moves the bytes and **not** the key count, which is the level-keyed
basis behaving as documented; `ncrit` moves the key count. Canopy measured 25438
keys (78% of cap) on 8640 particles at `ncrit` 8, and 18742 at 2562 particles
with `ncrit` 4 is on that trajectory -- so **the cap is reachable at production
vertex counts even though nothing here reached it**, and R6 remains a live risk
for a larger mesh rather than a retired one.

Two caveats on these key counts, both of which matter to whoever reads them
next. **They are rank 0's, not the maximum over ranks** -- the field is
`local_` by design because the cap is per-rank -- and at np=4 rank 0 reports
8047 where np=1 reports 10902, so a rank sweep is not a scaling measurement.
And **the printed diagnostics are the arm's *second* evaluation's**, the
potential one, whose maintenance action is therefore `Migrate` at every arm
where the first was `Setup`. **T8 gets no `MaintenanceAction` histogram from
this task**, exactly as T4 gave it none.


### Step 7 -- the divergence horizon, and the instrument that could not measure it

**This step did not land as written, and the reason is a defect in the
measurement instrument rather than in Beatnik.** The task says step 7 "rebuilds
none of M0-D1's machinery" and reuses `milestone0_ladder.py pair`. That tool
**cannot measure an FMM-driven horizon**, it fails **silently**, and the number
it produces is wrong by six orders of magnitude in the direction that looks like
catastrophe.

#### What the existing tool reported, and why it is wrong

Run against the level-3 FMM trajectory, `pair` reports `vertices`
`max|e| = 4.95e-1` at **step 25** and a first-failing step of 25 at every rung
including the loosest. On a bubble of radius 0.25 that is half a diameter: it
reads as the FMM destroying the trajectory inside 25 steps.

It did not. At step 25 the two meshes are geometrically identical to ~`1e-7`:

| | FMM run | Python gold |
| --- | --- | --- |
| `time` | 0.07493637649103907 | 0.074936383844851573 |
| centroid | (5.98e-10, 2.92e-10, 0.250501285) | (3.07e-17, 9.06e-18, 0.250501287) |
| bounding box min | (-0.24999976, -0.24999975, 0.00095576) | (-0.24999976, -0.24999976, 0.00095564) |
| radial extent about the centre | 0.249044 .. 0.250953 | 0.249044 .. 0.250953 |

and the run is physically healthy for all 2000 steps: **volume drift
`3.35e-9`**, entity counts constant at (642, 1280), minimum triangle quality
`0.0381` at its worst.

**The cause is the vertex pairing.** Neither side records a correspondence --
the Python `.npz` carries no `gid` -- so `compare_output.py` recovers one by
quantizing coordinates onto a grid of cell size `--match-eps` (default `1e-9`)
and lexsorting the integer keys. Its own docstring states the precondition: the
cell must be **much larger than the coordinate disagreement between the two
files** and much smaller than the vertex separation. A direct-driven run
satisfies it by nine decades. **An FMM-driven run does not**: at
τ_A = `5.0e-4` per evaluation the trajectories separate to `1.5e-7` by step 25,
which is **100x the cell**. The two files then quantize into different cells,
the lexsort orders them differently, and vertices are paired with the wrong
partners -- roughly antipodal ones, hence half a diameter.

**Two properties make this worse than an ordinary tolerance problem.**

- **`n_ambiguous` stays 0 throughout, so nothing reports it.** That counter
  counts rows sharing a cell with their predecessor *within one file*, which
  detects a within-file collision and not a cross-file mis-pairing. Entity
  counts match, the load succeeds, the exit status is an ordinary
  "compared and disagreed". This is risk **M0-R4**'s mechanism arriving through
  a hole M0-R4's own detector does not cover.
- **Raising `--match-eps` does not fix it.** Each step needs a larger cell than
  the last, and the window between "larger than the disagreement" and "smaller
  than the vertex spacing" closes:

| step | eps 1e-9 | 1e-7 | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | 4.95e-1 | 5.00e-1 | **1.46e-7** | 1.46e-7 | 1.46e-7 | 1.46e-7 |
| 100 | 4.97e-1 | 4.95e-1 | 4.99e-1 | **2.36e-6** | 2.36e-6 | 2.36e-6 |
| 400 | 5.28e-1 | 5.28e-1 | 5.22e-1 | 4.87e-1 | 4.46e-1 | **5.61e-5** |
| 1000 | 5.94e-1 | 5.94e-1 | 5.68e-1 | 5.68e-1 | 2.78e-1 | 5.41e-1 |

  By step 1000 no value works, and a per-step value chosen to make the answer
  small would be fitting the instrument to the result.

#### The instrument that replaced it, and its cross-validation

`tests/regression_tests/fmm_divergence_ladder.py`, in no tier, pairs by
**bijective nearest neighbour** and **refuses** any step whose pairing is not a
bijection -- the check a quantized lexsort cannot make across two files, since a
mis-pairing shows up as two run vertices claiming one reference vertex. It
imports `RUNGS` and `steps_in` from `milestone0_ladder` and `load_any` from
`compare_output`, so the two tools cannot disagree about what a rung is or what
a dataset is called, and it evaluates `compare_output.py`'s own elementwise
criterion `|e_i| <= atol + rtol*|g_i|` directly rather than bounding it -- with
the pairing in hand the first failing step is exact and needs no
derive-then-confirm pass. It is O(N^2) per step, which at 642 and 2562 vertices
is nothing; **at production vertex counts pair by `gid` instead**.

**It agrees with the tool on record wherever that tool is usable.** On the
level-4 **direct** run against the level-4 gold set, `milestone0_ladder.py pair`
confirms a first failing step of **750** at the `1e-12/1e-14` rung and `None` at
every looser rung; the new tool returns **exactly the same ladder**. That is the
cross-validation that makes the FMM numbers below quotable, and it is also the
attribution row step 7 asks for.

`remesh_material_position` is constant at `5.55e-17` in every comparison -- it
is the material reference position and the mesh is frozen -- so it never fails a
rung and carries no information here. Scalars (`time`, `initial_volume`,
`initial_min_edge`) are reported but **excluded from the ladder**: under
`--adaptive-dt` the timestep is a function of the state, so an FMM-driven run
and a direct-driven one are at slightly different physical times at the same
step, and a ladder that failed on `time` would be reporting that rather than the
trajectory.


#### The measurement

**Runs.** `l4dir`, `l3fmm` and `l4fmmA` in job `f3aPUHSZuj99`; `l4fmmB` in
`f3aPpr9TEDT5` and `l4fmm4` in `f3aPprHQJN6K`, both single-row jobs after the
budget guard skipped them (see *Cost and mechanics*); `l4fmm4B` in
`f3aQDvjGqxFh`, the second np=4 row R8's strict test needs. Every row 2000 steps,
checkpoint every 25, **81 files each**, level 4 unless stated, `ncrit` 8,
`order` 3, `CartesianTaylor`, `mac_theta` 0.3, `max_depth` 10,
`softening` 0.025, `near_softening_factor` 0, HIP.

**`max|e|` per step, under the bijective nearest-neighbour pairing:**

| step | L4 FMM run A `vertices` | L4 FMM run B | L4 FMM np4 | L4 FMM A `potential` | L4 **direct** `vertices` | L3 FMM `vertices` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5.551115e-17 | 5.551115e-17 | 5.551115e-17 | 0.000000e+00 | 5.551115e-17 | 5.551115e-17 |
| 25 | 4.585411e-07 | 4.585411e-07 | 4.585411e-07 | 1.027049e-08 | 6.661338e-16 | 1.455685e-07 |
| 50 | 1.819040e-06 | 1.819040e-06 | 1.819040e-06 | 9.045491e-08 | 1.165734e-15 | 6.060375e-07 |
| 75 | 4.006012e-06 | 4.006012e-06 | 4.006012e-06 | 3.209197e-07 | 1.498801e-15 | 1.345119e-06 |
| 100 | 6.838146e-06 | 6.838146e-06 | 6.838146e-06 | 7.097854e-07 | 1.998401e-15 | 2.361412e-06 |
| 200 | 3.122244e-05 | 3.122244e-05 | 3.122244e-05 | 8.699897e-06 | 3.053113e-15 | 1.093940e-05 |
| 400 | 8.557352e-05 | 8.557352e-05 | 8.557352e-05 | 4.179612e-05 | 8.160139e-15 | 5.606311e-05 |
| 600 | 7.899350e-04 | 7.899350e-04 | 7.899350e-04 | 2.054120e-04 | 6.861178e-14 | 7.254851e-05 |
| 800 | 2.065732e-03 | 2.065732e-03 | 2.065732e-03 | 3.979515e-04 | 1.962874e-13 | 1.180409e-04 |
| 1000 | 1.544184e-03 | 1.544184e-03 | 1.544184e-03 | 4.970535e-04 | 2.398082e-13 | 1.570048e-04 |
| 1200 | 1.588563e-03 | 1.588563e-03 | 1.588563e-03 | 5.230601e-04 | 2.617906e-13 | 2.507739e-04 |
| 1325 | 1.640647e-03 | 1.640647e-03 | 1.640647e-03 | 5.212907e-04 | 3.194112e-13 | 3.613944e-04 |
| 1925 | 2.501937e-03 | 2.501937e-03 | 2.501937e-03 | 6.485149e-04 | 1.110223e-13 | 4.569707e-03 |
| 2000 | 2.465232e-03 | 2.465232e-03 | 2.465232e-03 | 6.522437e-04 | 9.914292e-14 | 5.129741e-03 |

**The tolerance ladder** -- first failing checkpointed step, computed
elementwise and exactly rather than derived-then-confirmed:

| comparison | 1e-12/1e-14 | 1e-10/1e-12 | 1e-8/1e-10 | 1e-6/1e-8 | 1e-4/1e-6 |
| --- | --- | --- | --- | --- | --- |
| **L4 FMM np1 run A vs gold -- THE ENVELOPE** | **25** | **25** | **25** | **25** | **50** |
| L4 FMM np1 run B vs gold | 25 | 25 | 25 | 25 | 50 |
| L4 FMM np4 run A vs gold | 25 | 25 | 25 | 25 | 50 |
| L4 FMM np4 run B vs gold | 25 | 25 | 25 | 25 | 50 |
| L4 FMM run A vs L4 direct (larger field set) | 25 | 25 | 25 | 25 | 50 |
| **L4 direct np1 vs gold -- attribution** | **750** | None | None | None | None |
| L3 FMM np1 vs gold (control) | 25 | 25 | 25 | 25 | 75 |

**The envelope is: the FMM-driven level-4 trajectory matches the in-tree Python
gold set through step 25 at the `1e-4/1e-6` rung and fails by step 50; at every
tighter rung it has already failed at step 25, the first checkpoint.** 25 is the
checkpoint interval, so the true horizon at the tight rungs is somewhere in
steps 1-25 and this measurement cannot localize it further -- a finer
`--checkpoint-every-steps` would, and none was run.

**R8's spread, recorded separately from the direct-versus-FMM gap as R8
requires.** Runs A and B are the identical deck at the identical rank count.
Their `vertices` `max|e|` agrees to **6.9e-11 of the error** at worst over every
pairable step (0.0 exactly at steps 25, 50, 75 and 100), and all four ladders are
**identical**. Final volume drift across the four: `4.703227807e-9` (np1 A),
`4.703227363e-9` (np1 B), `4.703228917e-9` (np4 A), `4.703228473e-9` (np4 B).

**R8's fear did not materialize, and the envelope can therefore be tight.** The
reason it does not still has to be stated carefully, because the obvious pair of
runs does not test what R8 is about. R8's mechanism is **Zoltan2's partition
non-determinism across runs**, and **runs A and B are both np=1, where the
partition is trivial** -- so that pair bounds GPU reduction-order
nondeterminism and nothing else. The strict test is **two runs at a non-trivial
partition**, and it was run (`f3aQDvjGqxFh`, a second np=4 row at the identical
deck):

| comparison | what it varies | worst spread, as a fraction of the error |
| --- | --- | --- |
| np1 run A vs np1 run B | GPU reduction order only (trivial partition) | **6.9e-11** |
| **np4 run A vs np4 run B** | **Zoltan2's partition, run to run** | **1.4e-11** |
| np1 vs np4 | partition *and* rank count | 1.8e-11 |

All four runs give the **identical ladder** (25/25/25/25/50) and four final
volume drifts inside `4.7032274e-9 .. 4.7032289e-9`, a span of `1.5e-15`
absolute. **Zoltan2's partition non-determinism is not detectable in this
trajectory at 2000 steps** -- it is, if anything, *smaller* than the np=1 pair's
reduction-order spread, which is the opposite of what R8 anticipated. So the
envelope may be asserted at the observed horizon with a margin set by the
checkpoint interval rather than by run-to-run noise, and **R8 does not bind at
this configuration**. It is not retired in general: four runs at one
configuration on one machine is what this says, and a deforming or
self-contacting sheet redistributes particles far more aggressively than a
smooth bubble does.

#### The attribution, and why the horizon is the FMM's

The direct-driven row fails first at **step 750** and only at the tightest rung;
at step 25 it is `6.66e-16` against the FMM's `4.59e-7`, **nine orders
smaller**. So the Beatnik-versus-Python drift contributes nothing to the FMM
horizon, and the design's expectation that "the two ladders nearly coincide"
holds in the stronger form: the FMM-vs-gold and FMM-vs-direct ladders are
**identical at every rung**, because the FMM's own perturbation dominates both
comparisons. Note the two are **not over the same field set** -- FMM-vs-direct
is Beatnik-vs-Beatnik and adds `sheet_vector` (which reaches `4.28e-2` by step
2000, the largest of any field) while FMM-vs-gold cannot see it, since the
reference `.npz` carries no such key.

**The L3 control did its job.** At 642 vertices the far field carries 14.5% of
the pairs, so that row is mostly a direct sum with FMM bookkeeping -- and its
horizon is *later* (75 against 50 at the loosest rung) and its `max|e|` smaller
at every step through 1200. That is the expected ordering and it is what says
the level-4 horizon is the **expansion's** rather than an artifact of the FMM
code path as such.

#### The volume-drift bound claim B needs

**`4.703e-9` FMM-driven against `4.741e-9` direct-driven** at step 2000, both
monotone in step and both maximal at step 2000. The FMM is marginally *better*,
which is not a claim about the FMM -- both are dominated by the volume
projection, which is on in this configuration (`preserve_volume = true`).
Minimum triangle quality is `0.125005` (FMM) against `0.124242` (direct) at
level 4, and `0.038118` at step 1725 at level 3.

**This is the most robust number in step 7** and the only claim-B assertion
untouched by the pairing problem below: `milestone0_ladder.py series` computes
the drift from **one** directory, with no reference and therefore no vertex
correspondence to recover. T6 can assert against it directly.


### Cost and mechanics

`spack install` after the adapter header change: **10m33s wall / 102m42s CPU**
on the login node — a full rebuild, because `Beatnik_FarFieldInterface.hpp`
reaches every translation unit that creates a BR solver. T4's comparable figure
was 11m17s/88m40s. The scan itself is cheap: job `f3aPSQYSVTpb`, six launches
over 24 arms each, **82 s of launch wall** in total (9 s for L3 HIP np1, 29 s
for L4 Serial np1), one `pdebug` node, `-t 45m` requested and barely touched.

**`flux batch --flags=waitable` is still refused on this instance**, so
`flux job status <jobid>` remained the wait mechanism, as in T3 and T4.

### The step-7 sweep does not fit `pdebug`'s one-hour cap, and that is a finding

Job `f3aPUHSZuj99` ran three of its five rows and **skipped the other two
loudly** rather than starting a row it could not finish:

```
[t5d] === SKIPPED l4fmmB_...: 642s of budget left, estimate 1060s.
      THE SWEEP DOES NOT FIT pdebug's 1h cap.
[t5d] launched=3 skipped=2 total=2658s budget=3300s
```

**The cause is that a short probe under-predicts the FMM per-step cost, because
that cost grows along the trajectory.** Measured, at level 4 with `ncrit = 8`:

| row | probe (25 steps) | full run (2000 steps) | ratio |
| --- | --- | --- | --- |
| L4 HIP np1 `direct` | 0.009899 s/step | 0.008515 | 0.86 |
| L3 HIP np1 `fmm` | 0.068442 | 0.128000 | **1.87** |
| L4 HIP np1 `fmm` | 0.434082 | **1.188500** | **2.74** |

and within one run the cumulative rate climbs monotonically — L4 `fmm` run A
reads 0.554 s/step at step 300, 0.685 at 500, 1.050 at 1300, 1.143 at 1700,
1.189 at 2000 — as the bubble deforms, the tree deepens and the operator table
is rebuilt against a drifting root box. The `direct` row shows no such growth
(0.0085 flat), so this is the FMM path's own behaviour and not the physics
getting harder. **The absolute ratio — `fmm` at roughly 44x `direct` per step at
2562 vertices, 140x at the 2000-step rate — is T8's to characterize, not this
task's**; it is recorded because it is what the budget is built from.

**What was done about it, and what deliberately was not.** The two skipped rows
were re-run as **separate single-row `pdebug` jobs** (`f3aPpr9TEDT5` and
`f3aPprHQJN6K`), each at the full 2000 steps, the same queue and the same
per-job walltime. That is not any of the three responses the task forbids: no
walltime was lengthened past the cap, no queue was changed, and no run was
shortened. The script gained a `BEATNIK_T5_SWEEP` row override for it, which
prints a loud banner saying the job is a subset. **The five-row sweep as one
job is not recoverable** — at the measured rates its five rows need about 5100 s
against a 3300 s budget inside a 3480 s wall — so a later session should submit
it as the union of jobs it now is, and should re-measure the estimates after any
change that could move the per-step cost.

**One thing the guard got right that is worth keeping.** Its estimates were
wrong by 2.7x and it still protected the measurement, because it skips on the
*remaining budget* rather than trusting the estimate to be accurate: the row it
refused to start is the row that would have been killed at the wall with a
truncated series on disk. The post-row checkpoint-count check
(`steps/every + 1`, 81 files) is the second line and fired on nothing.


### Bugs only running revealed

**None in Beatnik's own code.** The adapter, the round trip, the dispatch,
`BRSolverFMM` and the FMM-driven timestep all ran clean at 144 scan arms and
four 2000-step trajectories; the two guards T2 added on its own initiative --
the self-validating forward distributor and the round-trip completeness throw --
stayed silent throughout, as did the softening-stability guard. The FMM-driven
runs held their entity counts for 2000 steps, stayed finite, and drifted
`4.7e-9` in volume.

Three things only running revealed, all in the measurement apparatus:

1. **`compare_output.py`'s vertex pairing degenerates silently on an FMM-driven
   run** -- the big one, written up under step 7 and recorded in README's Known
   Issues. It reports half a bubble diameter where the truth is `1.5e-7`, with
   no diagnostic firing.
2. **A 25-step probe under-predicts the FMM per-step cost by 2.7x**, because the
   cost grows along the trajectory. This is what made the step-7 sweep not fit
   `pdebug`; written up under *Cost and mechanics*.
3. **A `Kokkos::View<Real*[3], Device>` is LayoutLeft on HIP and LayoutRight on
   Serial**, so the scan driver's first cut -- which packed its `MPI_Allgatherv`
   through `.data()` -- would have transposed the source set on the GPU backend
   and built the potential reference from a scrambled point cloud. Caught by
   reading before submitting, not by a failure: it would not have crashed, it
   would have produced a disagreement of about the size T5 is trying to measure.
   The driver packs by explicit indexing and says why.

### Departures from T5's stated Do steps

- **Three files outside the stated Fill-in list changed**, each because the exit
  criterion cannot be met otherwise; they are listed under *Signatures changed*
  above. The adapter one is the substantive departure: step 2 requires a
  potential column and **no route to one existed**.
- **Step 7's ladder is not `milestone0_ladder.py pair`.** The task says step 7
  "rebuilds none of M0-D1's machinery" and names that tool. It cannot measure an
  FMM-driven horizon, for the reason written up above, so
  `fmm_divergence_ladder.py` measures it instead -- importing `RUNGS`,
  `steps_in` and `load_any` rather than redefining them, and cross-validated
  against `pair` on the direct run where `pair` *is* usable. **`pair` was not
  modified**, so every M0-D1 number it has produced stands unchanged.
- **The step-7 sweep ran as three jobs rather than one.** The five-row matrix
  does not fit `pdebug`'s cap; the guard skipped two rows and they were re-run
  as single-row jobs at full step count. No walltime past the cap, no queue
  change, no shortened run.
- **`ncrit` was scanned at level 3 and found to admit a far field**, which the
  design document says is impossible. This is a correction, not a tuning
  exercise: `ncrit` is a required scan axis, level 3 is not adopted for
  anything, and no run was spent pushing it toward liveness.
- **`--checkpoint-every-steps` stayed at 25**, so the horizon at the tight rungs
  is localized only to "somewhere in steps 1-25". Narrowing it was not run.


**Affects:** **T6** — this is the entry T6 reads its numbers out of, and three
of them change its plan rather than just filling it in.
**(a) τ_A is `5.01e-4` on the gradient** with the qualification list above, so a
claim-A tolerance compiled at level 4, `ncrit` 8, `order` 3 has 2x of headroom
at `1e-3` and 4x at `2.0e-3` (T4's budget, which it may reuse unchanged); the
**potential** figure `3.73e-5` is a different column and must not be compiled as
if it were the velocity.
**(b) T6 step 3 cannot be written as stated.** It asserts an FMM-driven horizon
against the Python gold set through `milestone0_ladder.py pair`, and **that tool
cannot measure one** — it mis-pairs vertices silently once the two files
disagree by more than `--match-eps`, which an FMM run exceeds by step 25, and no
`--match-eps` fixes it. T6 must either use
`tests/regression_tests/fmm_divergence_ladder.py` (bijective nearest neighbour,
refuses a non-bijective step, reproduces `pair` exactly on a direct run) or pair
by `gid`, which is unavailable against a `.npz` gold. **Do not assert a horizon
through `pair` on an `fmm` run.**
**(c) The horizon envelope is: first failing checkpointed step 25 at the
`1e-12`, `1e-10`, `1e-8` and `1e-6` rungs and 50 at `1e-4`**, identical across
**four** independent runs. The spreads, which R8 requires be recorded separately
from the direct-versus-FMM gap: **1.4e-11** of the error between two np=4 runs
(the strict test, since it varies Zoltan2's partition), **6.9e-11** between two
np=1 runs (reduction order only), **1.8e-11** between np1 and np4. **R8 does not
bind at this configuration** and the envelope may be asserted at the observed
horizon — but 25 is the checkpoint interval, so the tight-rung horizon is
localized only to steps 1-25 and a T6 assertion must not claim finer. Set the
margin from the checkpoint interval, not from run-to-run noise.
**(d) The volume-drift bound is `4.703e-9`** (FMM) against `4.741e-9` (direct)
at 2000 steps, maximal at the final step, minimum triangle quality `0.125005`.
This is the one claim-B assertion **immune to the pairing problem**, because
`series` needs no reference, and it is therefore the one T6 should lean on.
**(e) The per-level `ncrit`:** level 4 first has a far field carrying the
majority of pairs at `ncrit` 16 and the production 74.6% at `ncrit` 8; **level 3
is not dead after all** — the far field is absent only at `ncrit` >= 32, carries
14.5% at `ncrit` 16 and 8, and 58.5% at `ncrit` 4. T6's L3 member should say
"mostly a P2P comparison" with the measured 14.5% rather than repeat the design
document's "no far field under any `ncrit`".
**(f) Steps 1350-1900 at level 4 are unpairable by any position-based scheme**,
identically in three independent runs, so a T6 assertion must not depend on
them; the horizon is decided long before.
**(g) The milestone tier's walltime** must be set from `fmm` costs, not
`direct` ones: one 2000-step L4 FMM run is **2377 s** at np1 and **1473 s** at
np4 against the direct row's 25 s, so two FMM members at two rank counts is
roughly 2.2 hours of launch — well past `pdebug` and past **R9**'s already
uncomfortable estimate. T6 step 6 owns that and should plan on splitting.

**T7** — `evaluatePotential` is a third public evaluation on the adapter and the
Riesz path is unaffected by it; `computeSurfaceRieszScalar` still throws. The
`Contraction` enum now has three enumerators and the scatter branches on
`!= Trace`, so a Riesz change must keep that shape.
**T8** — this task contributes **no** `MaintenanceAction` histogram (every
reported action is `Migrate`, from each arm's second evaluation) and T8 still
starts from zero there. It does contribute the cost facts T8's walltime planning
needs: `fmm` is **~44x `direct` per step at 25 steps and ~140x at 2000** at 2562
vertices, and **the per-step cost grows 2.7x along the trajectory** (0.434 to
1.189 s/step), which is R12's operator-rebuild signature showing up as a wall
time rather than as a key count. `local_m2l_bytes_per_key` and
`local_m2l_op_cap` are now in `FarFieldDiagnostics` for its table.
**X1** — **not implied.** τ_A is `5.01e-4`, comfortably below `1e-3` at the
production order, so the conditional X1 describes did not fire and the
deliverable is a working, measured, bounded-error fast path at better than the
reference implementation's own fidelity.
