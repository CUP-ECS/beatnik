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
- **The divergence-horizon envelope and its run-to-run spread** (T5 step 6,
  R8). An envelope with no spread beside it is not usable, and T6 will trip on
  it intermittently.
- **The realized operator-key count and `total_fallback_pair_count()`** at each
  scanned `max_depth` and `order` (T5 step 4, R6). A non-zero fallback count
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
