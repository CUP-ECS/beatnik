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
