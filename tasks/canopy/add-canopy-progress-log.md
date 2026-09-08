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
`tasks/canopy/abstract-solver-backend.md` rather than from the headers, and the
basis-independent findings — maintenance cost, the knob semantics, the open
defects — from `tasks/canopy0.md`, which read those headers in full. **T2** is
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
  $\binom{p+3}{3}\sim p^3/6$ (`tasks/canopy/canopy-kernel-rec.md`, "Convergence
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
