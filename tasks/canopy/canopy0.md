# Canopy as the far-field engine for a distributed vortex-sheet solver

**Status:** IN PROGRESS — the survey below is complete; every task is NOT STARTED.

## Problem

A downstream application wants to use Canopy as the far-field engine for a
**distributed unstructured vortex-sheet solver**. That consumer is unlike the
workloads Canopy has been driven with so far, in five ways that all bear on the
API:

1. **Its kernel is a softened $1/r^2$ field, evaluated as a contraction.**
   The quantity it needs is not a potential and not a force; it is
   $\sum_s K(x_t,y_s) \times S_s$ with
   $K = \delta/(b + r^2)^{3/2}$, $\delta = x_t - y_s$,
   $r^2 = |\delta|^2$ — the Birkhoff–Rott integral — and, in a second
   configuration, the dot-product contraction $\sum_s (\delta \cdot G_s)/(b + r^2)^{3/2}$
   against a different vector source. The softening $b$ is not
   a numerical guard: it is the sheet thickness, a physical parameter of the
   model, and the reference answer the solver is validated against is a direct
   sum of exactly that kernel. A representative value is
   $b = \varepsilon^2 = 6.25\times10^{-4}$, i.e.
   $\varepsilon = \sqrt b = 2.5\times10^{-2}$, on a domain of extent $O(1)$.

2. **Its sources are owned entities of a distributed mesh, not free particles.**
   Each source is a mesh vertex owned by exactly one rank under a decomposition
   the *mesh* owns. The result must land back on that vertex, on that rank. The
   solver cannot adopt a decomposition chosen by the far-field engine, because
   every other operator in the solver — differentiation, remeshing, halo
   exchange, I/O — is expressed in the mesh's own layout.

3. **It calls the far field three times per timestep, not once.** Its time
   integrator is a three-stage Runge–Kutta; each stage moves every source and
   then requires a fresh evaluation. So a "one setup, one solve per step"
   cost model does not describe it: whatever the engine needs between
   evaluations is paid three times per step, not once.

4. **Its sources lie on a two-dimensional sheet embedded in three dimensions**,
   not in a volume. At late time the sheet rolls up and approaches itself, so
   the *geometric* separation between two parts of the surface falls toward
   $\sqrt b$ while the *along-surface* distance between them stays large.

5. **It is validated against a direct summation of the same kernel** at a
   tolerance it must state in advance. Its direct solver's own reproducibility
   floor across rank counts is $\sim\!10^{-15}$ relative, so the achievable
   comparison tolerance is set entirely by the far-field engine's accuracy, and
   the consumer needs that number to be *knowable*, not discovered by tuning.

This document records a **read-only survey** of Canopy answering five specific
questions from that consumer, and then one task per gap the survey found. The
survey opened no other dependency and changed no Canopy code.

**Second reading pass.** F6, F7, F8, F4's exact-radius note, C1's revised option
set and C11 come from a later read-only pass that opened
`src/Canopy_LaplaceKernel.hpp` in full (all 887 lines) alongside a reference
Barnes–Hut treecode (`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`,
all 138 lines) to answer the question C1 had left open — whether a *softened* far
field is reachable in the existing expansion basis. See [`treecode.md`](treecode.md)
for the reference treecode itself. That pass built nothing and ran nothing
either; every accuracy number in this document is still an estimate or a carried
figure, never a measurement.

**Out of scope:** the consumer's own adapter layer; any change to the consumer's
configuration surface; performance work not implied by one of the tasks below;
and the choice between Canopy and any other far-field method.

## Read this first

Four assumptions a reader coming to Canopy from that consumer's side is likely
to hold, and what is actually true.

**"The multipole far field expands the softened kernel."** It does not. The
far-field path (P2M → M2M → M2L → L2L → L2P) is built entirely from the
**unsoftened** $1/r$ Laplace kernel; softening exists only in the near-field
P2P kernel (`src/Canopy_P2P.hpp:799-803`, `883-896`) and is kept relevant by a
floor that pushes close pairs out of M2L and into P2P
(`src/Canopy_Solver.hpp:69-77`, `src/Canopy_CommunicationPlan.hpp:347-361`).
Everything a consumer wants to say about far-field accuracy has to account for
that. See **F1** and **C1**.

**"So the far-field operators must be missing or incomplete."** They are not.
All five are implemented in the solid-harmonic basis with Greengard theorem
citations on each — `p2m_contribution` (`src/Canopy_LaplaceKernel.hpp:230`),
`m2m_translate` (`:273`, Thm 5.22), `m2l_translate` (`:373`, Thm 5.23),
`l2l_translate` (`:688`, Thm 5.26), `l2p_evaluate` (`:800`), plus a
precomputed-operator M2L path (`m2l_build_operator`, `:516`). What is missing is
**softening inside them**, and that is a change of expansion basis rather than a
patch to these routines. See **F6**.

**"`setSources` then `solve` maps onto Canopy's `setup` then `solve`."** It maps
onto `setup`/`auto_maintain` plus `solve`, and the split is not where a caller
would put it. `solve()` reads current positions but uses the leaf membership,
communication plan and P2P neighbour lists cached by the *last* setup or
maintenance call (`src/Canopy_UpwardSweep.hpp:657-669` iterates
`_leaves_at_depth_local`; `src/Canopy_P2P.hpp:819`, `:834` use cached
`_leaf_particle_offsets` / `_particle_to_league`). Calling `solve()` after moving
particles, without a maintenance call, does not fail — it silently evaluates a
wrong near/far partition. See **F3** and **C3**.

**"Canopy evaluates at the caller's particles, in the caller's order."** It
evaluates at *its own* particles: `setup()` and every maintenance path migrate
particles across ranks and permute the local array, and
`TreePartitioner::migrate_particles` states outright that "the within-AoSoA order
after migration is unspecified" (`src/Canopy_TreePartitioner.hpp:576-580`). There
is no identity, global-ID, or inverse-permutation facility anywhere in `src/`.
See **F3** and **C2**.

## Findings

### F1 — Kernel generality and the form of the softening

**Canopy exposes one fixed kernel: the $1/r$ Laplace kernel.** There is no
kernel abstraction to specialize. `LaplaceKernel` is the only kernel in the tree
(`src/Canopy_LaplaceKernel.hpp:150-152`) and `Solver` hard-wires it —
`kernel_type` is a typedef, not a template parameter
(`src/Canopy_Solver.hpp:104-112`). A consumer cannot supply its own kernel.

**What it can express instead is a fixed *set* of Laplace solves.** `NComps` is
the number of simultaneous independent charge components, and the header already
names this consumer's use case: "3 for Biot-Savart via three parallel Laplace
solves" (`src/Canopy_LaplaceKernel.hpp:133-135`). With `NComps = 3` and
`compute_gradient = true`, one traversal produces a $3\times3$ tensor per
target (`src/Canopy_DownwardSweep.hpp:120-126`, gradient shaped
`(num_particles, NComps, 3)`).

**The softening is the same functional form the consumer needs, not merely an
analogue.** Canopy's near field is Plummer softening applied as
$r^2 \rightarrow r^2 + \varepsilon^2$ in every pairwise term
(`src/Canopy_P2P.hpp:799-803`), and the gradient it accumulates is

```
inv_r  = 1 / sqrt(r2 + eps2);   inv_r3 = inv_r^3
g[c]  -= q[c] * d * inv_r3                       // d = x_target - x_source
```

(`src/Canopy_P2P.hpp:883-896`). That is exactly
$-\sum_s q_s\,\delta/(r^2+\varepsilon^2)^{3/2}$: the consumer's
$K = \delta/(b+r^2)^{3/2}$ with $\varepsilon^2 = b$, up to an overall
sign. A consumer therefore sets `FmmConfig::softening = sqrt(b)` — an explicit
non-negative value, which also suppresses the distribution-based auto-softening
that would otherwise be derived once at first setup and frozen
(`src/Canopy_Solver.hpp:166-178`, `680-713`). No functional-form conversion is
needed and no reinterpretation of $b$ is needed.

**But only the near field is softened.** This is the single most consequential
finding in this document. The multipole far field expands $1/r$ unsoftened;
accuracy in the far field is bought by *excluding* every pair where softening
matters, via a floor in the acceptance criterion: an M2L pair is rejected
whenever the cell-centre separation $R \le \texttt{near\_softening\_factor} \cdot \varepsilon$
(`src/Canopy_CommunicationPlan.hpp:347-361`). A rejected
pair falls through the dual-tree traversal to a leaf-leaf P2P pair
(`src/Canopy_CommunicationPlan.hpp:623-634`), so the near/far partition stays a
partition and the close pairs do get the softened kernel.

For pairs that *are* taken by M2L, the evaluated kernel is not the consumer's
kernel, and the difference is a **systematic bias that does not shrink with
expansion order**. Per pair, at separation $R$:

| quantity | exact vs. unsoftened relative error |
| --- | --- |
| potential $1/\sqrt{R^2+\varepsilon^2}$ | $\approx \tfrac12 \varepsilon^2/R^2$ |
| gradient $\delta/(R^2+\varepsilon^2)^{3/2}$ | $\approx \tfrac32 \varepsilon^2/R^2$ |

The bound quoted in `README.md:52-63` — "far-field relative softening error
`~ 1/(2·factor²)`, ≈3% at the default `4`" — is the **potential's**. A consumer
that uses the gradient (as this one does; both of its contractions are gradient
contractions) sees three times that: $\tfrac32/\text{factor}^2 \approx 9.4\%$
at the default `near_softening_factor = 4`. Inverting for a target far-field
fidelity $\tau$ gives a required exclusion radius
$R > \varepsilon\sqrt{1.5/\tau}$:

| target $\tau$ | required `near_softening_factor` | exclusion radius at $\varepsilon = 2.5\times10^{-2}$ |
| --- | --- | --- |
| $10^{-2}$ | 12.2 | 0.31 |
| $10^{-3}$ | 38.7 | 0.97 |
| $10^{-6}$ | 1225 | 30.6 |

On a domain of extent $O(1)$ the $10^{-3}$ row already makes the near
field the entire domain — i.e. $O(N^2)$ — and the $10^{-6}$ row is
unreachable at any cost. **There is no setting of `near_softening_factor` that
delivers a $10^{-6}$-accurate softened-kernel evaluation with a bare-kernel
far field.** Either the far-field operator must carry the softening, or the
consumer's accuracy claim must be stated at the $10^{-2}$–$10^{-3}$
level and justified. This is task **C1**; what "carry the softening" would
actually cost is **F6**.

Two smaller notes on the kernel:

- P2P skips any pair with $r^2 < 10^{-24}$ (`src/Canopy_P2P.hpp:881`).
  For a consumer whose target set equals its source set this is the
  self-interaction, whose contribution to the *gradient* contraction is exactly
  zero ($\delta = 0$), so skipping it is correct rather than merely
  tolerable. It also silently drops genuinely coincident distinct sources, which
  for this kernel likewise contribute zero to the gradient.
- The far field carries no softening anywhere else either: `grep` for
  `softening` over `src/Canopy_DownwardSweep.hpp` and
  `src/Canopy_UpwardSweep.hpp` returns nothing.

### F2 — The cross-product contraction

**It cannot be folded into the kernel, and it does not need three separate
solves.** There is no hook in `LaplaceKernel` for a caller-supplied contraction
— its static methods (`p2m_contribution`, `m2m_translate`, `m2l_translate`,
`l2l_translate`, `l2p_evaluate`, `src/Canopy_LaplaceKernel.hpp:142-147`) are
fixed. But `NComps = 3` with `compute_gradient = true` yields, per target
$i$, the full tensor

$$
  T_{cj}(i) \;=\; \texttt{gradient}(i,c,j) \;=\; -\sum_s q_{c,s}\,
  \frac{\delta_j}{(r^2+\varepsilon^2)^{3/2}},
$$

from **one** tree traversal and **one** ghost exchange
(`src/Canopy_Solver.hpp:202-239`; `src/Canopy_DownwardSweep.hpp:120-126`). Load
the three charge components with the three components of the vector source, and
both contractions the consumer needs are purely local post-processing of that
$3\times3$ tensor:

- cross product: $u_i = -\epsilon_{ijk}\,T_{kj}$ — a nine-term local
  reduction, no communication;
- dot product: $\Psi = -\operatorname{tr} T = -\sum_c T_{cc}$ — a
  three-term local reduction.

So the two evaluations the consumer needs share the tensor *only* when they share
the source vector. They do not: the velocity contraction is against the sheet
strength and the scalar contraction is against a different vector field. That is
**two `solve()` calls with different charges over the same tree**, not one, and
both want the gradient and neither wants the potential. Canopy supports two
`solve()` calls over one tree directly (`solve()` re-reads the charge slice each
call and zeroes its outputs first, `src/Canopy_Solver.hpp:208-218`) — but it
always allocates, zeroes and accumulates the potential even when only the
gradient is wanted (`src/Canopy_Solver.hpp:209-211`; P2P accumulates
`phi[c]` unconditionally, `src/Canopy_P2P.hpp:891`). See **C8**.

Keeping the contraction *out* of the kernel is also the right design and not
merely the available one: the cross product is linear in the source strength, so
it commutes with any expansion of the kernel. The reference treecode bakes the
cross product into its own kernel evaluation
(`treecode.py:56-81`) and thereby loses that separation; a Canopy-side softened
operator (F6, C1) should produce the $3\times3$ tensor and let the caller
contract, exactly as this finding prescribes.

`NComps` is a **compile-time** template parameter, as is `P_ORDER`
(`src/Canopy_Solver.hpp:104-105`). A consumer whose expansion order is a runtime
configuration value cannot pass it through. See **C7**.

### F3 — Tree reuse across integrator stages

Answered against the real `setup`/maintenance/`solve` split. Three separate
problems, in increasing severity.

**(a) `solve()` after motion, with no maintenance, is silently wrong.** The
upward sweep runs P2M over the leaf lists cached at setup
(`src/Canopy_UpwardSweep.hpp:657-669`) and P2P uses the cached particle→leaf
mapping (`src/Canopy_P2P.hpp:819`, `:834`); the M2L interaction list is likewise
cached and only invalidated on a topology change
(`src/Canopy_Solver.hpp:564-565`, `609-610`). So a particle that has moved out of
its leaf still contributes to its old leaf's multipole and still gets its old
leaf's near-field list. No error is raised. The failure mode is a plausible,
slightly wrong field — the worst kind for a consumer whose only check is a
tolerance.

**(b) The cheapest maintenance path is not cheap.** `migrate()` — documented as
"cheapest maintenance … tree topology assumed unchanged"
(`src/Canopy_Solver.hpp:241-253`) — performs, per call:

- a full `TreeBuilder::build()` from current positions
  (`src/Canopy_Solver.hpp:269-271`), which is two `MPI_Allreduce` for the
  bounding box (`src/Canopy_TreeBuilder.hpp:345-346`) plus **one
  `MPI_Allreduce` per octree level** over the candidate-cell counts
  (`src/Canopy_TreeBuilder.hpp:743-744`), with the per-level candidate marshalling
  and the resulting globally-replicated cell list assembled on the host
  (`src/Canopy_TreeBuilder.hpp:659-780`);
- a host-side `std::unordered_set` over every cell key, twice, to detect
  topology change (`src/Canopy_Solver.hpp:262-286`);
- a particle redistribution (`src/Canopy_Solver.hpp:295-298`);
- then `_finish_topology_stable`, which builds the tree **again**, re-sorts the
  array by leaf, and re-runs all three sweep setups
  (`src/Canopy_Solver.hpp:625-645`).

There is no API for "the positions moved by less than a cell width; refresh only
the leaf membership and the P2P offsets". For a consumer paying this three times
per timestep, that is the difference between a usable and an unusable far field.
See **C3**.

**(c) On a deforming surface, `auto_maintain` will pick `Rebalance`, not
`Migrate`, almost every time.** The decision is: rebuild on bounding-box escape,
else `Rebalance` if *any* cell key differs from the previous tree, else
`Migrate` (`src/Canopy_Solver.hpp:362-466`). A rolling-up sheet changes its
occupancy pattern continuously, so the cell-key set changes essentially every
stage. `Rebalance` adds a Zoltan2 repartition and a full communication-plan
rebuild — and the communication-plan rebuild is a **serial host-side dual-tree
traversal over the globally replicated cell tree, executed on every rank**
(`src/Canopy_CommunicationPlan.hpp:549-665`). So the expensive path is the
common path, three times per step.

That path also makes the result **non-reproducible**. Zoltan2's `multijagged`
algorithm is non-deterministic, which Canopy handles by solving on rank 0 and
broadcasting (`src/Canopy_TreePartitioner.hpp:349-357`, `416-425`) — this makes
the assignment *consistent across ranks*, not *reproducible across runs*. Two
runs of the same problem on the same rank count can therefore get different
decompositions, hence different summation orders, hence different last bits. A
consumer whose direct-sum baseline is reproducible to $10^{-15}$ cannot make
any bitwise claim about the far-field path, and cannot distinguish a real
regression from a repartition reshuffle without a run-to-run noise measurement.
The `rcb` alternative is deterministic but is recorded as broken on the target
platform (`src/Canopy_TreePartitioner.hpp:417-419`). See **C4**.

**(d) Every maintenance path re-decomposes and permutes.** `partition`,
`repartition` and `redistribute` all migrate particles between ranks and reorder
the local array, and the order after migration is explicitly unspecified
(`src/Canopy_TreePartitioner.hpp:576-580`). Nothing in `src/` carries a
caller-supplied identity through that: the only `global_ids` in the tree are
leaf indices for the Zoltan2 adapter
(`src/Canopy_TreePartitioner.hpp:378-392`). Migration packs whole AoSoA tuples
(`src/Canopy_TreePartitioner.hpp:667-772`), so a caller-added identity member
*would* travel with its particle — but the caller must then run its own reverse
exchange to get results home, once per stage, in addition to Canopy's. For a
consumer whose decomposition is fixed by a mesh, this is the interface's
central mismatch. See **C2**.

### F4 — `ncrit`, `mac_theta`, `max_depth` on an unstructured sheet

| knob | in Canopy | notes for a consumer |
| --- | --- | --- |
| `ncrit` | `FmmConfig::ncrit` (`src/Canopy_Solver.hpp:54`), runtime. A cell becomes a leaf when its **global** count $\le$ `ncrit` or it hits `max_depth` (`src/Canopy_TreeBuilder.hpp:769`). | Same meaning as in any Barnes–Hut/FMM code; a value of 64 transfers directly. Note the refinement test is on the *global* count, so leaves are balanced in total occupancy, not per-rank occupancy. |
| `mac_theta` | `FmmConfig::mac_theta` (`:67`), runtime, default 0.5. The predicate is the exafmm spherical MAC: accept M2L iff $R\theta > \sqrt3\,(h_A + h_B)$ (`src/Canopy_CommunicationPlan.hpp:338-352`). | A **different predicate** from a Barnes–Hut opening angle, so an inherited numeric value does not carry its meaning across. It does carry its *direction*: smaller is more conservative, more M2L pairs, more accurate at fixed order. A value of 0.3 is conservative under Canopy's predicate too, and is exercised by one existing test (`tests/tstMultiSolve.hpp:692-697`). |
| `max_depth` | `FmmConfig::max_depth` (`:55`), runtime, **hard maximum 19**, enforced by a throw in the `TreeBuilder` constructor because the Morton key is a `uint64_t` (`src/Canopy_TreeBuilder.hpp:46`, `176-181`). | Has **no counterpart** in a treecode-derived parameter set; a consumer must choose it. It is coupled to the bounding box: the finest cell width is (root box width) / $2^{\text{max\_depth}}$. |

**Do the structured-workload values carry over to a sheet?** Partly, and the
part that does not is untested. The tree is *count*-adaptive, so a
two-dimensional source support in a three-dimensional box is handled without
special-casing: empty cells are dropped outright
(`src/Canopy_TreeBuilder.hpp:754-755`), and reaching $N/\texttt{ncrit}$
leaves on a surface costs roughly $\log_4(N/\texttt{ncrit})$ levels rather
than $\log_8$, i.e. a *deeper* tree than a volumetric distribution of the
same count — which is what makes the depth-19 ceiling worth checking rather than
assuming. Three sheet-specific effects have no coverage:

- **Anisotropic leaf occupancy.** A cube cell straddling a sheet has its sources
  on a plane through it, so the circumradius $\sqrt3 h$ used by the MAC
  overstates the actual source extent. The MAC stays conservative (safe), but
  the achieved accuracy at a given `mac_theta` is not the volumetric one, and no
  test measures it. **There is a cheap, kernel-independent improvement here.**
  The reference treecode uses the *exact* node radius $\max_j |y_j - c|$
  (`treecode.py:31`) where Canopy uses the geometric
  $\sqrt3\,(h_A+h_B)$ (`src/Canopy_CommunicationPlan.hpp:346`). The exact radius
  is tighter on a sheet by construction, is computable per cell in the upward
  sweep at negligible cost, and would reduce the number of pairs demoted to P2P
  at fixed accuracy — a win that is independent of every kernel question in F1
  and F6. It interacts with the near-softening floor, so it must be **measured**
  as part of **C5** rather than assumed.
- **Self-approach at roll-up.** Two along-surface-distant parts of the sheet come
  within $\sqrt b$ geometrically. `README.md:52-63` already names exactly
  this case — "a clustering system whose cells shrink below the softening length
  (e.g. a vortex sheet at full roll-up) gets a spurious, far too large far-field
  and blows up" — and the near-softening floor is the mitigation. So the
  mechanism is anticipated; what is unmeasured is the *cost*, since the floor
  converts a growing fraction of pairs to P2P as the sheet tightens.
- **Bounding-box sensitivity.** `TreeBuilder::compute_global_bounding_box` takes
  a raw global min/max (`src/Canopy_TreeBuilder.hpp:345-346`), and
  `README.md:296-307` records that a single outlier inflates the root box until
  a dense cluster collapses into one max-depth leaf, making the
  $O(N_{\text{leaf}}^2)$ P2P effectively hang. A surface that develops a
  single spurious vertex hits this.

**Every accuracy figure Canopy currently carries was measured on a uniform or
clustered *volumetric* random distribution, unsoftened.** `tests/tstSingleSolve.hpp`
and `tests/tstMultiSolve.hpp` place particles randomly in a box; neither has a
surface, sheet or manifold case (`grep -i "surface\|sheet\|manifold" tests/`
returns only unrelated prose). The validated envelope is:

| test | configuration | tolerance met |
| --- | --- | --- |
| `SingleSolve.PotentialAndGradientNComps3` | $P=8$, `ncrit` 16, `max_depth` 6, softening 0, 500 particles/rank, volumetric random; all nine gradient components vs. brute force (`tests/tstSingleSolve.hpp:79-87`, `365-375`, `424-431`) | $10^{-3}$ — but **fails at exactly 4 ranks**, see F5 |
| `MultiSolve` suite | $P=8$ (`tests/tstMultiSolve.hpp:52`), `mac_theta` 0.3–0.5, `ncrit` 8–16, `max_depth` 6–8, `softening = 0` (`:332`, `:781`) | $10^{-2}$–$3\times10^{-2}$ (`tests/tstMultiSolve.hpp:565-697`) |

There is therefore **no measured parameter set for a sheet, and no measured
accuracy figure with softening enabled at all**. Producing one requires
compiling and running; it cannot be settled by reading. This is task **C5**, and
it is the task that decides whether the consumer's whole approach is viable.

### F5 — Open defects on this path

**The `Rebalance` NIC-registration-cache defect is fixed in the tree surveyed.**
`TreePartitioner::migrate_particles` no longer routes through
`Cabana::Distributor`/`Cabana::migrate`; it packs outgoing tuples into per-peer
subviews of **one** persistent registered send region and posts one
`MPI_Isend` per peer, with matching receives in one persistent recv region, so
peak concurrent registrations are $O(1)$ per direction regardless of peer
count (`src/Canopy_TreePartitioner.hpp:547-580`,
`src/Canopy_RegisteredBufferPool.hpp:24-56`). The same pooling already covers the
M2L/L2L `coalesced_view_exchange` and the P2P ghost gather. `README.md:235-257`
records this as fixing the `dreg_evict NO_SPACE` deadlock on many-way
migrations, and as removing the need for a patched Cabana fork (the MPI element
type is one whole tuple, so a single peer's payload may exceed 2 GiB without
overflowing MPI's signed-`int` count).

This mattered acutely for this consumer, because per F3(c) `Rebalance` is its
*common* path rather than a rare one. As fixed, the defect does not affect it.
One residual remains, and it is a scaling concern rather than a defect: peer
discovery in migration is a single `MPI_Alltoall` of `comm_size` ints on every
`Rebalance` (`README.md:315-333`), which for a three-stage integrator is three
$O(\text{comm\_size})$ collectives per timestep on top of everything else.
That is folded into **C3** as motivation, not raised as its own task.

**Two other open defects sit on or beside this path.**

- **The exact code path this consumer needs is out of the regression gate.**
  `SingleSolve.PotentialNComps3` and `SingleSolve.PotentialAndGradientNComps3`
  — three components, gradient, versus brute force — fail at exactly 4 ranks
  (`max_pot_rel_err = 0.00207` vs. a $10^{-3}$ budget), pass at 1, 2, 3, 5
  and 6, and the whole `SingleSolve` binary additionally leaks state that
  deadlocks a later test in the same `ctest` process; it is consequently labelled
  `unit`, not `regression` (`README.md:363-392`). So the multi-component
  gradient path — the one thing this consumer's correctness rests on — is
  neither gated nor correct at one of the rank counts it will be run at. This is
  **C6**.
- The FP32 fused-M2L failure at $\ge 2$ ranks (`README.md:340-361`) does
  **not** affect a consumer running in double precision, and no task is raised
  for it here.

**One coverage gap rather than a defect:** every test in `tests/` gives every
rank the same non-zero particle count. A consumer distributing an unstructured
mesh can have a rank that owns **zero** sources, and must still enter every
collective. Nothing in `src/` obviously mishandles it — the per-level
`MPI_Allreduce` sums zero local counts, the replicated cell list is identical on
every rank, and the Zoltan2 solve runs on rank 0 over the global leaf set — but
"reads as though it should work" is not coverage. This is **C10**.

### F6 — What a softened far field would actually require

F1 says the far field expands the bare $1/r$. This finding answers the
question C1 originally left open: **can the softening be carried into the
existing operators?** No. It is a change of expansion basis, and it costs more
than "add $b$ to a few denominators".

**(a) Harmonicity is why no substitution exists.** Canopy's M2M/M2L/L2L are the
solid-harmonic addition theorems (Greengard Thms 5.22, 5.23, 5.26, cited at
`src/Canopy_LaplaceKernel.hpp:273`, `:373`, `:688`). Those theorems hold
*because* $1/r$ is harmonic. The softened potential is not:

$$
\nabla^2 (r^2+b)^{-1/2} \;=\; -\,\frac{3b}{(r^2+b)^{5/2}} \;\ne\; 0 .
$$

So there is no softened coefficient one can feed to `m2l_translate` to make it
evaluate the softened kernel. A Cartesian Taylor basis needs only *smoothness*
of the kernel, never harmonicity, which is why treecodes soften trivially — they
do not solve this problem, they sidestep it. Any softened far field in Canopy
therefore means a **second expansion basis alongside the existing one**, not a
patch to it.

**(b) A second basis is a wide, mechanical change through both sweeps.** The
sweeps *are* templated on `KernelType`
(`UpwardSweep<MemorySpace, ExecutionSpace, KernelType>`,
`src/Canopy_Solver.hpp:116-119`), so a new kernel struct is pluggable in
principle — but they hard-assume this kernel's storage: `complex_type***` views
(`src/Canopy_UpwardSweep.hpp:63,73`; `src/Canopy_DownwardSweep.hpp:108,118`),
`num_coeffs_per_cell = (P+1)(P+2)/2` (`:66`, `:111`), and the shared
$A_{n,m}$ table built to $2P$ (`src/Canopy_UpwardSweep.hpp:235`). A
Cartesian-Taylor kernel wants **real** symmetric-tensor storage of size
$(p{+}1)(p{+}2)(p{+}3)/6$ and no A-table. It fits only by generalizing a
`coeff_type` typedef through both sweeps, or by wasting every imaginary half.
`src/Canopy_Solver.hpp:112` also fixes `kernel_type` as a typedef rather than a
template parameter (F1).

**(c) Softening destroys scale invariance, and that is the expensive part.**
This is the obstruction most likely to be underestimated. Canopy's numerical
conditioning strategy rests on $1/r$ being **homogeneous of degree $-1$**.
Every operator is scale-normalized against cell half-width on that basis: P2M
produces $\bar M = M/w^{n+1}$ (`src/Canopy_LaplaceKernel.hpp:226-229`), M2M
applies $(w_c/w_p)^{j+1}$ (`:267-272`), M2L expands $\rho^{-(n+j+1)}$ as
$(w_s/\rho)^{n+1}(w_t/\rho)^j$ (`:369-372`), L2L applies $(w_c/w_p)^j$
(`:685-687`), L2P consumes $\bar L = L\,w^n$ (`:795-799`) — all for FP32
conditioning at depth.

The sharper consequence is the **precomputed M2L operator cache**. Operators are
keyed on $(dd, ii, jj, kk)$ — a depth difference and an integer offset in
half-widths — and `src/Canopy_DownwardSweep.hpp:1075` states the property
outright: the operator *"depends only on (dd, ii, jj, kk); no physical width
enters"*. That is true precisely because the kernel has no absolute length
scale. A softened kernel has one, $\sqrt b$, so:

- the cache key must additionally carry the physical cell width (or
  $w/\sqrt b$), multiplying the distinct-key count by the number of occupied
  depths and degrading the reuse the path exists to buy;
- all five width-normalization conventions above must be re-derived, because
  they are no longer mere rescalings.

**(d) A softened Cartesian FMM is only attractive at low order.** For a
Cartesian Taylor FMM truncated at multipole order $p_M$ and local order
$p_L$, M2L needs $\partial^\alpha\phi_b$ for
$|\alpha| \le p_M+p_L+1$. The reference treecode supplies closed-form
derivatives of the Plummer potential only to $|\alpha| \le 3$
(`treecode.py:56-81`) — genuinely the hard-to-get-wrong physics, and exactly the
seed of a Cartesian M2L, since a Cartesian M2L *is*
$\partial^{\alpha+\beta}K(R)$ — but that is order 2, not a ladder. Matching
Canopy's default $P=8$ needs derivatives to 17th order and
$\sim\!(19)(20)(21)/6 \approx 1330$ tensor slots per pair: the
$O(p^3)$-vs-$O(p^2)$ growth that is the standard reason solid harmonics
win at high order. **The basis that admits softening is the basis that does not
scale to high accuracy** — so a softened far field lives in the
$10^{-3}$ regime, not the $10^{-6}$ regime F1 wants.

**(e) Only one operator family is kernel-dependent, which bounds the work.** In
a Cartesian Taylor method, M2L (equivalently M2P) is the *only* operator that
knows what the kernel is; P2M, M2M, L2L and L2P are combinatorics on moments and
derivative coefficients (binomial and Taylor shifts) and need no
softening-specific derivation at all. So the derivation burden of a softened far
field is concentrated entirely in the M2L/M2P derivative tensors.

**(f) Literature to check before committing.** The precedent for a low-order
Cartesian-Taylor FMM with softening is **Dehnen's `falcON`** (W. Dehnen, *ApJ*
**536**, L39, 2000; *JCP* **179**, 27, 2002), a Cartesian Taylor-expansion FMM
built for softened gravity with a full M2L; Warren & Salmon's hashed oct-tree
work is the Cartesian-multipole precedent. **Neither has been read.** They are
named as the check C1 must run, not as verified results.

### F7 — Canopy's L2P gradient is a finite difference, not analytic

`src/Canopy_LaplaceKernel.hpp:851-879` evaluates the far-field gradient by a
central difference: six extra potential evaluations at
$h = 10^{-5} w_{\rm self}$, with an in-code `TODO: replace with analytical
derivatives` and a comment recording that a previously *fixed* step size was the
root cause of a premature full-rollup NaN. Two consequences:

- **Part of the gradient error C1 and C5 are about to measure is
  finite-difference error, not softening bias and not truncation.** At the
  roundoff/truncation optimum the relative FD error is
  $\sim\!10^{-10}$–$10^{-11}$, so it does not threaten a $10^{-3}$
  budget — but it is a **third** plateau in the `P_ORDER` scan that C1 step 1
  and risk **R1** are built around, and R1's "truncation falls, bias plateaus"
  discriminator has to account for it.
- A softened Cartesian far field (F6) would produce analytic gradients as a side
  effect, since its expanded quantity *is* $\nabla\phi_b$ and the derivative
  tensors are what it evaluates directly.

Replacing the FD with analytic solid-harmonic derivatives in the existing basis
is independent of every kernel question above and is task **C11**.

### F8 — There is no treecode / M2P mode, and adding one is the cheap option

`grep -i "barnes\|treecode\|m2p"` over `canopy/src/` returns nothing. There is
no opening-angle mode, no monopole mode, and no evaluate-a-multipole-at-a-point
operator anywhere; `mac_theta` is a spherical MAC used to *build M2L pairs*
(`src/Canopy_CommunicationPlan.hpp:338-352`), not a Barnes–Hut opening angle. So
a treecode is **not** a configuration of Canopy today.

It could become one, and that is the cheapest route to a far field that carries
the softening. The dual-tree traversal already produces accepted
**(target cell, source cell)** pairs (`src/Canopy_CommunicationPlan.hpp:549-665`)
and already has a working softened P2P for the rejected ones. A "treecode mode"
needs no new traversal, tree, partitioner or communication plan — it replaces
*one step*: where the downward sweep currently does M2L into a local expansion
and then L2L/L2P down to particles, evaluate the source cell's multipole
**directly at each particle in the target cell** (M2P) and skip L2L and L2P
entirely.

| piece | lift |
| --- | --- |
| Cartesian moments in the upward sweep (P2M) | Small. Kernel-independent; `treecode.py:33-35` is the formula. |
| M2M for those moments | Small. Binomial shift, kernel-independent — the one operator the reference Python skips (it recomputes moments per level) and a real implementation should have. |
| Softened M2P | **Small, and transcribable.** `treecode.py:56-81`, $\sim\!80$ lines of explicit index loops. |
| M2L / L2L / L2P | **Deleted from the path.** Not implemented, not needed. |
| Storage | Real, $(p{+}1)(p{+}2)(p{+}3)/6 \times$ `NComps` per cell. Needs F6(b)'s `coeff_type` generalization, but *only* in the upward sweep and the new M2P driver. |
| Operator cache | **Not applicable.** M2P has no per-offset operator to cache, so F6(c)'s loss of scale invariance costs nothing here. |

What it buys is exactly what F1 says is missing: the far field carries the
softening, `near_softening_factor` becomes unnecessary rather than load-bearing,
and F1's systematic $\tfrac32\varepsilon^2/R^2$ gradient bias — the one that
does not shrink with order — **disappears**, leaving ordinary truncation error
that *does* respond to `mac_theta` and order. It also yields analytic gradients
(F7) and is faithful to the consumer's reference algorithm and its
$\theta$/order/`ncrit` knobs.

What it costs, and neither number is measured:

- **Accuracy ceiling $\sim\!10^{-3}$** at $\theta=0.3$, order 2
  (carried from [`treecode.md`](treecode.md) §1), with the error a plateau in
  $N$. Higher order is available but pays F6(d)'s $O(p^3)$.
- **Complexity goes $O(N)\to O(N\log N)$**, and per-target work rises:
  every particle in a target cell re-evaluates every accepted source multipole
  instead of the cell paying M2L once and amortizing through L2L/L2P. On a leaf
  of `ncrit` particles that is an `ncrit`-fold increase in far-field arithmetic.
  **This is the real cost of the option and it is a measurement, not an
  estimate** — see C1 step 3 and risk **R7**.

Note the collision that reframes the whole choice: the honest accuracy claim for
the *status quo* (F1, $10^{-2}$–$10^{-3}$) and the ceiling of an M2P
mode are **the same number**. So M2P is not "trade accuracy for fidelity" — it
is "reach the same accuracy with an error controllable by `mac_theta` and order
instead of a fixed bias, plus analytic gradients and reference-faithful knobs".

It also bears on where a treecode for this consumer should live: an M2P mode
inherits Canopy's distributed tree, whereas a standalone port
([`treecode.md`](treecode.md) §3) must re-decide its own distribution strategy.

## Approach

Each finding above that blocks or degrades the consumer becomes one task below.
The tasks are independent except where stated: **C1** and **C5** together decide
whether the approach is viable at all and should be done first; **C2** and
**C3** are the interface changes that make a three-stage integrator affordable;
**C4**, **C6** and **C10** are correctness and confidence work; **C7**, **C8**,
**C9** and **C11** are small API and quality items that can be taken at any
time, though C11 is worth doing *before* C1's scan is interpreted.

### Conventions

| Choice | Rule |
| --- | --- |
| Library style | header-only under `src/`, `Canopy_` prefix, `namespace Canopy` (`detail` for internals, as `src/Canopy_RegisteredBufferPool.hpp:21-23`) |
| Parallelism | Kokkos + Cabana + MPI; no serial-only signatures |
| Configuration | one new knob goes in `FmmConfig` (`src/Canopy_Solver.hpp:52-78`) with a defaulted member and a comment stating units and the meaning of the default; never a new constructor parameter |
| New parameters | prefer an enum or tag type over a bool or magic number; a mode selector is an enum |
| Failure behavior | a violated precondition throws (`std::runtime_error`, as `src/Canopy_TreeBuilder.hpp:176-181`); never return a truncated or best-effort field |
| Comments | state units, sign convention, and which side of a difference is which, on the declaration; the sign of the gradient output is the single most misread thing in this API |
| Provenance | cite the paper, section or upstream code any new operator is derived from, on the routine |
| Test tier | new correctness tests are `regression` and must pass at ranks 1–6; a test that cannot yet pass at all six is `unit` **and** its exclusion is recorded in `README.md` "Known Issues" with the rank counts that fail |
| Accuracy claims | every stated tolerance names the distribution, the rank counts, `P_ORDER`, `ncrit`, `max_depth`, `mac_theta` and the softening it was measured at. A tolerance without that list is not a claim |

### Deliberate deviations

- **No task proposes a general kernel abstraction.** The consumer's kernel is
  reachable as a set of Laplace gradient solves (F2), so templating the whole
  pipeline on a kernel concept would be a large refactor bought for nothing. C1
  extends the *existing* pipeline to carry softening in the far field rather
  than making the kernel pluggable — and per F6(b) even that needs only a
  `coeff_type` generalization of the two sweeps, not a kernel concept.
- **No task proposes making the partitioner's non-determinism disappear by
  reverting to `rcb`.** It is recorded as broken on the target platform
  (`src/Canopy_TreePartitioner.hpp:417-419`); C4 addresses reproducibility
  without that assumption.
- **The consumer's configuration surface is fixed and cannot absorb these
  gaps.** No task below may be closed by asking the consumer to add a knob.

## Current state

Everything described in **Findings** is the state of the library as surveyed.
Concretely, and stated as what is *not* true:

- The far-field operators do not carry softening, and no diagnostic reports the
  resulting bias. A consumer gets a wrong-by-a-known-formula answer with no
  indication. All five far-field operators *are* implemented — in the
  solid-harmonic basis, which cannot carry softening at all (F6).
- There is no accuracy measurement for any softened configuration, and none for
  any non-volumetric source distribution.
- There is no M2P / treecode / opening-angle mode (F8), so there is no existing
  path whose far field is softening-consistent.
- The far-field gradient is a finite difference of the potential, not an analytic
  derivative (F7).
- There is no way to get results back in the caller's particle order or on the
  caller's ranks.
- There is no maintenance path cheaper than a full global tree build.
- The three-component gradient path is not in the regression gate and is known
  wrong at 4 ranks.
- `solve()` called after motion without maintenance returns a
  **defined-but-wrong** field rather than raising. This is the most dangerous
  single property of the current API for a new consumer.

## Progress log

`tasks/canopy0-progress-log.md` holds what actually happened: the reasoning behind
decisions this document states flatly, measured numbers, and things only running
revealed. **Read it before implementing any task, changing any signature, or
reopening a question this document treats as settled** — in particular before
choosing a tolerance, since a measured number in the log always outranks an
estimate here.

## Task sequence

### C1 — Make the far field consistent with the softened kernel, or bound the bias — **NOT STARTED**

**Depends on:** none. (Interacts with C11: C11 removes one of the three error
plateaus step 1 will see, so doing C11 first makes the scan easier to read.)

**Fill in:** `src/Canopy_LaplaceKernel.hpp` (`m2l_translate`, `l2p_evaluate`),
`src/Canopy_CommunicationPlan.hpp` (`mac_satisfied`, `set_near_softening`),
`src/Canopy_Solver.hpp` (`FmmConfig`), `README.md`, plus one new test. Option (d)
below additionally touches `src/Canopy_UpwardSweep.hpp` and
`src/Canopy_DownwardSweep.hpp`.

**Reference:** the softened near-field kernel
(`src/Canopy_P2P.hpp:799-803`, `883-896`); the floor that keeps the unsoftened
far field usable (`src/Canopy_CommunicationPlan.hpp:347-361`); the error bound
and its potential-vs-gradient factor of three, tabulated in **F1**; what a
softened far field costs, in **F6**; the M2P option and its unmeasured cost, in
**F8**; the finite-difference gradient, in **F7**.

**Do:**
1. **Measure first.** Build a test that evaluates a softened configuration
   (`softening = 2.5e-2`, domain extent $O(1)$) against a brute-force sum of
   the *same softened* kernel, reporting max relative error on **both** the
   potential and all `NComps × 3` gradient components, as a function of
   `near_softening_factor` over at least {4, 8, 16, 32} and of `P_ORDER`. The
   point is to show the gradient error plateauing with `P_ORDER` — that plateau
   is the softening bias, and its independence from `P_ORDER` is what
   distinguishes it from truncation error. Read the scan against F7: the
   finite-difference L2P contributes a *second*, much lower plateau
   ($\sim\!10^{-10}$), so "a plateau" is not by itself the softening bias.
2. Record the measured plateau and the P2P pair-count growth in the progress
   log, and put the achievable tolerance in `README.md` next to
   `near_softening_factor`, correcting the quoted bound to state that it is the
   **potential's** and that the gradient's is three times larger.
3. **Then decide, with the numbers in hand,** between four options — noting that
   (a) and (d) land on the *same* accuracy, $\sim\!10^{-3}$, so the choice
   between them is about the *kind* of error, not its size:
   (a) **Status quo.** Accept the bias, document it as the floor on far-field
   fidelity, and add a diagnostic that reports the predicted bias for the
   configured `softening` / `near_softening_factor`.
   (d) **A softened M2P mode over the existing interaction list** (F8). Cartesian
   moments in the upward sweep, kernel-independent M2M, a softened M2P
   transcribed from `treecode.py:56-81`, and M2L/L2L/L2P dropped from the path.
   Reuses the tree, partition, communication plan and P2P unchanged; removes the
   bias entirely; yields analytic gradients. Costs $O(N)\to O(N\log N)$ and
   an `ncrit`-fold rise in per-target far-field arithmetic, and caps accuracy at
   $\sim\!10^{-3}$.
   (b) **A full softened Cartesian-Taylor FMM.** Everything in F6(b)(c)(d): a
   second expansion basis, real storage through both sweeps, M2L derivative
   tensors to $p_M+p_L+1$, and a re-keyed M2L operator cache. Retains
   $O(N)$ and the L2L/L2P amortization; still capped at low order by F6(d).
   **Substantially larger than this task's original framing suggested**, and
   worth doing only after (d) has measured whether a softening-consistent far
   field actually buys the accuracy the consumer needs.
   (c) **Do nothing in Canopy** and port a standalone treecode into the consumer
   ([`treecode.md`](treecode.md) §3). Does not close this task; recorded so the
   option is not rediscovered.
   The measurement that decides between (a)/(d) and (b) is the per-target cost of
   M2P against the per-cell cost of M2L+L2L+L2P on this problem and platform. It
   needs a build and a run, nobody has it, and it should be taken as part of this
   step. Record the decision and its reasoning in the log **before** implementing
   anything.

**Additional information needed — partly answered.** The original question was
"whether option (b) is achievable without replacing the expansion basis". **F6
answers that: no.** The solid-harmonic addition theorems require harmonicity,
which the Plummer potential lacks, so any softened far field is a second basis
alongside the existing one — plus a loss of scale invariance that re-keys the M2L
operator cache and re-derives five width normalizations. What remains open is the
literature check: **whether Dehnen 2000/2002 in fact gives a usable softened
Cartesian M2L, and to what order** (F6(f)). Those papers are named from memory
and have not been read; doing so belongs to this task.

**Exit criterion:** a new test in the `regression` tier passes at ranks 1–6 and
asserts a **stated, measured** relative-error bound on all gradient components
for a softened configuration; and it fails, for the softening-bias reason
specifically, when `near_softening_factor` is set to 1 — demonstrating the test
is sensitive to the effect it exists to bound, rather than passing on slack.
`README.md` states the achievable far-field fidelity for the gradient. If option
(d) or (b) is taken, the same test must additionally show the bias plateau *gone*
rather than merely small.

---

### C2 — Return results in the caller's particle order, on the caller's ranks — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (public API and `_full_setup`,
`_finish_topology_change`, `_finish_topology_stable`),
`src/Canopy_TreePartitioner.hpp` (retain the forward map already computed by
`migrate_particles` and `sort_particles_by_leaf`), `README.md`.

**Reference:** `src/Canopy_TreePartitioner.hpp:547-580` (migration semantics and
the explicit "order after migration is unspecified"); `:667-772` (the pack/unpack
that already knows every particle's destination); `:378-392` (the only existing
`global_ids`, which are leaf indices, not particle identities);
`src/Canopy_Solver.hpp:519-525` (`sort_particles_by_leaf`, the second reordering).

**Do:**
1. Decide and record which of two shapes to expose: **(i)** an inverse map —
   Canopy retains, per local particle, the origin rank and origin index, and
   exposes a method that scatters `potential()`/`gradient()` back to the
   caller's pre-`setup` layout; or **(ii)** a caller-identity passthrough — a
   documented convention that the caller adds an identity member to its AoSoA
   and Canopy guarantees it travels intact, plus a helper that builds the
   reverse exchange from it. (i) is the smaller API and the larger internal
   change; (ii) is the reverse. Recommendation: **(i)**, because the destination
   information already exists inside `migrate_particles` at the moment it is
   needed and is thrown away, whereas (ii) makes every consumer reimplement the
   same exchange.
2. Implement it so it composes with *every* path that reorders: `setup`,
   `rebuild`, `rebalance`, `migrate`, `auto_maintain`, and the
   `sort_particles_by_leaf` permutation inside each. A map that is correct after
   `setup` and stale after `auto_maintain` is worse than none.
3. State on the declaration whether the returned ordering is the caller's
   ordering *at the most recent setup* or *at construction*, and which ranks the
   values land on.

**Exit criterion:** a `regression` test at ranks 1–6 in which each rank creates
particles with a known per-rank identity, runs `setup`, then `auto_maintain`
enough times to force at least one `Rebalance` (assert the returned
`MaintenanceAction`), and recovers `gradient()` values matching a brute-force
reference **indexed by the caller's original local index on the caller's
original rank**; and the same test fails with a clear error, not a wrong answer,
if the scatter-back is requested before any `setup`.

---

### C3 — A cheap per-evaluation refresh for multi-stage time integrators — **NOT STARTED**

**Depends on:** none. (Interacts with C2: if C2 lands first, the refresh must
keep C2's map valid.)

**Fill in:** `src/Canopy_Solver.hpp` (new public method plus the internals it
needs), `src/Canopy_TreeBuilder.hpp` (a keys-only recompute against the existing
cell list), `src/Canopy_P2P.hpp` / `src/Canopy_UpwardSweep.hpp` (refresh cached
per-leaf offsets without a full `setup`), `README.md`.

**Reference:** `src/Canopy_Solver.hpp:241-305` (`migrate`, and what it actually
costs); `:625-645` (`_finish_topology_stable`, the second full build);
`src/Canopy_TreeBuilder.hpp:659-780` (per-level `MPI_Allreduce` and host-side
cell-list assembly); `src/Canopy_CommunicationPlan.hpp:549-665` (the serial
host-side dual-tree traversal run on every rank per plan rebuild);
`src/Canopy_TreeBuilder.hpp:258-264` (`apply_particle_permutation`, precedent for
updating keys without a rebuild); `README.md:315-333` (the
$O(\text{comm\_size})$ `MPI_Alltoall` per `Rebalance`).

**Do:**
1. Add a maintenance path cheaper than `migrate()` for the case "positions moved;
   the cell list is unchanged and every particle is still in a cell owned by
   this rank". It must recompute particle→leaf keys against the *existing* cell
   list and refresh the cached leaf offsets, with **no** tree rebuild, **no**
   repartition, **no** communication-plan rebuild, and — in the common case — no
   particle exchange at all.
2. It must verify its own precondition rather than assume it. If any particle
   left its owning rank's cells, fail over to the existing path (and say which,
   via the returned `MaintenanceAction`) rather than returning a wrong field.
3. Do not change `migrate`/`rebalance`/`rebuild` semantics; this is an addition.
   `auto_maintain` gains this as its new cheapest branch, ahead of `Migrate`.

**Exit criterion:** a `regression` test at ranks 1–6 that performs three
successive small displacements and evaluations per step (the three-stage
integrator pattern) and asserts (a) the new path is selected for each — via the
returned `MaintenanceAction` — and (b) the resulting gradient matches, to the
tolerance C5 establishes, the result of a full `rebuild()` + `solve()` at the same
positions; plus a case where a particle is displaced across a cell boundary and
the assertion is that the path *declines* and reports the fallback it took.

---

### C4 — Reproducible results across runs at fixed rank count — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_TreePartitioner.hpp` (`partition_leaves` and the
assignment broadcast), `src/Canopy_Solver.hpp` (`FmmConfig`), `README.md`.

**Reference:** `src/Canopy_TreePartitioner.hpp:349-357` and `:416-425` (rank 0
solves the non-deterministic `multijagged` problem and broadcasts; `rcb` is
deterministic but recorded as broken on the target platform); `:190-195`
(`_cached_leaf_owners`, which already exists to avoid a second, differing
Zoltan2 call).

**Do:**
1. Measure and record the actual run-to-run spread first: run the same problem
   twice at the same rank count and report the max relative difference in
   `gradient()`. If it is at rounding level the task may reduce to documenting
   that; if it is not, continue.
2. Add a documented reproducible mode — a deterministic assignment derived from
   the (globally identical) leaf list, e.g. a Morton-order or
   space-filling-curve split with the same imbalance tolerance, selected by an
   enum in `FmmConfig` rather than a bool. Keep `multijagged` as the default if
   it load-balances better; the point is that a consumer validating against a
   reference can *opt into* determinism.
3. Document, next to the new knob, that determinism of the assignment does not
   by itself give bitwise reproducibility across *rank counts* — reduction order
   still differs — and say which of the two the mode guarantees.

**Exit criterion:** a `regression` test at ranks 1–6 that runs the same
configuration twice in the reproducible mode and asserts the two `gradient()`
results are **bitwise** identical; and that asserts the default mode is *not*
required to be, so the test does not silently start gating `multijagged`.

---

### C5 — Accuracy on a two-dimensional source distribution, and a validated parameter set — **NOT STARTED**

**Depends on:** none. Should be read together with C1 — C1 varies the softening
at fixed distribution; C5 varies the distribution.

**Fill in:** a new test in `tests/` plus its `tests/CMakeLists.txt` registration;
`README.md` (a validated-parameters table). Step 5 additionally touches
`src/Canopy_UpwardSweep.hpp` and `src/Canopy_CommunicationPlan.hpp`.

**Reference:** `tests/tstSingleSolve.hpp:79-87` (the brute-force comparison
harness to reuse, including its rank-0 gather), `:365-375` (how the tolerance is
asserted), `:389-431` (the existing parameter choices);
`tests/tstMultiSolve.hpp:52` (`P_ORDER = 8`), `:332` and `:781`
(`softening = 0`), `:565-697` (the tolerances currently met);
`src/Canopy_TreeBuilder.hpp:176-181` (the depth-19 ceiling this task must check
against a deeper, surface-driven tree); `README.md:296-307` (the
bounding-box outlier limitation, which a surface with one stray source hits);
`src/Canopy_CommunicationPlan.hpp:346` (the geometric $\sqrt3(h_A+h_B)$
source-extent bound) and `treecode.py:31` (the exact $\max_j|y_j - c|$
alternative), per **F4**.

**Do:**
1. Add a source distribution on a **two-dimensional manifold embedded in three
   dimensions** — a sphere is sufficient and is trivially generated — with
   sources on the surface only, `NComps = 3`, `compute_gradient = true`, softening
   set to a physically-motivated non-zero value, compared against a brute-force
   sum of the softened kernel.
2. Sweep `ncrit`, `mac_theta`, `max_depth` and `P_ORDER` and record the achieved
   max relative gradient error for each combination in the progress log. Include
   at least one case where the required depth for the target `ncrit` approaches
   the ceiling, and report the depth actually reached.
3. Add a self-approaching case: two surface patches brought to within a few
   times the softening length, so the near-softening floor engages. Report both
   the error and the P2P pair count, since the cost is the thing that decides
   viability here.
4. Publish the resulting validated parameter set in `README.md`, with the full
   qualification list the conventions table requires.
5. **Measure the exact-node-radius alternative** (F4). Compute
   $\max_j |y_j - c|$ per cell in the upward sweep, use it in `mac_satisfied`
   in place of $\sqrt3\,h$, and report — on the same surface cases as steps 1–3
   — the change in achieved gradient error *and* in P2P pair count. It is
   kernel-independent and expected to help on a sheet, but it interacts with the
   near-softening floor, so it must be measured rather than assumed; if it wins,
   it becomes a `FmmConfig` mode enum per the conventions table, not a silent
   change of predicate.

**Exit criterion:** a `regression` test passes at ranks 1–6 asserting a stated
max relative gradient error for a surface distribution with non-zero softening;
`README.md` carries the validated `(ncrit, mac_theta, max_depth, P_ORDER,
softening, near_softening_factor)` set and the error it achieves; the test
fails when `mac_theta` is loosened by 2× — showing it is measuring the
approximation rather than passing on a slack budget; and the progress log carries
the exact-radius-vs-geometric-radius comparison from step 5, with a recorded
decision either way.

---

### C6 — Return the three-component gradient path to the regression gate — **NOT STARTED**

**Depends on:** none.

**Fill in:** whatever the np=4 investigation implicates — most likely
`src/Canopy_TreePartitioner.hpp` or `src/Canopy_CommunicationPlan.hpp`; plus the
teardown path exercised by `tests/tstSingleSolve.hpp`; plus
`tests/CMakeLists.txt` (label change) and `README.md` (removing the Known Issue).

**Reference:** `README.md:363-392` (both symptoms: a $2\times$-over-budget
accuracy failure at exactly 4 ranks, and a state leak that deadlocks a later
test in the same `ctest` process); `tests/tstSingleSolve.hpp:365-375` (the
assertion and its $10^{-3}$ budget); `:424-431` (the two failing cases).

**Do:**
1. Fix the np=4 accuracy failure. The rank-count-specific signature points at a
   partition or decomposition edge case rather than at the expansion; the first
   diagnostic worth running is whether the np=4 leaf assignment produces an
   ownership or replication pattern absent at 3 and 5 ranks. Determine whether
   the correct outcome is a bug fix or a re-justified budget, and record which
   with the evidence — do **not** simply widen the tolerance.
2. Fix the teardown so a failed solve cannot poison a later test in the same
   process.
3. Relabel the suite `regression` and remove the Known Issue entry.

**Exit criterion:** `ctest -L regression` passes at ranks 1–6 with the
three-component potential-and-gradient cases included, in the same `ctest`
invocation as the rest of the suite (no deadlock); and `README.md` no longer
carries the `SingleSolve` Known Issue.

---

### C7 — Expansion order and component count selectable at runtime — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`createSolver`, or a new dispatch),
`README.md`.

**Reference:** `src/Canopy_Solver.hpp:104-112` (`P_ORDER` and `NComps` are
template parameters and `kernel_type` is fixed); `:719-727` (`createSolver`, the
existing factory, which inherits the same template parameters).

**Do:** provide a factory that accepts an expansion order — and optionally a
component count — as **runtime** values and dispatches to a documented,
explicitly enumerated set of instantiations, throwing for a value outside that
set. Do not template the entire consumer-visible API on a value the consumer
holds at runtime, and do not silently round an unsupported order to a supported
one. Document the supported set and the compile-time cost of extending it.

**Exit criterion:** a `unit` test constructs a solver for each supported order
through the runtime factory and gets results identical to the directly
instantiated template, and gets a thrown exception naming the supported set for
an unsupported order.

---

### C8 — Gradient-only solve — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`solve`), `src/Canopy_P2P.hpp`,
`src/Canopy_DownwardSweep.hpp`, `README.md`.

**Reference:** `src/Canopy_Solver.hpp:202-239` (`solve` always allocates and
zeroes the potential, and `compute_gradient` is the only selector);
`src/Canopy_P2P.hpp:891` (`phi[c]` is accumulated unconditionally inside the
innermost pair loop).

**Do:** replace the `bool compute_gradient` parameter with an enum selecting
`Potential`, `Gradient`, or `Both`, and skip the potential's allocation,
zeroing, and per-pair accumulation when it is not requested. Enumerate and
update all callers of `solve()`, `DownwardSweep::execute` and `P2P::execute` —
including every test in `tests/` and every example in `examples/` — rather than
adding an overload alongside the bool.

Note the interaction with **F7**: today the far-field gradient is computed *from*
potential evaluations, so a `Gradient`-only far-field path cannot skip the
potential internally until C11 lands. Skipping the potential's *output*
allocation, zeroing and P2P accumulation is still valid and is what this task
asks for.

**Exit criterion:** an existing gradient test passes unchanged through the new
enum, a `Gradient`-only solve leaves `potential()` zero-extent, and the full
`unit` + `regression` suites build and pass at ranks 1–6 with no remaining
`bool` overload in the tree.

---

### C9 — `FmmConfig` cannot be default-constructed safely — **NOT STARTED**

**Depends on:** none.

**Fill in:** `src/Canopy_Solver.hpp` (`FmmConfig`), `README.md`.

**Reference:** `src/Canopy_Solver.hpp:52-56` — every other member of
`FmmConfig` has a default initializer; `ncrit` and `max_depth` do not, so
`FmmConfig cfg;` followed by setting only some fields reads uninitialized
memory and builds an arbitrary tree. The README's parameter table lists their
defaults as "—" (`README.md:39-40`), which documents the hazard rather than
removing it.

**Do:** give both members either a defensible default initializer or a value
that is unambiguously invalid and checked in the `Solver` constructor with a
throw naming the unset field. Prefer the latter, since no default value for
`max_depth` is correct independently of the domain.

**Exit criterion:** a `unit` test constructs `FmmConfig` without setting `ncrit`
or `max_depth`, passes it to the `Solver` constructor, and gets a thrown
exception naming the unset field — rather than a built tree.

---

### C10 — A rank that owns zero sources — **NOT STARTED**

**Depends on:** none.

**Fill in:** a case added to an existing test in `tests/`; whatever `src/` path it
implicates.

**Reference:** every test in `tests/` gives every rank the same non-zero
`num_particles_per_rank` (e.g. `tests/tstSingleSolve.hpp:389-431`), so the
zero-particle rank is uncovered. The paths it must survive are the per-level
`MPI_Allreduce` over candidate counts (`src/Canopy_TreeBuilder.hpp:743-744`),
the bounding-box reduction (`:345-346`), the rank-0 Zoltan2 solve and broadcast
(`src/Canopy_TreePartitioner.hpp:416-425`), and the P2P and ghost-gather loops
over a zero-length local set (`src/Canopy_P2P.hpp:834-838`).

**Do:** add a case where at least one rank — including, in one variant, rank 0
itself, since it carries the partitioning solve — starts with zero local
particles, and one where a rank is left with zero after migration. Assert
completion and correctness, not merely absence of a hang. If the bounding-box
reduction over an empty local set is what breaks, fix it there rather than
special-casing the caller.

**Exit criterion:** a `regression` test passes at ranks 2–6 with one rank
holding zero particles at `setup` (and, in a second variant, rank 0 holding
zero), producing gradients on the non-empty ranks that match a brute-force
reference; the test times out or fails, rather than silently passing, if a
collective is skipped on the empty rank.

---

### C11 — Analytic far-field gradient in place of the finite difference — **NOT STARTED**

**Depends on:** none. Worth doing **before** C1 step 1's scan is interpreted, so
that scan has two error sources to separate rather than three.

**Fill in:** `src/Canopy_LaplaceKernel.hpp` (`l2p_evaluate`), plus whichever
existing tolerance in `tests/` moves; `README.md` if a stated accuracy changes.

**Reference:** `src/Canopy_LaplaceKernel.hpp:851-879` — the six-point central
difference at $h = 10^{-5} w_{\rm self}$, its in-code
`TODO: replace with analytical derivatives`, and the comment recording that a
previously fixed step size caused a premature full-rollup NaN; `:795-799` (the
$\bar L = L\,w^n$ normalization the analytic form must respect); **F7**.

**Do:** differentiate the local expansion analytically in the solid-harmonic
basis and evaluate the gradient directly, citing the identity used on the routine
per the conventions table. Keep the width normalization consistent with
`l2p_evaluate`'s existing $\bar L$ convention. Remove the step-size heuristic
and the TODO. State on the declaration the sign convention of the returned
gradient, since F2 records that as the most misread thing in the API.

**Exit criterion:** the existing gradient tests pass at ranks 1–6 with tolerances
no looser than today's; a `unit` test shows the analytic gradient agreeing with
the current finite difference to the finite difference's own accuracy
($\sim\!10^{-8}$ relative or better) on a case where both are computable; and
the progress log records the measured error floor before and after, so C1's scan
can be read against it.

## Known risks

**R1 — The softening bias is mistaken for expansion error, and "fixed" by
raising `P_ORDER`.** Both present as a relative-error figure above budget.
The distinguishing measurement is the *scan*: truncation error falls as
`P_ORDER` rises; the softening bias plateaus. C1 step 1 exists to produce that
scan, and no tolerance anywhere should be set before it has been read. Note the
scan has **three** components, not two: per F7 the finite-difference L2P
contributes its own, much lower ($\sim\!10^{-10}$) plateau, so "the error
stopped falling" is not by itself evidence of softening bias. C11 removes that
confound.

**R2 — A tolerance gets tuned to hide C6's np=4 defect.** The np=4 failure is
$\approx 2\times$ over a $10^{-3}$ budget — close enough that widening
the budget looks defensible. It is not, until the rank-count-specific mechanism
is understood: an error that appears at exactly one rank count is a
decomposition bug signature, not a budget signature. C6 requires the evidence
either way.

**R3 — The partitioner's non-determinism masquerades as a regression.** Once a
consumer compares against a reference, a re-partitioned run differs in the last
bits for a legitimate reason. Without C4's measured run-to-run spread there is
no way to tell that apart from a real change, and the likely outcome is a
tolerance loosened until the noise fits — which then hides real regressions of
the same magnitude. Measure the spread before setting any tolerance.

**R4 — C3's cheap path is implemented as an optimistic one.** The dangerous
version of C3 assumes its precondition and returns a wrong field when it is
violated — exactly the failure mode that already exists for `solve()` after
motion (F3(a)). Its exit criterion therefore requires a case where the
precondition is violated *and the path declines*, not merely a case where it
succeeds.

**R5 — C5 measures a distribution that is not the hard case.** A uniform
sphere is a two-dimensional distribution but it is not self-approaching, and the
self-approach is where the softening, the tree depth, the near-field cost and
the bounding box all degrade at once. A C5 that reports a clean number on a
smooth sphere and stops has answered the easy half; step 3 is the half that
decides viability.

**R6 — A gap gets closed on the consumer's side instead.** Several findings here
(C1's bias, C2's reordering, C3's cost) can be worked around by the consumer at
the price of a wrong answer, a duplicated exchange, or a tripled cost per
timestep. Those are not closures. Each task above is either done in this library
or accepted by name, with the accepted consequence written down.

**R7 — C1's option (d) is chosen or rejected on an estimate rather than the
measurement.** M2P's whole viability rests on one unmeasured number: the
per-target cost of evaluating every accepted source multipole against the
per-cell cost of M2L+L2L+L2P amortized through the leaf. F8 states the direction
(an `ncrit`-fold rise in far-field arithmetic, $O(N)\to O(N\log N)$) but no
magnitude, and the magnitude is what decides. Equally, "M2P is obviously too
slow" is an estimate too. Build it small and measure before deciding either way.

**R8 — The $10^{-6}$ target survives unexamined.** Every option in C1 lands
at $\sim\!10^{-3}$ except (b) at moderate order, and F6(d) shows that even a
*softened* far field is out of reach of $10^{-6}$ at low order. If
$10^{-6}$ is a hard consumer requirement rather than an aspiration, the only
remaining path is (b) at an order high enough to pay F6(d)'s $O(p^3)$, and
that should be costed before anything here is built. Establish what the consumer
actually needs first; it is the cheapest question on this list to answer.
