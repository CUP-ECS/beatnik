# Kernel-independent FMM as Canopy's blob-aware far field

**Status:** REFERENCE NOTE — not a task, and not Beatnik work. This document
explains **option 4** of the four routes named in
[tasks/add-canopy.md](add-canopy.md) **X1** ("The option set, and which one this
points at"). It carries no exit criterion of its own: the only acceptance test
Beatnik owns is X1's, and the implementation lives in
[tasks/canopy0.md](canopy0.md) **C1**/**F6**, in the Canopy repository.

It is the companion of [tasks/canopy-bbFMM.md](canopy-bbFMM.md), which does the
same job for option 3. The two are worth reading together: they are the two
"black-box" routes, they have the same interface consequences for Beatnik, and
they differ in exactly one place that matters for this kernel — **what their
convergence theory rests on when the kernel stops being a Green's function.**

Everything Beatnik-side — the adapter, the round trip, the measurement harness —
is independent of which route is taken (add-canopy.md **R9**).

**Reference:** Ying, Biros & Zorin, "A kernel-independent adaptive fast
multipole algorithm in two and three dimensions", *JCP* **196** (2004) 591-626.

## The one idea

A classical FMM represents a box's far field by **coefficients in a basis**
(solid harmonics in Canopy, Cartesian monomials in option 2, Chebyshev nodal
values in option 3). The kernel-independent FMM (KIFMM) represents it by an
**equivalent density**: a set of fictitious point sources placed on a surface
around the box, whose field — computed with the *same kernel* — reproduces the
box's field in the far region.

Because the representation is "some sources, evaluated with the kernel", every
operator reduces to (i) evaluating the kernel between two small point sets and
(ii) applying a precomputed pseudo-inverse. Nothing anywhere in the method knows
what the kernel is beyond being able to call it.

Two surfaces per box, per direction:

- **Upward equivalent surface** $y^{B,u}$ — just outside box $B$; carries the
  density $\sigma^{B,u}$ that stands in for $B$'s sources.
- **Upward check surface** $x^{B,u}$ — further out; where the fit is imposed.
  The *check potential* is the true potential of $B$'s sources sampled there.

and their downward mirror images ($x^{A,d}$ a check surface inside box $A$,
$y^{A,d}$ an equivalent surface outside it, carrying the density that stands in
for everything far from $A$).

The single recurring step is the **check-to-equivalent solve**: given a sampled
check potential $u^{\rm chk}$, find the density $\sigma$ on the equivalent
surface with

$$
M\,\sigma = u^{\rm chk}, \qquad M_{ij} = K(x_i^{\rm chk}, y_j^{\rm equiv}),
$$

solved as a regularized least-squares problem with a precomputed pseudo-inverse
of $M$. Both surfaces are the boundary of a $k\times k\times k$ tensor grid on a
cube, so each carries $k^3-(k-2)^3 = 6k^2-12k+8$ points — $k=6\to152$,
$k=8\to296$, $k=10\to488$. That count $k$ is the far-field knob, the counterpart
of `P_ORDER` today and of bbFMM's $n$.

## The five operators

- **P2M (S2U)** — evaluate the box's own sources at the upward check surface,
  then check-to-equivalent solve for $\sigma^{B,u}$. Kernel evaluations plus one
  precomputed pseudo-inverse.
- **M2M (U2U)** — evaluate the eight children's equivalent densities at the
  parent's upward check surface, then solve for the parent's. Note this is
  **not** kernel-independent here in the sense M2M is in option 2: it calls the
  kernel. It is *kernel-blind* — it calls it without knowing it.
- **M2L (V-list)** — evaluate the source box's upward equivalent density at the
  target box's downward check surface. Direct kernel evaluations between two
  point sets on a fixed relative offset, hence precomputable per (level,
  offset). This is the work-dominant operator and the one KIFMM accelerates by
  FFT (below).
- **L2L (D2D)** — evaluate the parent's downward equivalent density at the
  child's downward check surface, then solve for the child's.
- **L2P (D2T)** — evaluate the downward equivalent density at the targets.

Every one is "call the kernel between two point sets, then multiply by a stored
matrix". There is no basis, no addition theorem, no derivative tensor, and no
interpolation rule. That is the whole of the kernel-independence claim, and for
Beatnik's kernel it is literally the three lines P2P already uses
(`canopy/src/Canopy_P2P.hpp:799-803`).

## How it computes the Birkhoff-Rott kernel

Beatnik's far field is

$$
u_i(x) = \sum_s \epsilon_{ilm}\, K_l(x - y_s)\, \omega_s S_{m,s},
\qquad K_l(\delta) = \frac{\delta_l}{(\delta^2 + b)^{3/2}} = -\partial_l\varphi(\delta),
$$

with $\varphi(\delta) = (\delta^2+b)^{-1/2}$ and $b = \varepsilon^2$
([src/Beatnik_BRSolverBase.hpp:29-31](../src/Beatnik_BRSolverBase.hpp#L29-L31)).

### The softening is inert in the algorithm, and that is not the same as being harmless

Mechanically, $b$ enters only where $M_{ij}$ and the M2L blocks are filled, and
it changes nothing structural: the kernel is still translation-invariant (which
is what the FFT acceleration needs), still smooth (more so than $1/r$), still
evaluated pointwise. So `near_softening_factor` becomes unnecessary rather than
load-bearing — X1's discriminator — in the same mechanical sense as option 3.

The difference from option 3 is in *why the method converges*, and it is the
substantive finding of this document. See **The math it is rooted in**: KIFMM's
justification is potential-theoretic and **does** use the fact that the kernel
is a Green's function. Softening breaks that, and what is left is an
approximation-theoretic claim that has to be measured rather than a theorem that
can be cited. Option 3's convergence theory, by contrast, is kernel-blind and
*strengthens* under softening.

### The vector kernel, and the two ways to carry it

Two shapes are available, and they are not equivalent:

1. **Three scalar-$\varphi$ passes.** Fit equivalent densities against the
   scalar $\varphi$, exactly as option 2 does, one pass per strength component
   $\gamma_m = \omega_s S_{m,s}$; then recombine with $\epsilon_{ilm}$. The
   surfaces, the pseudo-inverses and the M2L tables are shared across the three
   — this is precisely Canopy's existing `NComps = 3` fusion
   (`canopy/src/Canopy_Solver.hpp:100`).
2. **The $3\times3$ curl kernel directly.** KIFMM is routinely run on tensor
   kernels (Stokes, Navier), so this is available. It fits vector-valued check
   potentials against vector-valued equivalent densities.

Shape 1 is the one to prefer here, for the same reason option 2's far field is
"three scalar-$\varphi$ passes": the fit is against a scalar kernel whose
relation to $1/r$ is at least understood, and the $\epsilon_{ilm}$ contraction
stays outside the solver where Beatnik's adapter already does it (add-canopy.md
"Approach"). Shape 2 puts the contraction inside and would be an interface
change.

### There is no finite-difference gradient, and here it is free

Canopy today expands the **scalar potential** and gets the gradient by **central
differencing of the L2P evaluation** — six extra `eval_phi` calls per target at
a fixed $h = 10^{-5}w_{\rm self}$
(`canopy/src/Canopy_LaplaceKernel.hpp:851-879`, carrying a
`TODO: replace with analytical derivatives`). That is the third,
$\sim\!10^{-10}$ error plateau add-canopy.md **R1** warns must not be mistaken
for the softening bias (canopy0.md **F7**), and a fixed step was the root cause
of develop-canopy's premature full-roll-up NaN
(`tasks/fmm_premature_nan.md` on that branch).

Under KIFMM the downward local representation **is a set of point sources**, so
its gradient is exact and costs nothing to obtain: differentiate the L2P
evaluation analytically by using $K_l$ in place of $\varphi$ at the same
equivalent-surface points,

$$
u_i(x) = \sum_p \epsilon_{ilm}\, K_l(x - y_p^{A,d})\; \sigma_{m,p}^{A,d},
$$

which is the same kernel call P2P makes. No differentiation of a basis function,
no interpolation derivative, no step size. Option 3 removes the finite
difference by interpolating the vector kernel; option 4 removes it by never
having had a basis to differentiate. Both delete the plateau; this is the
cheaper deletion of the two, and it also removes the $h$-tuning that failed
before.

## Does it need only one solve?

**No — three components are still carried, and that is the physics, not the
kernel representation.** The answer is the same as option 3's
([tasks/canopy-bbFMM.md](canopy-bbFMM.md) "Does it need only one solve?") and
for the same reason: $u$ is the curl of the softened vector potential
$A_i = \sum_s \omega_s S_{i,s}\,\varphi$, three intrinsically independent scalar
fields, and no change of kernel representation can reduce that count.

What KIFMM does well here is that the three passes share *more* than they do in
any of the other routes:

```
σ^{B,u}_(m)  : 3 densities on ONE upward surface, ONE pseudo-inverse
M2L          : 3 matvecs against the SAME per-(level,offset) block
σ^{A,d}_(m)  : 3 densities on ONE downward surface, ONE pseudo-inverse
u_i(x)       = ε_{ilm} Σ_p K_l(x − y_p^{A,d}) σ^{A,d}_{m,p}     ← one pass, exact gradient
```

One traversal, three components inside, one operator table — i.e. exactly the
shape Canopy's `solve()` already has (the gradient output view is
`(num_particles, NComps, 3)`, `canopy/src/Canopy_DownwardSweep.hpp:120-126`).

So from Beatnik's side **the adapter and the two `solve()` calls of
add-canopy.md "Approach" are unchanged**, which is what X1 already promises
("Beatnik requires no interface change"). The Riesz-scalar path (**T7**) remains
a second `solve()` with different charges over the same tree, unaffected.

| | Canopy today | option 4 (KIFMM) | option 3 (bbFMM) |
| --- | --- | --- | --- |
| far-field kernel | bare $1/r$ | the actual softened kernel | the actual softened kernel |
| far-field DOF per box | $(p{+}1)^2$ harmonics | $6k^2{-}12k{+}8$ surface points | $n^3$ volume nodes |
| M2L | addition theorem | precomputed dense block, FFT-accelerated | precomputed dense block, SVD-compressed |
| kernel-touching operators | 3 (M2M, M2L, L2L) | **all five** — but kernel-blind | 1 (M2L) |
| L2P | value + 6 FD evals | ×1, exact analytic gradient | ×1, returns velocity |
| new derivation required | — | **none** | none |
| convergence theory under softening | **invalid** (needs harmonicity) | **weakened** (see below) | **unaffected** |

## The math it is rooted in

### Potential theory, and exterior uniqueness

KIFMM's equivalent density is not an approximation dressed up as a
representation — for a Green's-function kernel it is a theorem. For $1/r$:

1. The field of box $B$'s sources is **harmonic** in the exterior of $B$ and
   decays at infinity.
2. By the solvability of the exterior Dirichlet problem, there exists a
   single-layer density on any surface enclosing $B$ whose field equals it
   throughout the exterior.
3. By **uniqueness** of that exterior problem, matching the field on the *one*
   check surface $x^{B,u}$ — a finite, sampled, least-squares match — certifies
   the match on the whole far region.

Step 3 is the load-bearing one and it is the reason the method is a method and
not a fit: a residual measured on one surface bounds the error everywhere
outside it. Steps 1-3 are all statements about solutions of $\nabla^2 u = 0$.

### Where softening breaks it, exactly

$\varphi = (r^2+b)^{-1/2}$ is not harmonic:
$\nabla^2\varphi = -3b\,(r^2+b)^{-5/2} \ne 0$ **everywhere**, not merely at the
source. There is no local elliptic PDE whose Green's function it is, hence no
exterior Dirichlet problem, hence neither the existence argument of step 2 nor —
and this is what costs — the uniqueness argument of step 3.

The right way to see the size of the damage uses the fact that $\varphi$ is the
Newtonian potential of a **Plummer cloud** of unit mass and density
$\rho_b(r) = \tfrac{3b}{4\pi}(r^2+b)^{-5/2}$. Therefore, with $G = 1/r$,

$$
\varphi = G * \rho_b, \qquad\text{so}\qquad u_b = \rho_b * u_{\rm bare},
$$

i.e. **the softened field is the mollification of the bare field** by a
radially symmetric unit-mass kernel. Now apply the mean value property: for a
point $x$ at distance $d$ from the nearest true source, every shell of $\rho_b$
of radius $r < d$ reproduces $u_{\rm bare}(x)$ exactly, and only the mass
outside radius $d$ contributes an error. The Plummer mass fraction beyond $r$ is

$$
1 - \frac{r^3}{(r^2+b)^{3/2}} \;\approx\; \frac{3b}{2r^2} \quad (r \gg \sqrt b),
$$

which recovers the $\tfrac32\varepsilon^2/r^2$ relative discrepancy quoted in
add-canopy.md **Problem** and in `canopy-questions.md` — from the geometry
rather than from a series truncation. Two consequences:

- **Non-harmonicity is not a small correction at leaf scale.** The mollifier's
  own width is $\sqrt b = \varepsilon = 0.025$, while a leaf cell on the
  milestone-0 bubble is of comparable size (root box $\sim\!0.6$ across, so
  $\sim\!0.075$ at depth 3, $\sim\!0.037$ at depth 4). Each source's cloud spans
  a leaf. Deep in the tree, "approximately harmonic outside the box" is not
  approximately true.
- **A surface density of softened sources spans a different set of functions
  than a surface density of bare ones.** Writing the equivalent field as
  $\rho_b * v$ with $v$ the harmonic single layer, the naive construction —
  choose $v = u_{\rm bare}$ outside the box — leaves a mismatch fed by exactly
  the mollifier mass that reaches back into the box interior, again
  $O(\tfrac32 b/d^2)$. That is the **same order as the plateau this whole route
  exists to remove.**

The last point is a bound on one particular construction, not on the *best*
density, and the surfaces carry many more degrees of freedom than that
construction uses — $152$ at $k=6$ — so the achievable residual may be far
smaller. But it cannot be settled by citing the KIFMM convergence theory,
because that theory is what softening invalidated. **It has to be measured, and
the measurement is not the check-surface residual** — the check-surface residual
is exactly the quantity that no longer certifies anything. The honest diagnostic
is an *off-surface* probe: fit the density, then compare against a brute-force
softened sum at targets scattered through the admissible far region, at several
values of $b/(\text{box width})^2$ and several $k$.

### What survives intact

- **Translation invariance**, hence the per-(level, offset) precomputation and
  the FFT acceleration of M2L: embed the upward equivalent density in a regular
  grid, convolve with the kernel, take the target values off the downward grid.
  $b$ is inert for this — it changes the numbers in the kernel grid and nothing
  about the convolution structure. This is KIFMM's main constant-factor win and
  it is not at risk.
- **Low numerical rank of the far-field block.** The underlying reason any of
  the black-box routes work is that the interaction matrix between
  well-separated boxes is numerically low-rank, and smoother kernels have
  *lower* rank than $1/r$, not higher. This is the $\mathcal{H}^2$-matrix
  argument option 3 rests on entirely, and it is what one would fall back on to
  justify option 4 without potential theory — at the cost of losing the
  one-surface-certifies-everywhere property, which is exactly what was lost.
- **Adaptivity.** KIFMM's U/V/W/X list machinery handles level-mismatched
  leaf-leaf interactions on non-uniform trees, which a surface-based bubble mesh
  produces in quantity. Canopy's dual-tree traversal and communication plan
  already resolve level-mismatched pairs by MAC
  (`canopy/src/Canopy_CommunicationPlan.hpp:338-361`), so this is an integration
  question — whether the existing plan can be asked for KIFMM's four lists — not
  a new algorithm. It is unmeasured and belongs in canopy0.md.

### What it is *not* rooted in

The solid-harmonic addition theorems (Greengard Thms 5.22, 5.23, 5.26, cited at
`canopy/src/Canopy_LaplaceKernel.hpp:273`, `:373`, `:688`) — which is the point
of the exercise, since needing harmonicity is precisely why Canopy's present far
field cannot carry the blob (add-canopy.md X1, "Why it cannot be a patch to
Canopy's existing M2L"). Note the asymmetry carefully, because it is easy to
overstate the win: KIFMM does not need harmonicity **in its operators**, but it
does use it **in its error argument**. Option 3 needs it in neither.

## The three costs X1 does not state

X1 records option 4 as "also black-box in the kernel; more to stand up than
bbFMM, but very robust". All three parts of that are true. The following make it
concrete.

### 1. Scale invariance dies, and KIFMM leans on it harder than bbFMM does

For a homogeneous kernel, $K(\alpha r) = \alpha^m K(r)$, KIFMM precomputes **one**
check-to-equivalent pseudo-inverse and **one** M2M/L2L set for the entire tree
and rescales per level. $(r^2+b)^{-3/2}$ has no such scaling, because $\sqrt b$
is an absolute length. This is the same fact as add-canopy.md's observation that
softening breaks the operator cache's documented "no physical width enters the
builder" property (`canopy/src/Canopy_DownwardSweep.hpp:1075`), and canopy0.md
**F6(c)**.

Consequence: per-level pseudo-inverses (small — a few hundred squared, times
`max_depth`) *and* per-level M2L tables (roughly 316 realized offsets × the
kernel grid × depth). As with option 3, the mitigating point matters for
add-canopy.md **R5** and **T8**: all of it depends only on (level, offset, $b$)
and **not** on particle positions, so it is built once at solver construction
and survives every `migrate` / `rebalance` / `auto_maintain`. It is a
precomputation cost, not a per-evaluation cost.

It is a bigger relative hit here than for option 3 because more of KIFMM's
per-level state was previously shared: bbFMM loses one table per level, KIFMM
loses the table *and* every pseudo-inverse.

### 2. The check-to-equivalent inversion is ill-conditioned, and that caps the accuracy

$M_{ij} = K(x_i^{\rm chk}, y_j^{\rm equiv})$ between two nested surfaces is
exponentially ill-conditioned in $k$ — it is a discretized compact operator, and
its singular values decay geometrically. KIFMM handles this with a regularized
pseudo-inverse (Tikhonov or truncated SVD), and the regularization parameter
places a **floor** on the achievable representation error. Published KIFMM
results sit in the $10^{-6}$-$10^{-9}$ band; reaching $10^{-10}$ requires
careful regularization and, in practice, extended precision in the
precomputation.

This bears directly on X1's exit criterion, which asks for
$\tau_A \le 10^{-10}$ with `near_softening_factor = 0`. Option 3's interpolation
step is by contrast **well-conditioned** — Chebyshev interpolation has no
inverse problem in it at all, and its SVD compression is a speed optimization
that can simply be turned off to buy accuracy. **For a $10^{-10}$ target,
option 3 has the better conditioning story and option 4 the worse one**, which
inverts the usual ordering (KIFMM is normally the more robust of the two, but
"robust" there means across kernels and geometries, not across tolerances).

### 3. Fewer degrees of freedom per box, and that is the real argument in its favor

A surface beats a volume on count. For comparable accuracy:

| far-field DOF per box | option 4, $6k^2{-}12k{+}8$ | option 3, $n^3$ |
| --- | --- | --- |
| $\sim\!60$ | $k=4 \to 56$ | $n=4 \to 64$ |
| $\sim\!200$ | $k=6 \to 152$ | $n=6 \to 216$ |
| $\sim\!400$-$500$ | $k=8 \to 296$ | $n=8 \to 512$ |
| high | $k=10 \to 488$, $k=12 \to 728$ | $n=10 \to 1000$, $n=12 \to 1728$ |

M2L is quadratic in the DOF count before acceleration, so the surface
representation is genuinely cheaper per unit accuracy — and this is before
KIFMM's FFT, which is the stronger of the two accelerations. This is the
performance case for option 4 and it is real. It has to be set against cost 2,
which is where the accuracy actually gets decided for Beatnik's target.

## What this changes about X1's recommendation

Nothing, and it is not this document's place to. X1's recommendation stands as
written: option 1 to unblock, then option 2 or 3 as the permanent fix depending
on how much is hand-derived versus interpolated. **Option 4 is not promoted by
this reading**, and three facts are added to the choice:

- **Option 4's convergence theory does not survive softening; option 3's does.**
  This is the one asymmetry that is not a matter of taste. KIFMM's
  equivalent-density representation is exact for a Green's function and its
  one-surface-residual-certifies-the-far-field property is what makes it
  trustworthy; both statements are consequences of harmonicity and both are lost
  at $b > 0$. What remains is a low-rank approximation claim indistinguishable in
  kind from option 3's, but without option 3's kernel-blind convergence estimate.
  A route whose error argument has to be replaced by measurement is a worse
  answer to X1's question than one whose error argument is unaffected.
- **Option 4's conditioning works against X1's $10^{-10}$ exit criterion**, where
  option 3's does not (cost 2 above). Both routes remove the kernel bias and
  therefore both clear X1's discriminator — `near_softening_factor = 0` — but
  clearing the discriminator and landing at $10^{-10}$ are different claims, and
  X1's closing paragraph already says so.
- **Option 4 removes the finite-difference L2P gradient the most cheaply of any
  route** (canopy0.md **F7**), because a point-source local representation
  differentiates by calling a different kernel at the same points. This is a
  genuine advantage and it is the strongest thing in option 4's favor besides
  the DOF count.

If option 4 is nonetheless taken, the first measurement to make is the one named
under **Where softening breaks it, exactly**: an *off-surface* probe of the
equivalent-density residual at several $b/(\text{box width})^2$ and several $k$,
before any of the five operators is written. It is cheap, it needs no tree, and
it is the check that distinguishes "the representation spans the softened far
field" from "the fit was good on the surface where we looked".

Beatnik does not care which route is taken. It requires whatever $\tau_A$ claim
A asserts, and add-canopy.md **T5** is what states that number.
