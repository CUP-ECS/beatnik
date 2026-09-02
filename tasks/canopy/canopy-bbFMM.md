# Black-box FMM as Canopy's blob-aware far field

**Status:** REFERENCE NOTE — not a task, and not Beatnik work. This document
explains **option 3** of the four routes named in
[tasks/add-canopy.md](add-canopy.md) **X1** ("The option set, and which one this
points at"). It carries no exit criterion of its own: the only acceptance test
Beatnik owns is X1's, and the implementation lives in
[tasks/canopy0.md](canopy0.md) **C1**/**F6**, in the Canopy repository.

It exists so that a session evaluating the routes does not have to re-derive how
black-box FMM works, what it costs, or how its component structure lines up with
Canopy's existing `NComps = 3` gradient path. Everything Beatnik-side —
the adapter, the round trip, the measurement harness — is independent of which
route is taken (add-canopy.md **R9**).

**Reference:** Fong & Darve, "The black-box fast multipole method", *JCP*
**228** (2009) 8712-8725.

## The one idea

Black-box FMM (bbFMM) replaces *analytic expansion of the kernel* with
*polynomial interpolation of the kernel*. For a well-separated box pair (source
box $B$, target box $A$) it writes

$$
K(x, y) \;\approx\; \sum_p \sum_q R(x, \bar{x}_p)\; K(\bar{x}_p, \bar{y}_q)\; R(y, \bar{y}_q),
$$

where $\bar{x}_p$ are the $n^3$ tensor-product Chebyshev nodes of box $A$,
$\bar{y}_q$ those of box $B$, and $R$ is the Chebyshev interpolation function

$$
R(x, \bar{x}) = \frac{1}{n} + \frac{2}{n}\sum_{k=1}^{n-1} T_k(x)\,T_k(\bar{x})
$$

per dimension, tensored across the three.

That is a **separable** approximation: $x$ and $y$ no longer appear together.
Separability is the entire content of an FMM — it is what lets you sum over
sources once per box instead of once per target — and here it is obtained by
interpolation rather than by a series identity.

The kernel appears in exactly one place: the $n^3\times n^3$ matrix of numbers
$K(\bar{x}_p, \bar{y}_q)$. You get those by **calling the kernel**. Nothing else
in the method knows what kernel it is.

## The five operators

- **P2M** — anterpolate charges onto the source box's Chebyshev nodes:
  $W_q = \sum_{s\in B} R(y_s, \bar{y}_q)\,\gamma_s$. Kernel-independent.
- **M2M** — interpolate a child's $n^3$ nodal values onto the parent's $n^3$
  nodes. A fixed $n^3\times n^3$ matrix per child octant. Kernel-independent.
- **M2L** — $g_p = \sum_q K(\bar{x}_p, \bar{y}_q)\,W_q$. The **only**
  kernel-touching operator, and it is a dense matvec against *evaluated* kernel
  values. No derivatives, no addition theorem, no basis.
- **L2L** — Chebyshev interpolation of parent nodal values down to child nodes.
  Kernel-independent.
- **L2P** — $u(x_i) = \sum_p R(x_i, \bar{x}_p)\,g_p$. Kernel-independent.

Four of the five are pure interpolation algebra. Fong & Darve then
SVD-compress the M2L matrices (they are numerically low-rank) to cut the
constant.

## How it computes the Birkhoff-Rott kernel

Beatnik's far field is

$$
u_i(x) = \sum_s \epsilon_{ilm}\, K_l(x - y_s)\, \omega_s S_{m,s},
\qquad K_l(\delta) = \frac{\delta_l}{(\delta^2 + b)^{3/2}} .
$$

Three properties of bbFMM matter here, and they are exactly the three things
option 2 (Cartesian-Taylor, add-canopy.md X1 "The recipe") has to work for.

### Softening is free, and free for the right reason

The method requires only that the kernel be *smooth on the box pair* — that is
what makes Chebyshev interpolation converge. It never requires
$\nabla^2 K = 0$, so the impossibility argument in X1 ("the only isotropic
harmonic functions in 3D are $\mathrm{const}$ and $1/r$, so no finite
solid-harmonic expansion represents the blob correction") simply does not arise.
$b$ sits inertly inside $w = \delta^2 + b$ at the point where
$K(\bar{x}_p, \bar{y}_q)$ is evaluated, and with $b > 0$ the kernel is *more*
analytic than $1/r$, so the interpolation converges at least as fast as it does
for the bare kernel.

`near_softening_factor` therefore becomes unnecessary rather than load-bearing,
which is X1's discriminator.

### The vector kernel goes in directly

No $\varphi$, no $\nabla\varphi$, no derivative ladder. Where option 2 must
derive $\partial_a P_m = -(2m+1)r_a P_{m+1}$ and the multi-index recurrence for
$b_{k+e_i}$, bbFMM evaluates $K_l(\bar{x}_p, \bar{y}_q)$ with the same three
lines of C++ that P2P already uses (`canopy/src/Canopy_P2P.hpp:799-803`). That
is what X1's "least new derivation" means, and it is accurate.

### There is no gradient step at all

This is the part X1 does not record, and it is worth having.

Canopy today expands the **scalar potential** and obtains the gradient by
**central finite differencing of the L2P evaluation** — six extra `eval_phi`
calls per target at $h = 10^{-5} w_{\rm self}$
(`canopy/src/Canopy_LaplaceKernel.hpp:851-879`, which carries a
`TODO: replace with analytical derivatives`). That finite difference is the
source of the third, $\sim\!10^{-10}$ plateau add-canopy.md **R1** warns must
not be mistaken for the softening bias (canopy0.md F7), and a fixed step was the
root cause of develop-canopy's premature full-roll-up NaN
(`tasks/fmm_premature_nan.md` on that branch).

Under bbFMM the interpolated quantity *is* the vector kernel, so L2P returns the
velocity components directly. The finite difference disappears, and that plateau
disappears with it.

## Does it need only one solve?

**No — three components are still carried, and that is the physics, not the
kernel representation.** But the accounting shifts favorably.

The velocity has three independent components per target and depends linearly on
a three-component source strength, so any far field must move a $3\to3$ map.
Concretely, per box pair:

```
W_q^(m)   = Σ_s R(y_s, ȳ_q) ω_s S_{m,s}       3 moment sets  (= NComps=3 today)
G_p^(l,m) = Σ_q K_l(x̄_p, ȳ_q) W_q^(m)         9 matvecs, 3 distinct kernel matrices
L_p^(i)   = ε_{ilm} G_p^(l,m)                  contract ε here → 3 local sets
u_i(x)    = Σ_p R(x, x̄_p) L_p^(i)             L2P once
```

Compare what Canopy does now. `NComps` is documented as "number of simultaneous
solves (charge components)" (`canopy/src/Canopy_Solver.hpp:100`), and the
multipole and local coefficient arrays are dimensioned per component — the
gradient output view is `(num_particles, NComps, 3)`
(`canopy/src/Canopy_DownwardSweep.hpp:120-126`). So the existing path is
**three scalar solves fused into one traversal**, sharing the tree, the
communication plan and the M2L operator table. One `solve()` call, three
components inside.

bbFMM has the same shape at that level — one traversal, three components — so
from Beatnik's side **the adapter and the two `solve()` calls of add-canopy.md
"Approach" are unchanged**, which is what X1 already promises ("Beatnik requires
no interface change").

What changes:

| | Canopy today | bbFMM |
| --- | --- | --- |
| M2L work | 1 operator × 3 comps | 3 operators × 3 comps, contracted to 3 |
| L2P work | ×7 (value + 6 FD evals) | ×1, returns velocity |
| $\epsilon_{ilm}$ contraction | after the solve, in Beatnik's adapter | inside M2L, before L2L |
| far-field kernel | bare $1/r$ | the actual softened kernel |

M2L gets $\sim3\times$ heavier, L2P gets $\sim7\times$ lighter, one error
plateau is deleted and one kernel bias is deleted.

Moving the $\epsilon_{ilm}$ contraction inside M2L is a small bonus — only three
local expansions descend the tree rather than nine — but it does not reduce the
component count, and nothing can: $u$ is the curl of the softened vector
potential $A_i = \sum_s \omega_s S_{i,s}\,\varphi$, which is intrinsically three
scalar fields. That is the same bilinearity add-canopy.md "Approach" already
relies on to do the cross product as local post-processing, and the same reason
option 2's far field is "three scalar-$\varphi$ passes".

Note that if the contraction is done inside M2L, Beatnik's adapter no longer
does it — which would be an interface change after all. Keeping the adapter's
contraction and having Canopy return the same $3\times3$ tensor is the
interface-preserving choice, at the cost of nine local expansions instead of
three. **This is a Canopy-side decision, not a Beatnik one**, and either way the
$\epsilon$ sign convention must be stated at the boundary (add-canopy.md
Conventions, "Comments").

## The math it is rooted in

- **Lagrange/Chebyshev interpolation and Bernstein-ellipse convergence.** For a
  function analytic on a neighborhood of the interpolation box, Chebyshev
  interpolation error falls geometrically, like $\rho^{-n}$ in the per-dimension
  node count $n$, with $\rho$ set by how far the nearest singularity sits from
  the box in the mapped complex plane. For a well-separated box pair the
  kernel's singularity at $\delta = 0$ lies outside the pair region, so
  $\rho > 1$ is guaranteed by admissibility; softening moves the singularity off
  the real axis entirely and can only increase $\rho$. **This replaces the
  multipole truncation estimate**: $n$ is the far-field knob, as X1 states.
- **Low-rank separable approximation of the kernel matrix.** bbFMM is a
  degenerate-kernel / pseudoskeleton method: the interaction block between two
  well-separated boxes is numerically low-rank, and Chebyshev nodes are a cheap,
  kernel-blind way of choosing the row/column skeleton. The SVD stage makes that
  explicit. This is the theory $\mathcal{H}^2$-matrices rest on, and it is why
  the method needs no relationship between the kernel and Laplace's equation.
- **What it is *not* rooted in:** the solid-harmonic addition theorems
  (Greengard Thms 5.22, 5.23, 5.26, cited at
  `canopy/src/Canopy_LaplaceKernel.hpp:273`, `:373`, `:688`), which need
  harmonicity and are precisely why Canopy's present far field cannot carry the
  blob; and multivariate Taylor expansion, which needs only smoothness (hence
  option 2 works) but needs its derivative tensors hand-derived.

## The two costs X1 does not state

Both are worth knowing before this route is chosen.

### 1. The softened kernel is not homogeneous, so the level-independent M2L cache dies

Fong & Darve get one M2L table per relative box offset for the **whole tree** by
exploiting homogeneity, $K(\alpha r) = \alpha^m K(r)$, and rescaling per level.
$(r^2+b)^{-3/2}$ has no such scaling — $\sqrt{b}$ is an absolute length. This is
the same fact as add-canopy.md's observation that softening breaks the operator
cache's documented "no physical width enters the builder" property
(`canopy/src/Canopy_DownwardSweep.hpp:1075`).

A table is therefore needed **per level**: roughly 316 realized offsets
$\times\ n^3 \times n^3 \times 3$ components $\times$ depth.

The mitigating point, which matters for add-canopy.md **R5** and **T8**: that
table depends only on (level, relative offset, $b$) and **not** on particle
positions, so it is built once at solver construction and survives every
`migrate` / `rebalance` / `auto_maintain`. It is a precomputation cost, not a
per-evaluation cost.

### 2. $n^3$ nodes is a large constant, and $10^{-10}$ is expensive

Node counts: $n = 4 \to 64$, $n = 6 \to 216$, $n = 8 \to 512$,
$n = 10 \to 1000$, $n = 12 \to 1728$. Geometric convergence at standard
admissibility puts $n \approx 4$–$5$ near $10^{-4}$ and $n \approx 8$ near
$10^{-8}$, so X1's exit criterion of $\tau_A \le 10^{-10}$ plausibly wants
$n \approx 10$–$12$ before SVD compression. SVD compression is what makes those
orders tractable and is not optional there.

This is the same open question X1 records under "The open question that only
measurement settles", with one clarification: for bbFMM the residual is a
**truncation** plateau and therefore tunable — unlike the kernel bias, which is
not — but the knob is a cubed one. canopy0.md **F6(d)** puts $10^{-6}$ out of
reach for a low-order *Cartesian-Taylor* basis; bbFMM's $n$ can be pushed
further than a hand-derived Taylor order can, at a cost that grows as $n^6$ in
the M2L matvec and $n^3$ in storage.

## What this changes about X1's recommendation

Nothing, and it is not this document's place to. X1's recommendation stands as
written: option 1 to unblock, then option 2 or 3 as the permanent fix depending
on how much is hand-derived versus interpolated. Two facts are added to that
choice:

- Option 3 removes the finite-difference L2P gradient, and with it a **second,
  independent** error floor (canopy0.md F7) and the mechanism of a past
  whole-field NaN. Option 2 does not — a Cartesian-Taylor local expansion
  differentiates analytically, so it also removes it, but option 3 removes it
  without differentiating anything at all.
- Option 3 trades the derivation away for a per-level operator precomputation
  and an $n^3$ constant. Option 2 keeps the level-independent structure and pays
  in derived code, of which the reference treecode has already written the
  order-2 part.

Beatnik does not care which is taken. It requires whatever $\tau_A$ claim A
asserts, and add-canopy.md **T5** is what states that number.
