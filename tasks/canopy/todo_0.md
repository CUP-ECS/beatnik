# Can one abstraction hold three far fields in Canopy?

**Status:** DECISION NOTE — answers one question, read-only against
`../canopy` at the working-tree state of 2026-09-02. Nothing was built, run or
modified. Every claim about Canopy carries a `file:line`; inferences are
labelled as such.

**The question.** Can Canopy support a Cartesian-Taylor far field (option 2), a
black-box FMM far field (option 3) and its existing solid-harmonic $1/r$ far
field behind a single abstraction, selected by a compile-time template parameter
on the solver?

**Companion documents:** [canopy-kernel-rec.md](canopy-kernel-rec.md),
[canopy-bbFMM.md](canopy-bbFMM.md), [canopy-kIndp.md](canopy-kIndp.md),
[canopy0.md](canopy0.md) **F1**/**F6**/**F7**/**F8** and **C1**/**C7**/**C11**,
[add-canopy.md](add-canopy.md) **X1**.

---

## 1. The verdict

**Feasible, with four named conditions — and criterion 4 fails as the repository
stands today.**

Against the four criteria, one at a time:

| Criterion | Verdict | Why |
| --- | --- | --- |
| **1 — no kernel enumerated in shared code** | **Passes with named work.** Twelve sites need changing; eleven are removed by a trait, and three of those are dead code that should simply be deleted. The twelfth — the fused M2L inner loop — is not removable by a trait and must be *moved into* the kernel. | [§3.1](#31-criterion-1--every-site-that-would-need-a-branch) |
| **2 — option 2 costs no more under the abstraction** | **Passes, narrowly.** The generalization splits roughly 80/20: nearly everything is work option 2 needs anyway. The 20% paid early for option 3 is **interface declaration, not implementation** — a wider M2L contract that option 2 fills with no-ops. Sized at ~250-350 lines of shared-code change plus one new test. | [§4](#4-what-the-abstraction-costs) |
| **3 — adding option 3 later touches only new code plus traits** | **Passes, conditionally on C-1 and C-3.** With those, a later option-3 session opens one new header, one line of `src/CMakeLists.txt`, and nothing else. Without C-1 it reopens `Canopy_DownwardSweep.hpp`; without C-3 it reopens it *and* `Canopy_Solver.hpp`. | [§3.3](#33-criterion-3--what-a-later-option-3-session-would-open) |
| **4 — existing Laplace path bit-for-bit identical** | **FAILS today.** Not because the abstraction breaks it, but because **no test in Canopy could detect a break.** The tightest full-pipeline assertion is a $5\times10^{-2}$ relative bound (`tests/tstMultiSolve.hpp:929-930`). Six named mechanisms could silently perturb the arithmetic. | [§3.4](#34-criterion-4--bit-for-bit-identity-and-why-it-is-currently-unverifiable) |

Criterion 4's failure is **a cost, not a fatality**, and its fix is the cheapest
item in this whole document: a golden-vector test that must land *before* any
refactor. But it is a genuine failure and it changes the sequencing — the
abstraction is not safe to start until it exists.

### The four conditions

**C-1 — M2L becomes a three-stage, kernel-owned operation.** The abstraction
must declare M2L as (i) an optional per-source-cell pre-pass, (ii) a per-pair
core apply, (iii) an optional per-target-cell post-pass — with the operator set
**opaque to the sweeps**, reached only through an integer `op_idx`. The sweeps
keep the traversal, the CSR, the team-per-target launch and the write-back. This
is the load-bearing condition and [§3.2](#32-the-m2l-operator-table-the-most-likely-single-point-of-failure)
is entirely about it. The harmonic path and option 2 supply no-op pre/post
passes; option 3's compressed form and option 4's FFT form both *require* them.

**C-2 — the operator-table cap becomes a byte budget, not a key count.**
`M2L_OP_COUNT_CAP = 32768` (`src/Canopy_DownwardSweep.hpp:304`) is a count. At
today's 58 KB per key that permits 1.9 GB; at option 3's $n=6$ (1.1 MB per key)
it permits 36 TB. Each route must declare bytes-per-key and each must have a
defined overflow path.

**C-3 — operator construction splits into a persistent geometry cache and a
per-tree index map.** Today the whole table is rebuilt whenever the interaction
list is dirty (`:645-646`, `:1079-1109`), which every topology change triggers
(`src/Canopy_Solver.hpp:565`, `:610`; `src/Canopy_DownwardSweep.hpp:637`). That
is affordable for the harmonic and Cartesian tables and unaffordable for an SVD.

**C-4 — a bit-for-bit golden test lands first.** See
[§5](#5-failure-modes-and-the-cheapest-falsifying-checks).

### The one thing that could make all of this moot

The abstraction's feasibility and **option 3's own viability** are separate
questions, and the second is in worse shape than
[canopy-kernel-rec.md](canopy-kernel-rec.md) assumes. That document's worked
memory example uses $N_{\rm keys} = 2000$. Canopy's own tuning comment states
the realized distinct-key count is **"globally ~16 k under MAC=0.5"**
(`src/Canopy_DownwardSweep.hpp:37-41`) — 8× larger, and 50× the textbook 316.
At 16 k keys, option 3 at $n=6$ uncompressed is ~17.6 TB per rank before the
softening key even carries a level. That is a fact about option 3, not about the
abstraction, but it belongs in the first section because it is the number most
likely to make "build the abstraction to hold option 3" the wrong call. See
[§5](#5-failure-modes-and-the-cheapest-falsifying-checks) for the threshold.

### If C-1 is rejected

A **narrower abstraction is still worth having, and option 2 needs all of it
anyway**: generalize the coefficient type, the coefficient count, the MPI
packing and the auxiliary-table ownership; leave the M2L *apply* alone. That
buys options 2 and 4-unaccelerated for free, keeps the harmonic apply
bit-identical by construction, and leaves option 3 to write a second M2L driver
later. It is strictly less than the full answer and strictly more than nothing.
[§4.3](#43-the-fallback-if-c-1-is-rejected) sizes it.

---

## 2. The factoring: two axes, one solver parameter

### The hypothesis fails as stated

"One template parameter, a kernel type, of which there are three" is **not** what
the code supports, and it is not what the three routes are. Canopy's
`LaplaceKernel` bundles two independent things into one struct:

| Concern | Where it lives in `LaplaceKernel` today |
| --- | --- |
| **Basis** — storage type, coefficient count, index map, symmetry, the five operators' algebra, five width normalizations | `:154`, `:157`, `:159`, `:177-193`, `:226-229`, `:267-272`, `:369-372`, `:685-687`, `:795-799` |
| **Kernel** — $1/r$, entering as harmonicity (the addition theorems, cited at `:273`, `:373`, `:688`) and as $\rho^{-(n+j+1)}$ in the operator builder (`:600`) | `:273`, `:373`, `:600`, `:688` |

The two are not in bijection, and the code shows why. Option 3's basis is
kernel-blind — the kernel enters only as $n^3\times n^3$ evaluated numbers,
which is the same three lines P2P already runs (`src/Canopy_P2P.hpp:883-896`).
Option 2's basis needs a kernel-specific derivative ladder. The present basis is
welded to harmonicity and cannot carry any other kernel at all
([canopy0.md](canopy0.md) **F6(a)**, confirmed by the theorem citations above).

### What the code actually supports

**Two axes, composed into one solver template parameter.** The sweeps already
take exactly one type — `UpwardSweep<MemorySpace, ExecutionSpace, KernelType>`
(`src/Canopy_UpwardSweep.hpp:55`), likewise `DownwardSweep`
(`src/Canopy_DownwardSweep.hpp:100`) and `P2P` — and read everything else off it
as traits. That slot is the right shape; what goes *in* it should be
`Basis<Scalar, Order, NComps>` where the basis names its kernel internally, so
that "option 3 generalizes to other kernels" is expressible as
`ChebyshevBasis<SoftPlummer, …>` versus `ChebyshevBasis<Stokes, …>` without a
new basis.

Validity is enforced by **what the basis requires of its kernel**, checked at
instantiation:

| Basis | Requires of its kernel | Valid kernels |
| --- | --- | --- |
| Solid-harmonic | Harmonicity — supplied as a tag, not a callable | bare $1/r$ only |
| Cartesian-Taylor | `deriv_tensor(order, r) → coeffs` (a ladder) | any kernel with a derived ladder: $1/r$, $(r^2{+}b)^{-1/2}$ |
| Chebyshev / bbFMM | `evaluate(x, y) → Scalar[out_comps]` | anything callable |

So the answer to "one parameter, two parameters with a traits layer, or one
parameter carrying traits the sweeps query" is: **one parameter carrying traits
the sweeps query, where that parameter is itself a two-parameter composition.**
The sweeps never see the kernel axis. That is both what the code is shaped for
and what makes option 3's generality claim real.

### The solver's parameter list

`Solver` currently reads `<MemorySpace, ExecutionSpace, Scalar, P_ORDER,
NComps>` (`src/Canopy_Solver.hpp:104-105`) and fixes the kernel as a typedef at
`:112`. The backward-compatible form is a **template template parameter with a
default**:

| Parameter | Today | Proposed | Note |
| --- | --- | --- | --- |
| `MemorySpace`, `ExecutionSpace` | unchanged | unchanged | |
| `Scalar` | `double` | `double` | `float` is a live path (`src/Canopy_DownwardSweep.hpp:301-302`; `tests/tstMultiSolve.hpp:1079`). **Assumption:** the two new bases are `double`-only and must say so, per [canopy-kernel-rec.md](canopy-kernel-rec.md)'s closing note. |
| `P_ORDER` | `int = 8` | `int = 8`, forwarded opaquely | Semantically it becomes "the basis's order knob": $P$ for harmonic, $p$ for Taylor, $n$ for Chebyshev. Different quantities, same slot. Must be documented, and **C7's runtime factory must enumerate per basis**, not per order alone. |
| — | — | `template <class, int, int> class FarField = LaplaceKernel` | New, defaulted. `Solver<MS,ES,double,8,3>` keeps compiling verbatim, so add-canopy **X1**'s "Beatnik requires no interface change" survives at the type level too. |
| `NComps` | `int = 1` | `int = 1` | Stays the *charge*-component count. See the `sets_per_component` trait below. |

This is the one place a defaulted template-template parameter earns its
awkwardness: it makes the change invisible to `createSolver`
(`src/Canopy_Solver.hpp:719-727`), to all ten test headers and to Beatnik's
adapter.

### The trait and operator contract a basis must supply

Existing traits are marked *(present)*; the rest are new. "Kernel-dep." means
the trait's *value* depends on the kernel axis, not just the basis.

| Name | Signature sketch | Harmonic | Option 2 (Taylor $p$) | Option 3 (Chebyshev $n$) | Kernel-dep. |
| --- | --- | --- | --- | --- | --- |
| `scalar_type` *(present, `:153`)* | typedef | `Scalar` | `Scalar` | `Scalar` | no |
| `coeff_type` | typedef (replaces `complex_type`, `:154`) | `Kokkos::complex<Scalar>` | `Scalar` | `Scalar` | no |
| `scalars_per_coeff` | `constexpr int` | 2 | 1 | 1 | no |
| `num_coeffs_per_cell` *(present, `:157`)* | `constexpr int` | $(P{+}1)(P{+}2)/2$ | $\binom{p+3}{3}$ | $n^3$ | no |
| `m2l_num_src_coeffs` *(present, `:488`)* | `constexpr int` | $(P{+}1)^2$ | $\binom{p+3}{3}$ | $n^3$ | no |
| `num_components` *(present, `:158`)* | `constexpr int` | `NComps` | `NComps` | `NComps` | no |
| `sets_per_component` | `constexpr int` | 1 | 1 | 9 (or 12) | no |
| `m2l_key_dd_max` | `constexpr int` | 6 / 4 for `float` | route-specific | route-specific | yes |
| `key_needs_level` | `constexpr bool` | `false` | `true` | `true` | **yes** |
| `canonicalize_key` | `(key) → key` | zeroes `max_d` | identity | identity | yes |
| `aux_tables_type` | typedef | the $A_{n,m}$ view | empty | Chebyshev interp. matrices | no |
| `build_aux_tables` | `(order, params) → aux` | `build_A_coefficients(2P)` | — | interp. matrices per octant | no |
| `m2l_operators_type` | typedef — **opaque to sweeps** | `View<coeff***>` | `View<Scalar***>` | `{U[level], V[level], core[key]}` | no |
| `bytes_per_key` | `constexpr size_t` | $N_tN_s\cdot16$ | $N_q^2\cdot 8$ | $r^2\cdot 8$ compressed | no |
| `m2l_scratch_bytes` | `(NComps) → size_t` | $N_t N_c\cdot16$ | $N_qN_c\cdot 8$ | $(N_t{+}r)N_c\cdot 8$ | no |

Operators. Signatures are sketches; the point is which arguments appear.

| Operator | Signature sketch | Harmonic | Option 2 | Option 3 | Kernel-dep. |
| --- | --- | --- | --- | --- | --- |
| `p2m_contribution` *(present, `:230-233`)* | `(charges[NC], d, w_self, aux, M_out)` | as today | Taylor moments | anterpolation | **no** |
| `m2m_translate` *(present, `:273-278`)* | `(team, M, child, d, w_c, w_p, aux, M_par)` | Thm 5.22 | binomial shift | interpolation matrix | **no** for 2/3; yes today |
| `build_m2l_operators` | `(keys[], unit_w[], kernel_params) → ops` **host** | as `:1094-1101` | derivative tensors | eval + SVD | **yes** |
| `m2l_pre_cell` | `(team, M, src_cell, ops, scratch)` | no-op | no-op | $\tilde M = V^{\!\top}M$ | yes |
| `m2l_core` | `(team, M, src_cell, ops, op_idx, scratch)` | today's `:1502-1520` inlined loop | dense real matvec | $\hat g \mathrel{+}= C_{\rm key}\tilde M$ | yes |
| `m2l_post_cell` | `(team, scratch, L_out, tgt_cell, ops)` | flush scratch | flush scratch | $L = U\hat g$ | yes |
| `m2l_translate` *(present, `:373-378`)* | per-pair fallback, unchanged shape | as today | derivative tensors on the fly | **route must declare one** | yes |
| `l2l_translate` *(present, `:688-693`)* | `(team, L, parent, d, w_c, w_p, aux, L_ch)` | Thm 5.26 | binomial shift | interpolation matrix | **no** for 2/3 |
| `l2p_evaluate` *(present, `:800-804`)* | `(L, leaf, d, w_self, phi[NC], grad(c,d), do_grad)` | **delete the FD** | differentiate Taylor | interpolate 9 vector sets | **no** — signature already generic |

Note what is *not* in the list: nothing touching the tree builder, the
partitioner, the MAC, the dual-tree traversal, the communication plan or the
CSR. `mac_satisfied` (`src/Canopy_CommunicationPlan.hpp:338-361`) and
`is_well_separated` (`:306-318`) are geometric and stay untouched by all three —
except that the softening floor at `:354-359` becomes dead weight for options 2
and 3, which is add-canopy **X1**'s discriminator and is a configuration change
(`near_softening_factor = 0`), not a code change.

---

## 3. The evidence

### 3.1 Criterion 1 — every site that would need a branch

Twelve sites in shared code encode the harmonic basis. Grouped by what fixes
them:

**(a) Three sites are dead code. Delete them and the leak is gone for free.**

| Site | What it encodes | Callers |
| --- | --- | --- |
| `UpwardSweep::apply_p2m_normalization_bridge` (`src/Canopy_UpwardSweep.hpp:417-449`) | `w^{n+1}` scaling, `n*(n+1)/2+m`, `.real()/.imag()` | **none** (declared `:212`, defined `:418`, never called) |
| `DownwardSweep::apply_l2p_normalization_bridge` (`src/Canopy_DownwardSweep.hpp:1953-1986`) | `w^n`, same index map, same member access | **none** |
| `DownwardSweep::scale_locals_at_depth` (`:1906-1949`) | `w^{n·sign}`, same | **none** |

Also dead: `DownwardSweep::M2L_NUM_SRC` (`:306`), which duplicates
`LaplaceKernel::m2l_num_src_coeffs` (`:488`) and has no reader; and
`DownwardSweep::P` (`:110`), whose only consumer is that dead constant. In
`LaplaceKernel`, `has_mplus_symmetry` (`:159`), `get_coeff_3d` (`:177-193`) and
`m2l_apply_operator` (`:639-672`) have no callers outside the kernel — the last
of these is notable, because it is the operator-apply interface the fused kernel
*should* be going through and does not. `Canopy_SphericalCoefficients.hpp:70-91`
(`get_coeff`) is likewise unreferenced.

`execute()`'s three-stage comment (`src/Canopy_DownwardSweep.hpp:2122-2125`)
already records that the bridges are obsolete: "after step 5 every
multipole/local in the pipeline is in scale-normalized form, so no bridges are
needed". Deleting ~120 lines of dead scaffolding is a prerequisite worth doing on
its own merits.

**(b) Six sites are removed by a `coeff_type` + `scalars_per_coeff` trait.**

| Site | Today |
| --- | --- |
| `src/Canopy_UpwardSweep.hpp:72-73` | `View<complex_type***, LayoutRight>` |
| `src/Canopy_DownwardSweep.hpp:117-118` | same, for locals |
| `src/Canopy_DownwardSweep.hpp:341-342` | same (`LayoutLeft`) for the operator table |
| `src/Canopy_UpwardSweep.hpp:664`, `src/Canopy_DownwardSweep.hpp:2073` | `deep_copy(view, complex_type(0,0))` |
| `src/Canopy_DownwardSweep.hpp:406`, `:1726` | `std::vector<complex_type>` snapshot, zero-filled |
| `src/Canopy_DownwardSweep.hpp:253`, `src/Canopy_UpwardSweep.hpp:179-180` | `CoalescedExchangeBuffers<complex_type, …>` |

The snapshot arithmetic (`:1778` subtract, `:1800` add) works unchanged for a
real type — `operator-`/`operator+` exist for both. These really are typedefs.

**(c) Three sites assume complex arithmetic *structurally*, not by typedef.**
This is the answer to "does something assume complex arithmetic structurally":
**yes, in exactly three places, all of them MPI packing.**

| Site | The structural assumption |
| --- | --- |
| `src/Canopy_MpiCoalescedExchange.hpp:71-72` | `using scalar_type = typename complex_type::value_type;` — a real `double` coefficient has **no** `::value_type`, so this function template does not compile. This is a hard compile failure, not a silent 2× waste. |
| `src/Canopy_MpiCoalescedExchange.hpp:96` | `per_cell_real = 2 * per_cell_complex` |
| `src/Canopy_UpwardSweep.hpp:534-535`, `:581-584` and `src/Canopy_DownwardSweep.hpp:1782-1788` | `reinterpret_cast<scalar_type*>(buf)` with count `2 * total_complex`, and `MPI_DOUBLE`/`MPI_FLOAT` chosen from `sizeof(scalar_type)` |

Each is fixed by the same two traits (`component_scalar_type`,
`scalars_per_coeff`). Encouragingly, `coalesced_view_exchange` is **already**
shape-generic in the other two extents — it reads `view.extent(1)` and
`view.extent(2)` at `:93-94` rather than the compile-time constants — so the
`sets_per_component` change costs nothing there.

**(d) Two sites need a trait to supply a value shared code currently derives
from harmonic reasoning.**

- `src/Canopy_UpwardSweep.hpp:233-235` — the $A_{n,m}$ table, built to $2P$
  because "M2L accesses A at degree n+j where both n and j go up to P". Shared
  code owns a table whose *existence* is basis-specific. `DownwardSweep` borrows
  it (`:529`) and passes it into three kernel operators (`:1099-1100`,
  `:1636-1639`, `:1688-1691`), plus `UpwardSweep:501-504`. Fixed by
  `aux_tables_type` + `build_aux_tables` and dropping `A_table` from every
  operator signature in favour of `aux`. Option 2 supplies an empty struct;
  option 3 supplies its interpolation matrices, which it wants anyway.
- `src/Canopy_DownwardSweep.hpp:301-302` — `M2L_KEY_DD_MAX` branches on
  `KernelType::scalar_type` being `float`, with a rationale (`:296-300`) derived
  entirely from the harmonic scale normalization: "the |dd|-dependent factor
  reaches $2^{j\cdot|dd|}$ (worst j = P)". For a non-homogeneous kernel that
  rescaling does not exist, so the constant is meaningless. Fixed by
  `m2l_key_dd_max`.

**(e) One site is not fixable by a trait and must be moved.** The fused M2L
inner loop, `src/Canopy_DownwardSweep.hpp:1496-1526`, hardcodes:

- the packed triangular storage index, `n*(n+1)/2 + abs_m` (`:1508-1509`);
- the flat source index `n*n+n+m` (`:1506`);
- the conjugate-symmetry expansion for $m<0$ (`:1512-1517`);
- the $(n, m)$ loop bounds from `P_local` (`:1502-1504`);
- complex accumulation and multiply (`:1501`, `:1518`);
- a real/imag *split* scratch accumulator, chosen deliberately to halve shared-
  memory bank conflicts (`:1454-1462`, `:1478-1486`, `:1523-1524`, `:1538-1541`).

No trait removes this. The whole contraction is basis-specific and the scratch
layout is a basis-specific optimization currently owned by the *sweep*. It has to
become a kernel method — which is condition **C-1** and
[§3.2](#32-the-m2l-operator-table-the-most-likely-single-point-of-failure).

**(f) Two sites outside the sweeps, both bounded.**

- **Softening never reaches the far field.** `FmmConfig::softening`
  (`src/Canopy_Solver.hpp:69`) is routed to `_p2p.set_softening` and
  `_comm_plan.set_near_softening` only — in the constructor (`:169-178`) and in
  `_init_auto_softening` (`:698-705`). Nothing passes it to `_upward` or
  `_downward`, and `m2l_build_operator` is a **static** method whose arguments
  are `(dd, ix, iy, iz, A_table, T_out)` (`src/Canopy_LaplaceKernel.hpp:516-519`)
  — integers only. Options 2 and 3 need $b$ *and* the physical unit width. So the
  kernel contract must stop being purely static in its operator-*construction*
  half: `build_m2l_operators` takes a `kernel_params` argument, and `Solver`
  must forward `softening` (and the root box, available at `:685`) into
  `_downward` before the table build. Plumbing, not a branch, but it is shared
  code and it is unavoidable for both new routes.
- **P2P never calls the kernel.** It uses only `scalar_type` and
  `num_components` (`src/Canopy_P2P.hpp:70-72`) and inlines
  $1/\sqrt{r^2+\varepsilon^2}$ and $-q\,\delta/(r^2+\varepsilon^2)^{3/2}$
  directly (`:883-896`, and again in the inter-leaf kernel near `:1082`). For
  these three routes that is **not** a leak — all three share the same softened
  near kernel, and it is already the right one (canopy0.md **F1**). It *is* a
  leak for the claim "option 3 generalizes to other kernels": a fourth kernel
  needs P2P changed. Name it, bound it, do not do it now.

### 3.2 The M2L operator table: the most likely single point of failure

The prompt is right that this is where a superficial reading says "same shape,
fine". Here is what the code says.

**Can one table abstraction hold a dense per-key operator and a factored
shared-basis-plus-core operator? Yes — but only by removing the table from the
sweeps' ownership.**

The sweeps' contact with `_m2l_op_table` is remarkably thin — **five sites**:

| Site | What it does |
| --- | --- |
| `src/Canopy_DownwardSweep.hpp:341-342` | declares `View<complex_type***, LayoutLeft>` |
| `:622-623` | resets it to a default-constructed view in `setup()` |
| `:1079-1109` | builds it: `(Nt, Ns, n_unique_ops)` on host, serially, one `m2l_build_operator` call per key, then one `deep_copy` |
| `:1449` | captures it by value for the device lambda |
| `:1518` | indexes it — `op_table(out_idx, j, op_idx)` |

Only the last is in device code, and only the last is shape-committing. If sites
3-5 move behind `build_m2l_operators` / `m2l_pre_cell` / `m2l_core` /
`m2l_post_cell`, the sweeps carry nothing but `int op_idx` and the CSR. Then:

- **harmonic** fills `m2l_operators_type` with today's exact
  `View<complex<Scalar>***, LayoutLeft>` and a no-op pre/post;
- **option 2** fills it with a real `(N_q, N_q, key)` view and a no-op pre/post;
- **option 3 compressed** fills it with `{U[level], V[level], core[key]}` and
  non-trivial pre/post.

**Why the pre/post-pass shape is not optional.** The Fong & Darve shared-basis
form is
$$
L^A \mathrel{+}= U_\ell \Big( \sum_{\text{pairs}} C_{\rm key}\, \big(V_\ell^{\!\top} M^B\big) \Big),
$$
and the *efficient* evaluation computes $V_\ell^{\!\top}M^B$ **once per source
cell** and $U_\ell(\cdot)$ **once per target cell**, with only the small
$r\times r$ core inside the pair loop. Doing $V^{\!\top}M$ inside the pair loop
throws away most of the compression's flop win. Today's driver has no place to
put a per-cell pass: the loop at `:1489-1527` is strictly per-pair inside a
per-target team, and the only per-target work is the zeroing (`:1481-1487`) and
the write-back (`:1533-1542`). **The post-pass fits naturally where the
write-back already is; the pre-pass does not fit anywhere and is the one genuinely
new structural element.** It is also, per
[§6](#6-option-4-as-a-falsification-test), exactly what option 4's FFT M2L
needs — which is the strongest argument for declaring it now rather than later.

**The key, the cap and the overflow path, checked against all three.**

- The key struct is `{dd, ii, jj, kk}` (`:308-318`) with an FNV-style hash
  (`:319-335`). **The class comment at `:285-290` says the key is
  `(max_d, dd, ii, jj, kk)` — the struct has no `max_d`. The comment is stale;
  the code wins.** `max_d` *is* computed in the classify pass (`:870`) and
  discarded, so extending the key costs one field, one `mix()` call and no new
  computation.
- **The clean way to extend it without branching in shared code** is to have the
  classify pass *always* emit `max_d` and then apply
  `KernelType::canonicalize_key` before hashing. The harmonic kernel zeroes
  `max_d` — reproducing today's key set **exactly**, which is what criterion 4
  needs — and the softened kernels return the key unchanged. That is a trait
  call, not an `if constexpr`, and it satisfies criterion 1.
- **Physical width has to come back, and the code deliberately removed it.** The
  S3 classify pass is a pure-integer pipeline precisely so it is "bit-identical
  M2LKey output by construction" with "no per-source `h_dc_for_filter` gather and
  no FP rounding" (`:778-787`). `half_width_at_depth` is computed **only under
  `CANOPY_ENABLE_DEBUG`** (`:742-744`, `:762-776`), so in a release build the
  interaction-list builder knows no physical length at all. The fix does not
  reinstate the gather: `unit_w(max_d) = w_{\rm root}/2^{max\_d}` is exact from
  `TreeBuilder::root_box()` (used at `src/Canopy_Solver.hpp:685`), one array of
  `max_depth+1` doubles handed to `build_m2l_operators`. Integer keys stay
  integer; only the *builder* sees lengths.
- **The cap is a count and must become a budget.** `M2L_OP_COUNT_CAP = 32768`
  (`:304`); overflow assigns `op_idx = -1` with one `fprintf` warning
  (`:1028-1047`) and routes those pairs to the per-pair `m2l_translate` fallback
  (`:1244-1314` builds the table, `:1600-1641` runs it). Two problems for the new
  routes: (i) once the key carries a level the key count multiplies by occupied
  depth, so the cap is reached far sooner; (ii) 32768 keys means wildly different
  memory per route — 1.9 GB harmonic, 320 MB for option 2 at $p=4$, 36 TB for
  option 3 at $n=6$ uncompressed. Hence **C-2**.
- **The overflow path is a real gap for option 3.** The fallback is "evaluate the
  operator on the fly, per pair, on device" (`m2l_translate`,
  `src/Canopy_LaplaceKernel.hpp:373-481`). Option 2 can do that — the derivative
  ladder is device-evaluable, at the register cost
  [canopy-kernel-rec.md](canopy-kernel-rec.md) already flags. Option 3 cannot
  usefully do it: an on-the-fly $n^3\times n^3$ kernel evaluation per pair is
  $n^6$ kernel calls. So option 3 must declare *some* fallback — most plausibly
  "escalate the pair to P2P", which the traversal already supports as its
  complementary branch (`src/Canopy_CommunicationPlan.hpp:300-304`, `:623-634`
  per canopy0.md **F1**) but which the *downward sweep* has no path to trigger.
  **This is the sharpest single unresolved item in the design** and it is
  criterion 3's real risk: if option 3's fallback is not declared as part of the
  interface now, a later option-3 session reopens the classify pass.

**Verdict on this tension:** one table abstraction holds both, at the price of
(i) declaring M2L as three stages rather than one, (ii) making the operator set
opaque, (iii) replacing the cap with a byte budget, and (iv) declaring a
per-route overflow policy. None of these is large in lines. All four must be
decided *before* option 2 is written, because all four are shared-code shape.
That is the honest content of "paid early for option 3".

### 3.3 Criterion 3 — what a later option-3 session would open

Assuming C-1 through C-4 are honoured in the option-2 session:

| File | Opened? | Why |
| --- | --- | --- |
| new `src/Canopy_ChebyshevBasis.hpp` | yes | the whole route |
| `src/CMakeLists.txt` | yes, one line | add to `HEADERS_PUBLIC` (`:4-16`) |
| `src/Canopy_Solver.hpp` | **no** | the far field is a defaulted template-template parameter; Beatnik's adapter names the new type |
| `src/Canopy_UpwardSweep.hpp` | **no** | traits + operator calls only |
| `src/Canopy_DownwardSweep.hpp` | **no** | *only if* the three-stage M2L, the byte budget and the overflow policy are already in place |
| `src/Canopy_CommunicationPlan.hpp` | **no** | geometric; the softening floor is disabled by configuration |
| `src/Canopy_P2P.hpp` | **no** | already the softened kernel |
| `CMakeLists.txt` (top level) | **no** | Trilinos is already required and linked — see below |
| `tests/` | yes, new | new members, plus one line in `tests/CMakeLists.txt:35-64` |

**On the dense-linear-algebra dependency.**
[canopy-kernel-rec.md](canopy-kernel-rec.md) states that Canopy's CMake "finds
Kokkos, Cabana and GTest, and no BLAS/LAPACK". **That is stale.** Trilinos is
found and marked `TYPE REQUIRED` (`CMakeLists.txt:73-74`) for load balancing,
and `src/CMakeLists.txt:44`/`:52` link `${Trilinos_LIBRARIES}` and include
`${Trilinos_INCLUDE_DIRS}` unconditionally; `Canopy_TreePartitioner.hpp:22-29`
uses Zoltan2, Teuchos and Tpetra. In this environment's include view,
`Teuchos_LAPACK.hpp` declares `GESVD` for `double` (`:1519`) and for `float`,
`std::complex<float>` and `std::complex<double>`, and
`KokkosBlas3_gemm.hpp`, `KokkosLapack_gesv.hpp` and
`KokkosBatched_SVD_Decl.hpp` are all present. **Inference** (confirmable only by
a compile, which is out of scope): option 3's host SVD needs no new
`find_package` and no new link line — `Teuchos::LAPACK<int,double>::GESVD` is
already reachable. The abstraction therefore does not have to accommodate a
dependency that exists for one member and not the others; the dependency already
exists for all of them.

**Where operator construction happens, for each route.**

| | Where | Host/device | Cost per rebuild | Survives `migrate`/`rebalance`? |
| --- | --- | --- | --- | --- |
| harmonic today | `src/Canopy_DownwardSweep.hpp:1088-1102`, inside `build_interaction_list_device` | host, **serial** loop over keys, one `deep_copy` (`:1103-1107`) | cheap | **No** — the whole table is rebuilt whenever `_interaction_list_dirty` (`:645-646`), set by `setup()` (`:637`) and `invalidate_interaction_list()` (`src/Canopy_Solver.hpp:565`, `:610`) |
| option 2 | same place | host, serial (parallelizable) | cheap; tables are $\le$ 56 KB/key | no, and it does not matter |
| option 3 | same place + an SVD stage | host, LAPACK | **an SVD per key per level** | no, **and it matters enormously** |

The documents' mitigating claim — that the tables depend only on
(level, offset, $b$) and not on particle positions, so they are built once at
construction ([canopy-bbFMM.md](canopy-bbFMM.md) cost 1;
[canopy-kIndp.md](canopy-kIndp.md) cost 1) — **is true of the mathematics and
false of the current code.** The code rebuilds on the dirty flag, and the dirty
flag is set by every topology change, which add-canopy **R5**'s "maintain before
every solve" discipline makes routine. Hence **C-3**: split construction into a
geometry-keyed cache persisting across topology changes (option 3's expensive
half) and the per-tree key→`op_idx` map (cheap, rebuilt as today). Option 2 does
not need this. It is the second-largest item in the "paid early" pile.

### 3.4 Criterion 4 — bit-for-bit identity, and why it is currently unverifiable

**There is no bit-for-bit test in Canopy.** The tightest full-pipeline assertion
is `SolveFusedM2L.matchesPriorReference` (`tests/tstMultiSolve.hpp:915-931`),
which compares against a brute-force $N^2$ sum with bounds
$5\times10^{-2}$ on the potential and $1\times10^{-1}$ on the gradient
(`:929-930`), and whose own comment says it exists to "catch a
complete-regression bug" — "even a ~5% bound would fire on, e.g., a sign error"
(`:926-928`). `sweepConvergence` (`:939`) checks monotonicity in $P$;
`multipleSolvesIdempotent` (`:982`) checks within-run repeatability, not
cross-version identity. The per-operator unit tests in `tests/tstLaplace.hpp`
(`:49-828`) are the right *granularity* but compare against analytic or direct
references with tolerances, not against stored bytes.

So every risk below is currently **undetectable** by the suite:

| Risk | Mechanism | How to pin it |
| --- | --- | --- |
| **Changed arithmetic order in M2L** | Moving `:1496-1526` into `m2l_core` must preserve the $n$-ascending, $m=-n..n$ accumulation *and* the real/imag split scratch (`:1454-1462`, `:1523-1524`). If the kernel is handed a `complex` scratch instead, the summation is mathematically identical and bitwise different. | Golden dump of `locals()` after `execute()`, compared with `EXPECT_EQ` on the bit pattern, at fixed rank count/`P`/`ncrit`/`max_depth`/seed |
| **Changed layout** | `coeff_type = Kokkos::complex<Scalar>` preserves layout exactly; a "generalization" to `struct {Scalar re, im;}` or to split real/imag planes would not. The op table's `LayoutLeft` (`:337-342`) is chosen so a subview is a contiguous column-major $(N_t,N_s)$ matrix consumable by cuBLAS without transpose — a `coeff_type` change must keep it. | Golden dump of `_m2l_op_table` for a fixed key set; assert `LayoutLeft` (the existing `DownwardSweepLayout.layoutTagIsLayoutRight` test at `tests/tstDownwardSweep.hpp:1329` is the model) |
| **Changed A-table** | `build_A_coefficients` uses `Kokkos::tgamma` (`src/Canopy_SphericalCoefficients.hpp:113-119`); moving *where* it is called does not change values, but changing the `2*P` argument (`src/Canopy_UpwardSweep.hpp:235`) would silently zero high-degree entries — and `m2l_build_operator` already `continue`s on `A == 0` (`src/Canopy_LaplaceKernel.hpp:612-614`), so the failure is a *quietly wrong* operator, not a crash | Golden dump of the A-table and its extent |
| **`op_idx` renumbering** | Extending the key changes hash values, hence per-thread map bucket order, hence `local_ops[t]` order, hence the global `op_idx` assignment (`:1014-1050`). **Numerically inert**, because the per-target summation order comes from the CSR (built from `entries` order, `:1197-1205`), not from `op_idx`. Worth stating so nobody "fixes" it. | Covered by the `locals()` golden |
| **Changed overflow set** | Which pairs get `op_idx = -1` decides which take the per-pair `m2l_translate` path — *different arithmetic* from the operator path. A byte-budget cap (C-2) must be set so the harmonic path's overflow set is empty exactly as today. | `total_fallback_pair_count()` (`:424-430`) is already exposed for the bin-edge test; assert it is 0 for the golden configuration |
| **Lost specialization** | `num_coeffs_per_cell`, `Nt`, `NComps` are `constexpr` and drive unrolling (`:1443-1445`, `:1499`). If any becomes a runtime value through a trait indirection, the fused kernel deoptimizes — a *performance* regression that no correctness test sees. | Compare the profiling breakdown (`CANOPY_PRINT_SOLVE_BREAKDOWN`, `src/Canopy_Solver.hpp:238`) before/after; not a correctness gate |

**One favourable finding.** Bit-for-bit identity is *achievable* at fixed rank
count despite `M2LPlan::interaction_lists` being a `std::unordered_map`
(`src/Canopy_CommunicationPlan.hpp:95-96`), whose iteration order is not
guaranteed: `entries` is subsequently `std::sort`ed by `(depth, target_idx)`
(`src/Canopy_DownwardSweep.hpp:713-719`), a total order over distinct targets,
and within-entry source order is the traversal's deterministic push order
(`src/Canopy_CommunicationPlan.hpp:481`). So the CSR — and therefore the
summation order — is deterministic. That is what makes a golden test meaningful
rather than flaky. (It says nothing about canopy0.md **C4**, cross-*rank-count*
reproducibility, which is a different and still-open question.)

### 3.5 Width normalization

Five conventions exist for FP32 conditioning at depth, all inside the kernel
struct: P2M produces $\bar M = M/w^{n+1}$
(`src/Canopy_LaplaceKernel.hpp:226-229`), M2M applies $(w_c/w_p)^{j+1}$
(`:267-272`), M2L expands $\rho^{-(n+j+1)}$ as
$(w_s/\rho)^{n+1}(w_t/\rho)^{j}$ (`:369-372`), L2L applies $(w_c/w_p)^{j}$
(`:685-687`), L2P consumes $\bar L = L\,w^{n}$ (`:795-799`). A sixth lives in
the operator builder: the `F_row`/`F_col` factors $2^{j\,dd}$ and
$2^{-(n+1)dd}$ (`:540-561`), which are what make the operator depth-independent.
All six are homogeneity arguments and all six die at $b>0$.

**Is normalization a per-basis concern the trait owns, or is it baked into
shared sweep code? Almost entirely the former.** The sweeps only *plumb* widths:
they pass `w_child/w_parent` (`src/Canopy_UpwardSweep.hpp:501-504`),
`w_source/w_target` (`src/Canopy_DownwardSweep.hpp:1636-1639`),
`w_child/w_parent` again (`:1688-1691`) and `w_self`
(`:2029-2031`) from `DeviceCellInfo::half_width`
(`src/Canopy_UpwardSweep.hpp:83-85`). That plumbing is basis-agnostic and stays.

Two exceptions, and one is free:

1. `M2L_KEY_DD_MAX`'s `float` cut (`src/Canopy_DownwardSweep.hpp:296-302`) is a
   normalization-derived constant in shared code — fixed by the
   `m2l_key_dd_max` trait (§3.1(d)).
2. The three dead scaffolding functions hardcode `w^n` scaling in shared code
   (§3.1(a)) — fixed by deleting them.

So: **the trait owns normalization**, and the exceptions cost one trait and one
deletion. For the two new routes the honest position is that there is nothing to
re-derive — a softened kernel has no scale invariance to exploit, so both routes
should carry *physical* operators keyed by level and state
`Scalar = double` as a precondition, per
[canopy-kernel-rec.md](canopy-kernel-rec.md)'s closing note. That is simpler
than the present arrangement, not harder.

### 3.6 L2P and the gradient

All three routes delete the finite-difference gradient
(`src/Canopy_LaplaceKernel.hpp:851-879`, with its in-code
`TODO: replace with analytical derivatives` at `:860`) — canopy0.md **F7**,
**C11**.

**Does one L2P interface cover both? Yes — the signature is already generic.**
`l2p_evaluate(L_full, leaf_cell, dx, dy, dz, w_self, phi_out[NComps],
grad_out(c,d), compute_gradient)` (`:800-804`) names no basis concept: inputs
are the locals view, a cell index, an offset and a width; outputs are
`phi[NComps]` and a 2-D accessor. The call site (`:2029-2031`) hands it a
`GradWriter` struct writing `grad(p,c,d)` (`src/Canopy_DownwardSweep.hpp:133-141`,
existing to dodge CUDA's nested-extended-lambda restriction). Option 2
differentiates its Taylor local; option 3 interpolates its nine vector local
sets. Both fit.

**One shared-code shape change is required.** `_locals` is allocated
`(num_cells, coeffs_per_cell, NComps)` (`:534`) and the third extent is
`NComps` in the snapshot pack (`:1736-1739`) and the Allreduce pack/unpack
(`:1774-1779`, `:1796-1801`). Option 3 under the fixed output contract descends
**nine** local sets per charge component, so the third extent becomes
`NComps * sets_per_component`. `coalesced_view_exchange` already reads
`view.extent(2)` (`src/Canopy_MpiCoalescedExchange.hpp:94`) so it needs nothing;
only the allocation and the two hand-rolled Allreduce loops change.

**Assumption I had to make.** `solve()` unconditionally allocates and zeroes the
potential (`src/Canopy_Solver.hpp:209-211`) and P2P accumulates `phi[c]` inside
the innermost pair loop unconditionally (`src/Canopy_P2P.hpp:891`), so the potential
output is mandatory until canopy0.md **C8** lands. For option 3 that means a
**tenth** interpolated quantity — the scalar $\varphi$ — and a tenth local set,
i.e. `sets_per_component = 10`, not 9. If C8 lands first and a gradient-only
solve is possible, it is 9. Flagging, not renegotiating.

**The fixed contract's cost, in one sentence, as instructed.** Keeping the
$\epsilon_{ilm}$ contraction in Beatnik's adapter costs option 3 a factor of
three in descending local expansions, in L2L work, and in the shared-cell local
Allreduce payload (`:1782-1788`) relative to contracting inside M2L — the trade
[canopy-bbFMM.md](canopy-bbFMM.md) records under "Does it need only one solve?"
— and it is not this document's to reopen.

---

## 4. What the abstraction costs

### 4.1 Option 2 needs this anyway

| Work | Sites | Rough size |
| --- | --- | --- |
| Delete dead harmonic scaffolding | `src/Canopy_UpwardSweep.hpp:212`, `:417-449`; `src/Canopy_DownwardSweep.hpp:306`, `:110`, `:474`, `:481`, `:1906-1949`, `:1953-1986`; `src/Canopy_LaplaceKernel.hpp:159`, `:177-193`, `:639-672`; `src/Canopy_SphericalCoefficients.hpp:70-91` | −150 lines |
| `coeff_type` + `scalars_per_coeff` + `component_scalar_type` traits through both sweeps | the six typedef sites of §3.1(b) | ~30 lines |
| Fix the three structural complex assumptions in MPI packing | `src/Canopy_MpiCoalescedExchange.hpp:71-72`, `:96`; `src/Canopy_UpwardSweep.hpp:534-535`, `:581-584`; `src/Canopy_DownwardSweep.hpp:1782-1788` | ~25 lines |
| Move auxiliary-table ownership into the kernel; drop `A_table` from five operator signatures | `src/Canopy_UpwardSweep.hpp:233-235`, `:501-504`, `:529`; `src/Canopy_DownwardSweep.hpp:1099-1100`, `:1636-1639`, `:1688-1691` | ~40 lines |
| `m2l_key_dd_max` trait | `src/Canopy_DownwardSweep.hpp:301-302` | ~5 lines |
| Key carries `max_d`; `canonicalize_key` trait; hash extended | `:308-335`, `:869-919`, `:1010-1050` | ~30 lines |
| Plumb `softening` + per-level `unit_w` to the operator builder | `src/Canopy_Solver.hpp:169-178`, `:698-705`; `src/Canopy_DownwardSweep.hpp:1079-1109` | ~40 lines |
| `Solver` gains the defaulted far-field template-template parameter | `src/Canopy_Solver.hpp:104-119`, `:719-727` | ~15 lines |
| `sets_per_component` for the locals' third extent | `src/Canopy_DownwardSweep.hpp:534`, `:1736-1739`, `:1774-1801` | ~20 lines |
| Delete the FD L2P (= canopy0.md **C11**) | `src/Canopy_LaplaceKernel.hpp:851-879` | −30 lines |

Roughly **250 lines net of shared-code change, half of it deletion**. Every item
is a precondition of a real Cartesian-Taylor far field regardless of whether
option 3 ever exists.

### 4.2 Paid early for option 3

Four items. All four are **interface shape**, and option 2's implementation is
not slowed by any of them — it fills them with no-ops and dense tables.

| Item | What it is | Size | If deferred |
| --- | --- | --- | --- |
| **Three-stage M2L** (C-1). Split `run_m2l_fused` (`src/Canopy_DownwardSweep.hpp:1431-1544`) into a basis-agnostic driver and three kernel calls; kernel declares scratch bytes; make the operator set opaque (`:341-342`, `:1449`, `:1518`) | The one item that cannot be a trait. The scratch-split optimization (`:1454-1462`) must be preserved *inside* the harmonic kernel. | ~120 lines of shared code, plus re-homing ~40 lines into `LaplaceKernel` (where a dead `m2l_apply_operator` at `:639-672` is already 80% of it) | option 3 reopens the hottest kernel in the sweep, and criterion 3 fails |
| **Byte-budget cap + per-route overflow policy** (C-2). `bytes_per_key` trait; cap becomes memory; option 3 must declare a fallback that is not per-pair on-the-fly evaluation | The sharpest unresolved design point (§3.2) | ~40 lines, plus a decision | option 3 reopens the classify pass and the cap; criterion 3 fails |
| **Split operator construction** (C-3). Persistent geometry-keyed cache + per-tree index map; `build_m2l_operators` called only for keys not already cached | Option 2's tables are cheap to rebuild; option 3's are not | ~80 lines | option 3 reopens `build_interaction_list_device` *and* `Solver`'s maintenance paths; criterion 3 fails |
| **`sets_per_component ≠ 1` exercised** | Option 2 uses 1. Building the trait but never testing it $\ne 1$ means option 3 discovers the bugs. | ~0 lines, one test | latent bugs in the Allreduce packing surface during option 3 |

**Total early payment: ~240 lines of shared-code restructuring and one design
decision.** Set against ~250 lines option 2 needs anyway, the abstraction
roughly doubles the shared-code work — but it does *not* double option 2's total
work, because option 2's own cost is dominated by its derivation and its new
kernel struct, not by shared-code plumbing (canopy0.md **F6(e)**: M2L is the
only kernel-touching operator, and the delicate part is the sign convention
between the moment definition and $(-1)^{|q|}$, add-canopy **X1**).

So criterion 2 **passes narrowly**: the abstraction does not spend the time it
was meant to save, but the margin is not comfortable and it depends entirely on
the four items above being *declarations* rather than implementations. If any of
them turns into implementation work during the option-2 session, criterion 2
fails and option 3 should be built directly instead.

### 4.3 The fallback if C-1 is rejected

Take §4.1 and nothing from §4.2. This yields:

- storage, coefficient count, MPI packing, auxiliary tables, keying, softening
  plumbing and the solver parameter all generalized;
- the M2L *apply* left exactly as it is today, harmonic-specific, bit-identical
  by construction — which turns criterion 4 from a risk into a near-certainty;
- options 2 and 4-unaccelerated both expressible (option 4's five operators are
  all dense matvecs against per-level tables — [canopy-kIndp.md](canopy-kIndp.md)
  "The five operators" — and its unaccelerated form is exactly the shape today's
  driver has);
- option 3 requiring a **second** M2L driver alongside the first, written when
  option 3 is written.

The cost of the fallback is that option 3 later duplicates ~150 lines of driver
(team launch, CSR walk, scratch, write-back) rather than reusing one. That is a
real but bounded duplication, and it is the *correct* trade if the answer to
§5's threshold question turns out to be that option 3 is not memory-viable on
this tree at any useful $n$.

---

## 5. Failure modes and the cheapest falsifying checks

For the verdict above to be **wrong**, one of these would have to be true.

| # | What would have to be true | Cheapest check | If it fails |
| --- | --- | --- | --- |
| 1 | **The harmonic M2L cannot be moved into the kernel without perturbing its arithmetic.** Plausible: the scratch-split accumulator (`src/Canopy_DownwardSweep.hpp:1454-1462`, `:1523-1524`) is a sweep-owned optimization, and any kernel-facing scratch abstraction that hands back a `complex` view changes the summation. | Write the golden test (**C-4**) *first*, then do the M2L move *alone*, on the harmonic path only, and confirm bitwise equality before any other change. **This is the whole plan's gate and it costs one test plus one commit.** | C-1 fails → fall back to §4.3 |
| 2 | **Option 3 is not memory-viable, so the abstraction holds a member that never gets built.** The code's own comment says the realized key count is "globally ~16 k under MAC=0.5" (`:37-41`), 8× the 2000 used in [canopy-kernel-rec.md](canopy-kernel-rec.md)'s worked example. | Instrument `n_unique_ops` (`:1070`) and print it on a milestone-0 tree — the one-line change the recommendation note already asks for. Then compare $\texttt{bytes\_per\_key}\times n_{\rm unique}\times L$ against per-GPU memory left after particles and tree. **Threshold: if that product exceeds available HBM at every $n \ge 6$ even with shared-basis compression at $r=50$, option 3 is not buildable here and option 2 standalone is correct.** | verdict inverts: skip the abstraction, build option 2 directly |
| 3 | **The `canonicalize_key` trick does not preserve today's key set**, so the harmonic path silently gets a depth-multiplied table. | Golden dump of `ops.size()` and of the sorted key list for a fixed tree, before and after. | criterion 4 fails on the memory axis; recoverable by keeping two key types |
| 4 | **Option 3's overflow policy has no acceptable answer**, so the cap must never be reachable — which at 6 MB/key it will be. | Same measurement as #2, read against the byte budget rather than the count | C-2 fails → option 3's viability, not the abstraction's, is in question |
| 5 | **Trilinos does not in fact export `teuchosnumerics` in `Trilinos_LIBRARIES`**, so the SVD does need a new dependency after all. Marked as **inference** in §3.3. | A one-TU compile-and-link of `Teuchos::LAPACK<int,double>::GESVD` against the existing Canopy target. Out of scope here (build nothing). | a `find_package(LAPACK)` line; trivial |
| 6 | **`sets_per_component ≠ 1` breaks the shared-cell Allreduce** in a way the option-2 session never sees. | One `unit` test running the existing harmonic path with an artificially doubled third extent | latent, cheap to fix if found early |

Checks 1 and 2 are the two that matter, and they are independent: check 1 decides
whether the *abstraction* works, check 2 decides whether *option 3* is worth
abstracting for. **Both are cheap. Neither has been done.** Recommended order:
write the golden test, run check 2's instrumentation, then decide.

---

## 6. Option 4 as a falsification test

Option 4 is not being implemented ([canopy-kIndp.md](canopy-kIndp.md)'s
conclusion: its convergence theory does not survive softening, and its
regularized pseudo-inverse floors the achievable error in exactly the band
add-canopy **X1** asks for). It appears here only to test whether "option 3
generalizes to other kernels" is a real claim about the *abstraction* or just
about the method.

**Could KIFMM later join the proposed abstraction without reopening the
interface? Yes for the unaccelerated form; yes for the FFT form too, but only
because of a hook option 3 already needs.**

- **Traits.** Coefficients are real surface densities, count
  $6k^2-12k+8$ ([canopy-kIndp.md](canopy-kIndp.md) "The one idea"), so
  `coeff_type = Scalar`, `scalars_per_coeff = 1`, `num_coeffs_per_cell` from
  $k$. Fits.
- **Operators.** All five are "call the kernel between two point sets, then
  multiply by a stored matrix" (ibid., "The five operators"), i.e. dense matvecs
  against per-level tables. `aux_tables_type` holds the per-level regularized
  pseudo-inverses — exactly what `build_aux_tables` exists for. Notably, M2M and
  L2L *call the kernel* here, unlike in option 2 where they are pure
  combinatorics; the operator table above already marks those cells
  route-dependent, so nothing in the contract objects.
- **Keying.** Per-(level, offset) precomputation with softening killing scale
  invariance (ibid., cost 1) — the same key extension options 2 and 3 need, and
  `canonicalize_key` is identity. Fits.
- **The FFT-accelerated M2L is the interesting stress case, and it is the reason
  to declare C-1 now.** The convolution form embeds the source equivalent
  density in a regular grid, convolves, and extracts on the target grid (ibid.,
  "What survives intact"). The forward transform is natural **once per source
  cell**, the inverse **once per target cell**, with only the pointwise grid
  multiply per pair. Today's driver has no place for either — the loop at
  `src/Canopy_DownwardSweep.hpp:1489-1527` is strictly per-pair. Under C-1's
  three-stage contract, the forward transform *is* `m2l_pre_cell`, the pointwise
  multiply *is* `m2l_core`, and the inverse *is* `m2l_post_cell`.
- **What it would still reopen:** `Canopy_P2P.hpp`, if and only if the kernel
  changes — and for option 4 on Beatnik's kernel it does not (§3.1(f)). Plus a
  batched-FFT dependency, which `Kokkos`/`KokkosKernels` may or may not supply;
  unchecked here.

So the falsification test **passes**, and it passes for a specific and
non-obvious reason: the per-cell pre/post hooks that option 4's FFT form needs
are the same hooks option 3's shared-basis compressed form needs. That
convergence is the strongest single argument in this document for declaring the
three-stage M2L contract in the option-2 session rather than deferring it — and
correspondingly, if C-1 is rejected, both option 3-compressed and option 4-FFT
are outside the abstraction, and the claim "option 3 generalizes to other
kernels" reduces to a claim about the *method* only, not about Canopy.

---

## Appendix — documents corrected against the code

Where a cited document disagrees with the code, the code wins. Four cases:

1. **[canopy-kernel-rec.md](canopy-kernel-rec.md), "How much refactoring":**
   "its `CMakeLists.txt` finds Kokkos, Cabana and GTest, and no BLAS/LAPACK".
   **Stale.** Trilinos is `TYPE REQUIRED` (`CMakeLists.txt:73-74`) and linked at
   `src/CMakeLists.txt:44`; `Teuchos_LAPACK.hpp` with `GESVD` for `double`
   (`:1519`) is in this environment's include view. Option 3's SVD needs no new
   dependency (inference on the export list; see §5 check 5).
2. **[canopy0.md](canopy0.md) F6(b):** the sweeps "hard-assume …
   `num_coeffs_per_cell = (P+1)(P+2)/2` (`:66`, `:111`)". **Imprecise, in the
   favourable direction.** Those lines read the count *from the kernel trait*
   (`src/Canopy_UpwardSweep.hpp:66`, `src/Canopy_DownwardSweep.hpp:111`); the
   formula lives only in `src/Canopy_LaplaceKernel.hpp:157` and
   `src/Canopy_SphericalCoefficients.hpp:47-50`. The real hardcoding is the
   *packing* — the triangular index and conjugate symmetry at
   `src/Canopy_DownwardSweep.hpp:1506-1517`.
3. **`src/Canopy_DownwardSweep.hpp:285-290`** (in-code): describes the key as
   `(max_d, dd, ii, jj, kk)`. **Stale comment** — `M2LKey` (`:308-318`) has only
   `{dd, ii, jj, kk}`. Favourable: `max_d` is already computed at `:870` and
   discarded, so extending the key is one field.
4. **[canopy-bbFMM.md](canopy-bbFMM.md) cost 1 and
   [canopy-kIndp.md](canopy-kIndp.md) cost 1:** the operator tables "survive
   every `migrate` / `rebalance` / `auto_maintain`". **True of the mathematics,
   false of the code** — `build_interaction_list_device` rebuilds the whole table
   on the dirty flag (`:645-646`, `:1079-1109`), which every topology change
   sets. Hence condition C-3.

Two further in-code facts worth recording because they are not in any document:

- The realized M2L key count is stated in a tuning comment as **"globally ~16 k
  under MAC=0.5"** (`src/Canopy_DownwardSweep.hpp:37-41`). This is a claim in a
  comment, not a measurement made here, and it is 50× the textbook 316-offset
  V-list figure.
- **Canopy has no bit-for-bit regression test.** The tightest full-pipeline bound
  is $5\times10^{-2}$ / $1\times10^{-1}$ (`tests/tstMultiSolve.hpp:929-930`).
  This is why criterion 4 fails today and why C-4 is sequenced first.
