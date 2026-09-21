/****************************************************************************
 * Copyright (c) 2025 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file Beatnik_Params.hpp
 * @brief Plain-old-data parameter structs, one per Python parameter group.
 *
 * These carry **only** values; no behavior, no stubs. Defaults are transcribed
 * from `run_adaptive_mesh_bubble.py::parse_args` (lines 64-533) and from the
 * dataclass defaults in `mesh_solver.py::MeshZModelParams` (lines 33-56) and
 * `dynamic_remesh.py::DynamicRemeshParams` (lines 17-46). Where the two
 * disagree the CLI default wins, because the driver always constructs the
 * dataclass from `args` (`run_adaptive_mesh_bubble.py:1248-1271`).
 *
 * Units: the reference problem is nondimensional. Lengths are in units of the
 * bubble radius scale (`--radius`, default 0.25), time in units where the
 * gravitational acceleration `--g` is 1, and velocity in length/time. No
 * routine in the port introduces a dimensional constant.
 */

#ifndef BEATNIK_PARAMS_HPP
#define BEATNIK_PARAMS_HPP

#include <Beatnik_Types.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Physics and Birkhoff-Rott parameters for the z-model right-hand side.
 *
 * Port of mesh_solver.py::MeshZModelParams (lines 33-56)
 */
struct ZModelParams
{
    /// Atwood number \f$A=(\rho_2-\rho_1)/(\rho_2+\rho_1)\f$. Multiplies the
    /// Bernoulli forcing in the circulation equation. Dimensionless.
    /// CLI `--A`, default 0.3 (README configuration (a)).
    Real A = 0.3;

    /// Gravitational acceleration magnitude, pointing in \f$-\hat z\f$. Enters
    /// the Bernoulli potential as \f$-2 g z_3\f$. CLI `--g`, default 1.0.
    Real g = 1.0;

    /// Kernel desingularization parameter. Its *meaning* depends on
    /// `blob_mode`: with `Length` it is a smoothing length (denominator
    /// \f$(\epsilon^2+r^2)^{3/2}\f$), with `Matlab` it is already a squared
    /// length (denominator \f$(\epsilon+r^2)^{3/2}\f$). CLI `--eps`, default
    /// 0.025.
    Real eps = 0.025;

    /// Artificial viscosity coefficient on the circulation equation. Units of
    /// length^2/time. CLI `--mu`, default 0.002.
    Real mu = 0.002;

    /// Surface-tension coefficient \f$\sigma\f$ for the mean-curvature flow
    /// \f$\dot x \mathrel{+}= \sigma\,\Delta_{LB} x\f$. CLI `--sigma`,
    /// default 0.0 (off).
    Real sigma = 0.0;

    /// If > 0, localize surface tension to a ball of this radius about
    /// `sigma_center` with a smoothstep taper (full inside 0.6R, zero at R).
    /// CLI `--sigma-radius`, default 0.0 (global).
    Real sigma_radius = 0.0;

    /// Centre of the localization ball. CLI `--sigma-center`, default (0,0,0).
    std::array<Real, 3> sigma_center = { 0.0, 0.0, 0.0 };

    /// Kernel denominator convention. CLI `--kernel-blob-mode`, default
    /// `length`.
    KernelBlobMode blob_mode = KernelBlobMode::Length;

    /// Sign multiplying the Atwood/Bernoulli forcing. CLI `--forcing-sign`,
    /// default +1. Only -1 and +1 are accepted.
    Real forcing_sign = 1.0;

    /// Sign multiplying the Birkhoff-Rott velocity. CLI `--br-sign`,
    /// default +1. Only -1 and +1 are accepted.
    Real br_sign = 1.0;

    /// Surface quadrature for the BR source. CLI `--source-quadrature`,
    /// default `face`.
    SourceQuadrature source_quadrature = SourceQuadrature::Face;

    /// Marker-motion rule. CLI `--velocity-mode`, default `full`.
    VelocityMode velocity_mode = VelocityMode::Full;

    /// Apply the discrete mean-normal-flux projection that removes the net
    /// volume rate from the interface velocity. CLI `--no-preserve-volume`
    /// clears it; default true.
    bool preserve_volume = true;

    /// Far-field approximation. CLI `--br-approximation`. The Python default
    /// is `treecode`; Beatnik maps that to `Fmm`.
    BRApproximation br_approximation = BRApproximation::Fmm;

    /// Scalar entering the Bernoulli forcing.
    /// CLI `--bernoulli-scalar-mode`, default `normal-speed`.
    BernoulliScalarMode bernoulli_scalar_mode = BernoulliScalarMode::NormalSpeed;

    /// Operator used for the `mu` viscous term. CLI `--viscosity-mode`,
    /// default `laplace-beltrami`.
    ViscosityMode viscosity_mode = ViscosityMode::LaplaceBeltrami;

    /**
     * @brief The kernel denominator offset actually used by the BR kernel.
     *
     * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sources
     * (line 394)
     *
     * @return \f$\epsilon\f$ under `Matlab`, \f$\epsilon^2\f$ under `Length`.
     *         Has units of length squared in both cases.
     */
    Real blob() const
    {
        return ( blob_mode == KernelBlobMode::Matlab ) ? eps : eps * eps;
    }
};

//---------------------------------------------------------------------------//
/**
 * @brief Canopy fast-multipole tunables for the far-field BR evaluation.
 *
 * These have **no** Python counterpart as a group: the reference uses a
 * barnes-hut treecode (`--br-treecode-theta/-order/-ncrit`, lines 239-244)
 * which Beatnik replaces with Canopy's FMM. The three treecode knobs are still
 * accepted at the CLI (so a Python command line runs) and are mapped onto the
 * three members below that have a counterpart; everything else here is
 * FMM-only and has no CLI option. **The numbers do not transfer uniformly**,
 * and each of the three fails differently — the per-knob comments say which
 * way. `mac_theta` and `ncrit` denote the same quantity in both algorithms and
 * keep the reference's values; `order` denotes the same quantity but not the
 * same accuracy, and its default is raised from the reference's 2 to 3.
 *
 * Every member is default-initialized to a value the adapter may build a tree
 * from, which is what keeps a default-constructed `FmmParams` from producing an
 * arbitrary Canopy tree: Canopy's `FmmConfig::ncrit` and `FmmConfig::max_depth`
 * have no default initializer of their own.
 *
 * **There is deliberately no `softening` member.** Canopy's
 * `FmmConfig::softening` is a **length**; `ZModelParams::blob()` above is a
 * **squared length** — \f$\epsilon^2\f$ under `Length` and \f$\epsilon\f$ under
 * `Matlab`. The adapter therefore passes \f$\sqrt{\texttt{blob()}}\f$, which is
 * `eps` under `Length` and \f$\sqrt{\texttt{eps}}\f$ under `Matlab`. Deriving
 * it at that one call site is what makes both blob modes correct without a
 * second source of truth; a member here would be that second source. Getting
 * it wrong is a silent factor of \f$\epsilon\f$ in the softening length, and
 * under `FarFieldBasis::CartesianTaylor` that value enters the far field's own
 * \f$w=r^2+b\f$ rather than only the near-field sum, so the error is not
 * confined to close pairs. It must also be passed **explicitly positive**:
 * Canopy's default of \f$-1\f$ selects distribution-based auto-softening at
 * first setup, which is a different kernel rather than a fallback, and which
 * additionally disables `near_softening_factor`.
 */
struct FmmParams
{
    /// Far-field basis the adapter instantiates. Default
    /// `FarFieldBasis::CartesianTaylor`, the validated production path; see the
    /// enum in `Beatnik_Types.hpp` for what the alternative costs. No CLI
    /// option and no Python counterpart. Reaches no `FmmConfig` member —
    /// Canopy's basis is a template parameter on its solver, not a config
    /// field, which is why it must be named at every instantiation.
    FarFieldBasis basis = FarFieldBasis::CartesianTaylor;

    /// Multipole acceptance criterion (opening angle), dimensionless. Mapped
    /// from `--br-treecode-theta`, default 0.3. Reaches
    /// `FmmConfig::mac_theta` (whose own default is 0.5).
    ///
    /// **This knob transfers.** It is the same opening angle in both
    /// algorithms and the reference's value is kept unchanged. It is the
    /// expensive end of Canopy's range and deliberately so: at
    /// \f$\theta=0.3\f$ Canopy accepts cell pairs only beyond
    /// \f$R/w>2\sqrt3/\theta\approx11.55\f$ in half-widths, which is what lets
    /// order 3 reach \f$10^{-3}\f$ on the gradient at all. Raising it to
    /// Canopy's 0.5 would need order 4 for the same accuracy (35 coefficients
    /// per cell per component against 20).
    ///
    /// Two other members are read against this value rather than
    /// independently: `ncrit`'s liveness inequality and the realized M2L key
    /// count `max_depth` bounds both scale with \f$\theta\f$, so changing it
    /// re-opens both.
    Real mac_theta = 0.3;

    /// Basis order knob, dimensionless count. Mapped from
    /// `--br-treecode-order`, but **default 3, not the reference's 2**.
    /// Reaches no `FmmConfig` member: it is Canopy's `P_ORDER` template
    /// parameter, so the adapter dispatches on it at compile time.
    ///
    /// Under `FarFieldBasis::CartesianTaylor` this is the Cartesian Taylor
    /// truncation order \f$p\f$, costing \f$\binom{p+3}{3}\f$ coefficients per
    /// cell per component — 20 at \f$p=3\f$.
    ///
    /// **This knob denotes the same quantity in both algorithms and not the
    /// same accuracy**, which is the one place the example's "option names and
    /// defaults match the Python script exactly" promise genuinely breaks. The
    /// reference is a treecode with no target-side expansion (`treecode.py`
    /// evaluates its expansion batch at `rv = target - node.center`, lines
    /// 121-126), so its order-2 *velocity* carries the truncation order of an
    /// FMM's order-2 *potential*. An FMM's local expansion is differentiated to
    /// get the gradient, and \f$\nabla\f$ of a degree-\f$p\f$ Taylor local is
    /// degree \f$p-1\f$ — so an FMM's order-2 gradient is one order short of
    /// where the treecode's order-2 velocity sits, and Beatnik reads only the
    /// gradient. Order 3 here is the counterpart of the reference's order 2,
    /// not an upgrade of it.
    ///
    /// The default is traceable to a measurement rather than to that argument:
    /// on Canopy's volumetric cloud at \f$\theta=0.3\f$ the relative error on
    /// the gradient is \f$8.996\times10^{-3}\f$ at \f$p=2\f$ against
    /// \f$7.0718\times10^{-4}\f$ at \f$p=3\f$, so 3 is the smallest order that
    /// reaches Beatnik's \f$10^{-3}\f$ target and 2 misses it by an order.
    /// That cloud is not a sheet; the curve on Beatnik's own geometry is
    /// unmeasured and is what revises this value.
    ///
    /// `--br-treecode-order` still overrides, so a Python command line that
    /// passes 2 explicitly still gets 2 — and gets the accuracy above, which
    /// does not meet the target.
    int order = 3;

    /// Leaf occupancy target, particles per leaf cell. Mapped from
    /// `--br-treecode-ncrit`, default 64. Reaches `FmmConfig::ncrit`, which has
    /// no default initializer in Canopy.
    ///
    /// **This knob transfers** — it is the same leaf occupancy in both
    /// algorithms — but it is right only at production vertex counts, and the
    /// inequality rather than the conclusion is what belongs here so that it
    /// re-evaluates if `mac_theta` changes. Under Canopy's MAC the near field
    /// reaches \f$\sqrt3/\theta\f$ **cell widths**. Beatnik's sources are a
    /// 2-manifold, so the leaves the surface occupies form a two-dimensional
    /// grid and that neighbourhood covers \f$\pi(\sqrt3/\theta)^2\f$ of them —
    /// about 105 at \f$\theta=0.3\f$. A surface of \f$N\f$ vertices refines
    /// until a leaf holds `ncrit`, so a far field that carries any of the field
    /// at all needs
    ///
    /// \f[
    ///   N \;\gg\; \pi\,(\sqrt3/\theta)^2 \cdot \texttt{ncrit} ,
    /// \f]
    ///
    /// which is \f$N\gg6720\f$ at \f$\theta=0.3\f$ and this default. Below
    /// that the solve is a direct sum with FMM bookkeeping around it: it agrees
    /// with `BRSolverDirect` to round-off, at any `order`, silently and at full
    /// accuracy. Any accuracy measurement below the bound must lower `ncrit`
    /// to satisfy it and must report the realized near-field pair fraction
    /// rather than assume a far field exists.
    int ncrit = 64;

    /// Hard cap on tree depth, in levels. Default 10. Reaches
    /// `FmmConfig::max_depth`, which has no default initializer in Canopy and
    /// which Canopy bounds at 19 by its `uint64_t` Morton key. Sets the finest
    /// cell width as (root box width) / \f$2^{\texttt{max\_depth}}\f$. No CLI
    /// option and no counterpart in the treecode knob set.
    ///
    /// **This is a cap, not a target.** The tree stops subdividing at `ncrit`
    /// occupancy long before it, and on a well-behaved sheet 10 never binds: a
    /// 2-manifold at leaf occupancy `ncrit` reaches \f$N/\texttt{ncrit}\f$
    /// occupied leaves in \f$\log_4\f$ rather than \f$\log_8\f$ levels, so at
    /// `ncrit = 64` the sheet needs about 3 levels at 2562 vertices and about 7
    /// at a million.
    ///
    /// What it does bound is the case where occupancy does not fall off:
    /// a self-contacting roll-up drives the occupied-depth count up, and under
    /// `FarFieldBasis::CartesianTaylor` the M2L operator keys carry the tree
    /// level, so every occupied depth multiplies the realized key count
    /// against Canopy's 32768-key cap. That cap is close, not distant —
    /// Canopy measured 25438 keys, 78% of it, at \f$\theta=0.3\f$ on a
    /// volumetric cloud at `max_depth` 6, and saw the table saturate it
    /// outright on an amplified trajectory. Keys past the cap are not an error:
    /// they route to a per-pair translate that is slower and bitwise different
    /// from the table path, so an overflow yields an accuracy number that
    /// mixes two code paths. 10 is chosen to leave the sheet's own refinement
    /// unconstrained while keeping a roll-up from reaching depths whose only
    /// effect is key count.
    ///
    /// The number is reasoned, not measured, and the measurement is owed:
    /// lower it only on evidence of realized overflow at the production
    /// configuration, never to shrink the table pre-emptively. develop-canopy
    /// ran 19 and hit a depth-driven finite-difference blow-up at roll-up, but
    /// that mechanism was in a finite-difference L2P the analytic Taylor L2P
    /// removes, so it is not a reason to fear 19 here.
    int max_depth = 10;

    /// Near-field softening floor, as a multiple of the softening length.
    /// Default 0, which disables the floor. Reaches
    /// `FmmConfig::near_softening_factor` (whose own default is 4.0).
    /// No CLI option: `--br-near-factor` is a different quantity — the Python's
    /// local/clustered near-field radius — and is accepted and ignored.
    ///
    /// Canopy forces any pair closer than
    /// `near_softening_factor` \f$\times\f$ the softening length out of the
    /// far field and into the near-field sum. **That is meaningful only under
    /// `FarFieldBasis::SolidHarmonic`**, where the far field expands the bare
    /// \f$1/r\f$ and so is accurate only where the softening is negligible.
    /// Under `CartesianTaylor` the far field expands
    /// \f$(r^2+b)^{-1/2}\f$ and already carries the blob, so a non-zero floor
    /// moves work into the near-field sum and slows the solve without
    /// improving it. It also has no effect at all unless the softening is
    /// positive, which is a second reason the adapter must set the softening
    /// explicitly rather than leave Canopy's auto-softening sentinel in place.
    Real near_softening_factor = 0.0;

    /// Coarsening hysteresis band on `ncrit`, as a fraction of it. Default 0.1,
    /// matching Canopy's own. Reaches `FmmConfig::ncrit_tol`, which Canopy uses
    /// as `coarsen_threshold = ncrit * (1 - ncrit_tol)` — 57 at `ncrit = 64` —
    /// so a group of children merges only once its combined count falls below
    /// that, and a cell that was just split does not immediately re-merge when
    /// a few particles leave. Beatnik's surface deforms on every RK stage,
    /// which is precisely the thrashing the band exists to damp, so the default
    /// is kept rather than tightened.
    Real ncrit_tol = 0.1;

    /// Depth at and above which cells are replicated on every rank, in levels.
    /// Default 3. Reaches `FmmConfig::replication_depth` (whose own default is
    /// 1, below the 2-4 range Canopy's `TreePartitioner` documents as typical;
    /// that class's own constructor default is 3, and develop-canopy's
    /// integration ran 3).
    ///
    /// Cells at depth \f$\le\f$ this value are owned by all ranks and deeper
    /// cells get unique owners, so the value trades an allreduce over the
    /// coarse layers against the ownership bookkeeping below them. Canopy
    /// bounds the cost directly: at depth 3 there are at most 585 cells
    /// (1 + 8 + 64 + 512). Beatnik runs this path at 1-6 ranks, where Canopy's
    /// default of 1 would replicate only 9 cells — fewer coarse cells than
    /// ranks at the top of the range. This is ownership, not occupancy, and
    /// does not interact with `max_depth`'s key-count bound.
    int replication_depth = 3;

    /// Zoltan2 partition imbalance tolerance, as a fraction. Default 0.10,
    /// meaning a 10% load imbalance across ranks is accepted. Reaches
    /// `FmmConfig::imbalance_tolerance` (whose own default is 0.05; Canopy
    /// passes it on as \f$1+\f$ this value).
    ///
    /// Looser than Canopy's default on purpose: a deforming surface picks
    /// Canopy's `Rebalance` maintenance path essentially every RK stage, so the
    /// partitioner runs at that frequency and a tighter tolerance buys balance
    /// at a cost paid every stage. At the 1-6 ranks this path runs at, 10% is a
    /// small absolute imbalance. develop-canopy's integration ran 0.10 for the
    /// same reason.
    Real imbalance_tolerance = 0.10;

    /// Per-face padding applied to the global root bounding box, each as a
    /// fraction of that axis's width. Default 0.10 on all six faces. They reach
    /// `FmmConfig::xmin_tol` … `zmax_tol` in that order (whose own defaults are
    /// all 0.0, i.e. a box that hugs the particles exactly); develop-canopy's
    /// integration ran 0.10 uniformly.
    ///
    /// Padding gives particles room to move before they leave the box, which is
    /// what keeps Canopy's cheaper maintenance paths valid for a stage or two
    /// instead of forcing the heavy one the moment the surface expands. Uniform
    /// rather than asymmetric because the bubble is not confined on any face;
    /// asymmetric values are permitted by Canopy and nothing here needs them.
    ///
    /// **Padding does not stabilize the box**, and raising it will not make it
    /// do so. Canopy recomputes the root box from the particles on every
    /// maintenance path, `migrate` included, and a padded box is a fraction of
    /// a moving box, so it drifts with it — measured at a smooth 0.18-0.40% per
    /// build, which under `FarFieldBasis::CartesianTaylor` clears the entire
    /// M2L operator cache every time (zero keys retained across 336 builds).
    /// There is no `FmmParams` value that changes that; it is a Canopy-side
    /// property.
    ///
    /// The cost of padding is that the root box is 20% wider per axis at 0.10,
    /// so every cell at a given depth is 20% wider — a shift in where `ncrit`
    /// occupancy is reached, not a change in the number of occupied depths.
    Real xmin_tol = 0.10;
    Real xmax_tol = 0.10; ///< See `xmin_tol`.
    Real ymin_tol = 0.10; ///< See `xmin_tol`.
    Real ymax_tol = 0.10; ///< See `xmin_tol`.
    Real zmin_tol = 0.10; ///< See `xmin_tol`.
    Real zmax_tol = 0.10; ///< See `xmin_tol`.

    /// Per-rank memory budget for Canopy's hashed M2L operator table, in
    /// **bytes**. Default 2 GiB, matching Canopy's own. Reaches
    /// `FmmConfig::m2l_op_table_byte_budget`.
    ///
    /// Canopy bounds the table by the smaller of this budget's worth of
    /// operator columns and its own 32768-key count cap, so the default is
    /// chosen to be the **non-binding** half of that minimum: at `order`
    /// \f$\le4\f$ a column costs at most about 9.8 KB, so the full 32768 keys
    /// occupy roughly 0.3 GiB and the count cap binds first at every order
    /// this path supports. Lowering this is the only way to make the byte
    /// budget bind instead, and that is the wrong lever — the constraint to act
    /// on is the count cap, and the response to realized overflow is a lower
    /// `max_depth` or `order`, not a smaller table.
    std::size_t m2l_op_table_byte_budget = 2ull * 1024ull * 1024ull * 1024ull;
};

//---------------------------------------------------------------------------//
/**
 * @brief Indicator-driven red-green AMR controls.
 *
 * Used only when `--no-dynamic-remesh` is in effect; the default path is the
 * metric-based dynamic remesher (`RemeshParams`).
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 263-302) and
 * mesh_solver.py::refine_potential_mesh_state (lines 1374-1431)
 */
struct AmrParams
{
    /// Refine a face whose relative area change since its reference area
    /// exceeds this. Dimensionless. CLI `--area-threshold`, default 0.16.
    Real area_change_threshold = 0.16;

    /// Refine a face whose relative curvature-indicator change since its
    /// reference exceeds this. Dimensionless.
    /// CLI `--curvature-change-threshold`, default 0.35.
    Real curvature_change_threshold = 0.35;

    /// Refine a face whose flat-triangle sagitta error \f$\kappa h^2/8\f$
    /// exceeds this. Units of length. 0 disables the criterion.
    /// CLI `--curvature-resolution-threshold`, default 0.0.
    Real curvature_resolution_threshold = 0.0;

    /// Hard cap on the projected post-refinement face count.
    /// CLI `--max-faces`, default 1400.
    int max_faces = 1400;

    /// Cap on the fraction of faces that may be seed-marked in one pass.
    /// CLI `--max-refine-fraction`, default 0.05.
    Real max_refine_fraction = 0.05;

    /// Grow the marked set by this many face-neighbor rings before closure.
    /// CLI `--refine-neighbor-rings`, default 1.
    int refine_neighbor_rings = 1;

    /// Promote poor green transition faces to full red refinement.
    /// CLI `--no-balance-refinement` clears it; default true.
    bool balance_refinement = true;

    /// Absolute quality floor below which a one-edge green split is promoted
    /// to red. CLI `--transition-quality-floor`, default 0.18.
    Real transition_quality_floor = 0.18;

    /// Relative quality floor, as a fraction of the parent face quality.
    /// CLI `--transition-quality-fraction`, default 0.45.
    Real transition_quality_fraction = 0.45;

    /// Never refine a face whose shortest edge is already below this length.
    /// CLI `--min-refine-edge`, default 0.0 (no floor).
    Real min_refine_edge = 0.0;

    /// Refine every this many accepted steps. CLI `--refine-every`, default 5.
    int refine_every = 5;
};

//---------------------------------------------------------------------------//
/**
 * @brief The `--remesh-flip-min-gain` at or above which the reference's quality
 *        flip pass accepts **nothing**, i.e. flips are configured off through
 *        the reference's own knob rather than through a Beatnik-only switch.
 *
 * `dynamic_remesh.py:449-450` accepts a flip only when
 * \f$\min(q_{\text{new}}) > \min(q_{\text{old}})(1+g)\f$; every candidate is
 * `continue`d otherwise. Triangle quality is \f$4\sqrt3 A/\sum \ell^2\f$ and
 * therefore lies in \f$[0,1]\f$, so a gain of \f$10^{12}\f$ makes the accept
 * test unsatisfiable for every pair of triangles that can exist — the pass
 * still runs and still mutates nothing.
 *
 * **T4b uses this as the acceptance criterion for `--dynamic-remesh`**: the
 * flip third is `DynamicRemesh::flipEdgesForQuality`, which is T4d and blocked
 * on Tessera gap G5c, so a run is accepted only when the reference itself would
 * flip nothing. See `Solver::requireSupportedConfiguration`.
 */
constexpr Real kFlipsDisabledMinGain = 1.0e12;

//---------------------------------------------------------------------------//
/**
 * @brief Metric-based dynamic remeshing controls (the default adaptivity path).
 *
 * Port of dynamic_remesh.py::DynamicRemeshParams (lines 17-46)
 *
 * The driver builds two of these: a baseline set from `--remesh-*` and, when
 * `--remesh-tight-after >= 0`, a tighter set from `--remesh-tight-*` that
 * takes over past that simulation time
 * (run_adaptive_mesh_bubble.py:1358-1396).
 */
struct RemeshParams
{
    /// Target chord ("sagitta") error of a flat triangle against the curved
    /// surface it represents. Sets the curvature sizing field via
    /// \f$h = \sqrt{8\,\text{tol}/\kappa}\f$. Units of length.
    /// CLI `--remesh-sagitta-tolerance`, default 0.004.
    Real sagitta_tolerance = 0.004;

    /// Lower clamp on the target edge length. CLI `--remesh-h-min`,
    /// default 0.0015.
    Real h_min = 0.0015;

    /// Upper clamp on the target edge length. CLI `--remesh-h-max`,
    /// default 0.05.
    Real h_max = 0.05;

    /// Split an edge longer than `split_factor * target`.
    /// CLI `--remesh-split-factor`, default 1.35.
    Real split_factor = 1.35;

    /// Collapse an edge shorter than `collapse_factor * target`.
    /// CLI `--remesh-collapse-factor`, default 0.45.
    Real collapse_factor = 0.45;

    /// Trigger the flip/smooth repair pass when the worst triangle quality
    /// drops below this. CLI `--remesh-min-quality`, default 0.18.
    Real min_quality = 0.18;

    /// Relative quality gain required to accept a flip.
    /// CLI `--remesh-flip-min-gain`, default 1e-3.
    Real flip_min_gain = 1.0e-3;

    /// Tangential smoothing sweeps per pass.
    /// CLI `--remesh-smooth-iters`, default 1.
    int smoothing_iterations = 1;

    /// Tangential smoothing relaxation factor.
    /// CLI `--remesh-smooth-relaxation`, default 0.04.
    Real smoothing_relaxation = 0.04;

    /// Split/collapse/flip/smooth passes per remesh call.
    /// CLI `--remesh-passes`, default 1.
    int passes = 1;

    /// Cap on splits per pass; <= 0 means unlimited.
    /// CLI `--remesh-max-splits`, default 300.
    int max_splits_per_pass = 300;

    /// Cap on collapses per pass; <= 0 means unlimited.
    /// CLI `--remesh-max-collapses`, default 120.
    int max_collapses_per_pass = 120;

    // --- nonlocal proximity sizing -----------------------------------------

    /// Include the nonlocal-proximity term in the sizing field.
    /// CLI `--remesh-proximity`, default false.
    bool use_proximity = false;

    /// Target edge length as this fraction of the nonlocal gap.
    /// CLI `--remesh-proximity-fraction`, default 0.25.
    Real proximity_fraction = 0.25;

    /// Only apply proximity sizing where the gap is below this distance.
    /// **Resolved by `Solver::setup`**, not by the driver: it is either this
    /// absolute value, or `proximity_activation_factor * initial_min_edge` when
    /// this is <= 0 (run_adaptive_mesh_bubble.py:1272-1276), and
    /// `initial_min_edge` does not exist until the mesh does.
    /// CLI `--remesh-proximity-activation-distance`, default 0.0.
    Real proximity_activation_distance = 0.0;

    /// Multiplier on \f$h^0_{\min}\f$ used for
    /// `proximity_activation_distance` when the absolute value above is <= 0.
    /// CLI `--remesh-proximity-activation-factor`, default 6.0.
    ///
    /// **T1c CHANGE — this lived in the example's `ClArgs` and had to move
    /// here.** `Solver::setup`'s documented step 3 is the resolution against
    /// `initial_min_edge`, and the solver is handed a `SolverParams`; a factor
    /// held only in the driver's own struct was therefore unreachable at the
    /// one place able to use it, so the resolution could not have been written
    /// at all. The CLI option name and default are unchanged.
    Real proximity_activation_factor = 6.0;

    /// Same-surface face rings excluded from the proximity search, so a smooth
    /// coarse surface does not refine against its own neighbors.
    /// CLI `--remesh-proximity-exclusion-rings`, default 3.
    int proximity_exclusion_rings = 3;

    /// Faces closer than this in *carried material coordinates* are treated as
    /// the same piece of sheet and excluded. **Resolved by `Solver::setup`**
    /// from this absolute value, or from
    /// `proximity_material_exclusion_factor * initial_min_edge` when this is
    /// <= 0 (run_adaptive_mesh_bubble.py:1277-1286).
    /// CLI `--remesh-proximity-material-exclusion-radius`, default 0.0.
    Real proximity_material_exclusion_radius = 0.0;

    /// Multiplier on \f$h^0_{\min}\f$ used for
    /// `proximity_material_exclusion_radius` when the absolute value above is
    /// <= 0. CLI `--remesh-proximity-material-exclusion-factor`, default 4.0.
    /// Moved here from the example's `ClArgs` at T1c, for the reason recorded
    /// on `proximity_activation_factor`.
    Real proximity_material_exclusion_factor = 4.0;

    /// Skip proximity sizing above this face count; <= 0 at the CLI maps to
    /// effectively unlimited. CLI `--remesh-proximity-max-faces`,
    /// default 100000.
    long long proximity_max_faces = 100000;

    // --- sizing-field gradation --------------------------------------------

    /// Cap the ratio of adjacent vertex target sizes at this factor, to avoid
    /// refinement cascades. CLI `--remesh-target-gradation-factor`,
    /// default 1.35.
    Real target_gradation_factor = 1.35;

    /// Gradation sweeps. CLI `--remesh-target-gradation-iters`, default 8.
    int target_gradation_iterations = 8;

    // --- surgical proximity splits -----------------------------------------

    /// Directly split faces in exact nonlocal close-pair regions before the
    /// ordinary remesh. CLI `--remesh-surgical-proximity`, default false.
    bool surgical_proximity = false;

    /// Target length as this fraction of the exact pair gap.
    /// CLI `--remesh-surgical-proximity-fraction`, default 0.35.
    Real surgical_proximity_fraction = 0.35;

    /// Floor for surgical splits; <= 0 reuses `h_min`.
    /// CLI `--remesh-surgical-proximity-h-min`, default 0.0.
    Real surgical_proximity_h_min = 0.0;

    /// Activation gap for surgical splits; <= 0 reuses
    /// `proximity_activation_distance`.
    /// CLI `--remesh-surgical-proximity-activation-distance`, default 0.0.
    Real surgical_proximity_activation_distance = 0.0;

    /// Cap on close pairs handled per call.
    /// CLI `--remesh-surgical-proximity-max-pairs`, default 64.
    int surgical_proximity_max_pairs = 64;

    /// k for the nearest-neighbor query that seeds the pair search.
    /// CLI `--remesh-surgical-proximity-query-k`, default 48.
    int surgical_proximity_query_k = 48;
};

//---------------------------------------------------------------------------//
/**
 * @brief Valence-equalizing sliver cleanup applied after each remesh.
 *
 * Port of mesh_quality.py::isotropic_cleanup (lines 146-167)
 */
struct CleanupParams
{
    /// Run the cleanup pass. CLI `--isotropic-cleanup` /
    /// `--no-isotropic-cleanup`, default true.
    bool enabled = true;

    /// Valence-equalizing flip passes. CLI `--isotropic-cleanup-flips`,
    /// default 3.
    int flip_passes = 3;

    /// Tangential relaxation passes. CLI `--isotropic-cleanup-relax`,
    /// default 2.
    int relax_passes = 2;

    /// Relaxation weight. CLI `--isotropic-cleanup-weight`, default 0.4.
    Real relax_weight = 0.4;
};

//---------------------------------------------------------------------------//
/**
 * @brief Time-stepping and adaptive-dt controls.
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 133-157) and
 * ::choose_step_dt (lines 889-901)
 */
struct TimeParams
{
    /// Maximum number of steps taken by this invocation (local, not global —
    /// a restart adds to the loaded step counter). CLI `--steps`, default 140.
    int steps = 140;

    /// Stop once the simulation time reaches this. Negative means "unset",
    /// matching the Python `None`. CLI `--t-end`, default None.
    Real t_end = -1.0;

    /// Whether `t_end` was supplied.
    bool have_t_end = false;

    /// Nominal step size. CLI `--dt`, default 0.003.
    Real dt = 0.003;

    /// Past this simulation time, clamp dt to `dt_after_switch`. Negative
    /// disables. CLI `--dt-switch-time`, default -1.0.
    Real dt_switch_time = -1.0;

    /// Clamp value used past `dt_switch_time`. CLI `--dt-after-switch`,
    /// default 0.001.
    Real dt_after_switch = 0.001;

    /// Throttle dt by the smallest triangle. CLI `--adaptive-dt` /
    /// `--no-adaptive-dt`, default true.
    bool adaptive_dt = true;

    /// Floor for the adaptive dt. CLI `--min-dt`, default 2.5e-4.
    Real min_dt = 2.5e-4;

    /// Exponent on the edge-length ratio in the adaptive dt scaling.
    /// CLI `--dt-edge-power`, default 1.0.
    Real dt_edge_power = 1.0;

    /// When > 0, additionally require `dt * max|sheet_vector|` below this.
    /// CLI `--max-sheet-dt-product`, default 0.0 (off).
    Real max_sheet_dt_product = 0.0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Checkpoint / restart controls.
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 309-331)
 */
struct CheckpointParams
{
    /// Output directory; empty disables checkpointing entirely.
    /// CLI `--checkpoint-dir`, default "".
    std::string directory;

    /// Filename prefix. CLI `--checkpoint-prefix`, default "checkpoint".
    std::string prefix = "checkpoint";

    /// Save whenever this much simulation time has elapsed; 0 disables.
    /// CLI `--checkpoint-every-time`, default 0.0.
    Real every_time = 0.0;

    /// Save every this many accepted steps; 0 disables.
    /// CLI `--checkpoint-every-steps`, default 0.
    int every_steps = 0;

    /// Path to a checkpoint to restart from; empty means build the initial
    /// surface instead. CLI `--restart-from`, default "".
    std::string restart_from;

    /// True when a restart path was supplied.
    bool restarting() const { return !restart_from.empty(); }

    /// True when checkpoint output is enabled.
    bool writing() const { return !directory.empty(); }
};

//---------------------------------------------------------------------------//
/**
 * @brief Initial-surface geometry and initial vorticity seeding.
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 66-132) and
 * ::apply_initial_geometry (lines 713-886)
 */
struct InitialConditionParams
{
    // --- base sphere -------------------------------------------------------

    /// Latitude bands for the `latlon` mesh. CLI `--n-theta`, default 7.
    int n_theta = 7;

    /// Longitude divisions for the `latlon` mesh. CLI `--n-phi`, default 14.
    int n_phi = 14;

    /// Base mesh generator. CLI `--mesh-kind`, default `icosphere`.
    MeshKind mesh_kind = MeshKind::Icosphere;

    /// Icosahedron subdivision level. Vertex count is
    /// \f$10\cdot 4^{s}+2\f$, face count \f$20\cdot 4^{s}\f$; the default
    /// s = 2 gives 162 vertices and 320 faces.
    /// CLI `--icosphere-subdivisions`, default 2.
    int icosphere_subdivisions = 2;

    /// Sphere radius. CLI `--radius`, default 0.25.
    Real radius = 0.25;

    /// Sphere centre height; the centre is (0, 0, center_z).
    /// CLI `--center-z`, default 0.25.
    Real center_z = 0.25;

    // --- shape deformation -------------------------------------------------

    /// CLI `--initial-shape`, default `sphere`.
    InitialShape shape = InitialShape::Sphere;

    /// Radial stretch applied to non-sphere shapes.
    /// CLI `--horizontal-scale`, default 1.28.
    Real horizontal_scale = 1.28;

    /// Vertical stretch applied to non-sphere shapes.
    /// CLI `--vertical-scale`, default 0.68.
    Real vertical_scale = 0.68;

    /// Gaussian rim bulge amplitude for `mushroom-seed`.
    /// CLI `--rim-amp`, default 0.14.
    Real rim_amp = 0.14;
    /// Rim bulge centre in z/radius. CLI `--rim-center`, default 0.05.
    Real rim_center = 0.05;
    /// Rim bulge width in z/radius. CLI `--rim-width`, default 0.32.
    Real rim_width = 0.32;

    /// Skirt bulge amplitude for `skirt-seed`. CLI `--skirt-amp`, default 0.42.
    Real skirt_amp = 0.42;
    /// Skirt centre in z/radius. CLI `--skirt-center`, default -0.42.
    Real skirt_center = -0.42;
    /// Skirt width in z/radius. CLI `--skirt-width`, default 0.16.
    Real skirt_width = 0.16;
    /// Neck (negative) amplitude. CLI `--skirt-neck-amp`, default 0.16.
    Real skirt_neck_amp = 0.16;
    /// Neck centre in z/radius. CLI `--skirt-neck-center`, default -0.04.
    Real skirt_neck_center = -0.04;
    /// Neck width in z/radius. CLI `--skirt-neck-width`, default 0.24.
    Real skirt_neck_width = 0.24;
    /// Downward lip displacement as a fraction of the undeformed radius.
    /// CLI `--skirt-drop`, default 0.11.
    Real skirt_drop = 0.11;

    /// Azimuthal ripple mode number m. CLI `--azimuthal-mode`, default 4.
    int azimuthal_mode = 4;
    /// Azimuthal ripple amplitude. CLI `--azimuthal-amp`, default 0.035.
    Real azimuthal_amp = 0.035;

    /// Legendre mode l for the axisymmetric radial perturbation
    /// \f$r \to r(1+a P_l(\cos\theta))\f$. CLI `--polar-mode`, default 0.
    int polar_mode = 0;
    /// Amplitude a of that perturbation. CLI `--polar-amp`, default 0.0.
    Real polar_amp = 0.0;

    // --- initial vorticity -------------------------------------------------

    /// Amplitude of the seeded potential / sheet vorticity. 0 leaves the
    /// surface quiescent. CLI `--initial-potential-strength`, default 0.0.
    Real initial_potential_strength = 0.0;

    /// Spatial profile used with the strength above.
    /// CLI `--initial-vorticity-mode`, default `vertical`.
    InitialVorticityMode vorticity_mode = InitialVorticityMode::Vertical;

    /// Reference z/radius for the localized rim modes.
    /// CLI `--initial-vorticity-center`, default -0.15.
    Real vorticity_center = -0.15;

    /// Reference vertical width for the localized rim modes.
    /// CLI `--initial-vorticity-width`, default 0.18.
    Real vorticity_width = 0.18;

    /// Outer-radius localization power for the lip modes.
    /// CLI `--initial-vorticity-radial-power`, default 2.0.
    Real vorticity_radial_power = 2.0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Post-step field filtering and mesh redistribution.
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 387-405) and
 * ::filter_circulation_field (lines 923-948)
 */
struct FilterParams
{
    /// Tangential relaxation sweeps used by the redistribute pass and by the
    /// post-refinement quality repair. CLI `--smooth-iters`, default 1.
    int smooth_iters = 1;

    /// Relaxation factor for those sweeps. CLI `--smooth-relaxation`,
    /// default 0.12.
    Real smooth_relaxation = 0.12;

    /// Run a tangential redistribution every this many steps; 0 disables.
    /// CLI `--redistribute-every`, default 0.
    int redistribute_every = 0;

    /// Only filter the circulation field past this simulation time; negative
    /// means always. CLI `--field-filter-after`, default -1.0.
    Real field_filter_after = -1.0;

    /// Filter every this many steps; 0 disables. CLI `--field-filter-every`,
    /// default 0.
    int field_filter_every = 0;

    /// Graph-Laplacian smoothing iterations. CLI `--field-filter-iters`,
    /// default 1.
    int field_filter_iters = 1;

    /// Graph-Laplacian relaxation. CLI `--field-filter-relaxation`,
    /// default 0.01.
    Real field_filter_relaxation = 0.01;

    /// Only filter when max|sheet_vector| exceeds this; 0 means always.
    /// CLI `--field-filter-threshold`, default 0.0.
    Real field_filter_threshold = 0.0;

    /// Quality-flip passes after an indicator-driven refinement.
    /// CLI `--flip-passes`, default 0.
    int flip_passes = 0;
};

} // namespace Beatnik

#endif // BEATNIK_PARAMS_HPP
