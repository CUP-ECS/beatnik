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
 * These have **no** Python counterpart: the reference uses a barnes-hut
 * treecode (`--br-treecode-theta/-order/-ncrit`, lines 239-244) which Beatnik
 * replaces with Canopy's FMM. The Python treecode knobs are still accepted at
 * the CLI (so a Python command line runs) and are mapped here where a
 * counterpart exists.
 */
struct FmmParams
{
    /// Multipole acceptance criterion (opening angle). Mapped from
    /// `--br-treecode-theta`, default 0.3.
    Real mac_theta = 0.3;

    /// Expansion order. Mapped from `--br-treecode-order`, default 2.
    int order = 2;

    /// Leaf occupancy target. Mapped from `--br-treecode-ncrit`, default 64.
    int ncrit = 64;
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
