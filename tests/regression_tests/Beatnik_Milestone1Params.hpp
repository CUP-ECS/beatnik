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
 * @file Beatnik_Milestone1Params.hpp
 * @brief The milestone-1 configuration — the reference's default rising-bubble
 *        run, M1-G2's command line — as a `SolverParams`.
 *
 * Shared by the measurement driver `Beatnik_Test_Milestone1Run.cpp` and the
 * milestone comparison test (`tasks/milestone1.md`, M1-T2j), which include it
 * with a quoted `#include`: the generated per-backend translation unit
 * includes each test source by absolute path, so the quoted include resolves
 * next to the source.
 *
 * The command, recorded verbatim with the gold set in
 * `tests/regression_tests/milestone1-bubble/README.md`:
 *
 *     python examples/run_adaptive_mesh_bubble.py \
 *       --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
 *       --viscosity-mode laplace-beltrami --br-approximation direct \
 *       --adaptive-dt --dynamic-remesh --isotropic-cleanup \
 *       --source-quadrature vertex \
 *       --steps 600 --checkpoint-every-steps 10 --no-video --checkpoint-dir results
 *
 * Every other option is at its `parse_args` default. **No knob is turned
 * off**: collapse, quality flips, guarded smoothing and the isotropic cleanup
 * all run at the reference's defaults. Every value is set EXPLICITLY rather
 * than inherited from a Beatnik default, so a later change to a default breaks
 * the milestone loudly instead of silently changing what it runs; the comment
 * beside each one names the reference default it reproduces
 * (`run_adaptive_mesh_bubble.py` at the version the gold README records).
 *
 * The one Beatnik-only knob, `RemeshParams::renumber_gids_every`, has no
 * reference counterpart and is left at its `RemeshParams` default, which is
 * what a milestone run uses; the measurement driver overrides it from its
 * third positional.
 */

#ifndef BEATNIK_MILESTONE1PARAMS_HPP
#define BEATNIK_MILESTONE1PARAMS_HPP

#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include <string>

namespace Beatnik
{
namespace Test
{
namespace Milestone1
{

/// `--steps 600`.
constexpr int kSteps = 600;
/// `--checkpoint-every-steps 10`: 61 files, step 0 included.
constexpr int kCheckpointEvery = 10;

//---------------------------------------------------------------------------//
/**
 * @brief M1-G2's command line, field for field.
 *
 * @param checkpoint_dir Where `Solver` writes its checkpoints; empty disables
 *        them by `writeCheckpoint()`'s own guard.
 */
inline SolverParams makeParams( const std::string& checkpoint_dir )
{
    SolverParams p;

    // --state-model potential (default, :385).
    p.state_model = StateModel::Potential;
    // --mesh-kind icosphere (:68), --icosphere-subdivisions 2 (:69),
    // --radius 0.25 (:70), --center-z 0.25 (:71).
    p.initial.mesh_kind = MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = 2;
    p.initial.radius = 0.25;
    p.initial.center_z = 0.25;
    // --initial-shape sphere (:72-76), --initial-potential-strength 0 (:108),
    // --polar-amp 0 (:102-107).
    p.initial.shape = InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 (explicit on the command line),
    // --sigma 0 (default).
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
    p.zmodel.sigma = 0.0;
    // --forcing-sign 1 --br-sign 1 --kernel-blob-mode length (defaults).
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = KernelBlobMode::Length;
    // --viscosity-mode laplace-beltrami (explicit), --velocity-mode full and
    // --bernoulli-scalar-mode normal-speed (defaults), volume preservation on
    // (default).
    p.zmodel.viscosity_mode = ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    // --br-approximation direct --source-quadrature vertex (explicit).
    p.zmodel.br_approximation = BRApproximation::Direct;
    p.zmodel.source_quadrature = SourceQuadrature::Vertex;

    // --steps 600 (explicit), --adaptive-dt (explicit; default true, :143-148),
    // --dt 0.003 (:135), --min-dt 2.5e-4 (:150), --dt-edge-power 1.0 (:151),
    // --max-sheet-dt-product 0 (:152-157), --dt-switch-time -1 (:136-141),
    // --t-end None (:134).
    p.time.steps = kSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --dynamic-remesh (explicit; default true), --remesh-every 1,
    // --remesh-tight-after -1 (defaults).
    p.dynamic_remesh = true;
    p.remesh_every = 1;
    p.remesh_tight_after = -1.0;

    // The remesher at the reference's defaults (:420-431, :494-495). Nothing
    // is configured off: this is the configuration the milestone is about.
    p.remesh.passes = 1;
    p.remesh.sagitta_tolerance = 0.004;
    p.remesh.h_min = 0.0015;
    p.remesh.h_max = 0.05;
    p.remesh.split_factor = 1.35;
    p.remesh.collapse_factor = 0.45;
    p.remesh.min_quality = 0.18;
    p.remesh.flip_min_gain = 1.0e-3;
    p.remesh.smoothing_iterations = 1;
    p.remesh.smoothing_relaxation = 0.04;
    p.remesh.max_splits_per_pass = 300;
    p.remesh.max_collapses_per_pass = 120;
    p.remesh.target_gradation_factor = 1.35;
    p.remesh.target_gradation_iterations = 8;
    // Both proximity paths are off by default in the reference
    // (dynamic_remesh.py:33,41); spelled out because they are T4e.
    p.remesh.use_proximity = false;
    p.remesh.surgical_proximity = false;

    // --isotropic-cleanup (explicit; default true), --isotropic-cleanup-flips
    // 3, --isotropic-cleanup-relax 2, --isotropic-cleanup-weight 0.4
    // (:511-513).
    p.cleanup.enabled = true;
    p.cleanup.flip_passes = 3;
    p.cleanup.relax_passes = 2;
    p.cleanup.relax_weight = 0.4;

    // The refine-branch knobs, at their defaults: --refine-every 5 (:302),
    // --flip-passes 0 (:405), --smooth-iters 1 (:387), --smooth-relaxation
    // 0.12 (:388). Inert under --dynamic-remesh, which makes the two
    // adaptivity branches mutually exclusive (:1424 versus :1469).
    p.amr.refine_every = 5;
    p.filter.flip_passes = 0;
    p.filter.smooth_iters = 1;
    p.filter.smooth_relaxation = 0.12;
    // --redistribute-every 0 (:389), --field-filter-every 0 (:396).
    p.filter.redistribute_every = 0;
    p.filter.field_filter_every = 0;

    // --checkpoint-every-steps 10 (explicit). `setup()` writes step 0
    // unconditionally, so 600 steps every 10 give the gold set's 61 files.
    p.checkpoint.every_steps = kCheckpointEvery;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

    return p;
}

} // namespace Milestone1
} // namespace Test
} // namespace Beatnik

#endif // BEATNIK_MILESTONE1PARAMS_HPP
