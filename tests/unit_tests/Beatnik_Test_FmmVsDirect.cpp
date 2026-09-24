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
 * @file Beatnik_Test_FmmVsDirect.cpp
 * @brief `unit`-tier test for T4: the FMM Birkhoff-Rott velocity against the
 *        direct one, **on the same state**, at ranks 1-6, with three negative
 *        cases.
 *
 * THIS IS T4's EXIT CRITERION, MECHANIZED (`tasks/canopy/add-canopy.md`). It is
 * a `unit`-tier member because the conventions table says new correctness tests
 * are `unit` unless a task says otherwise: the ship gate stays at five
 * `regression` members and 60 launches.
 *
 * THE RANK SWEEP DOES NOT COME FROM THIS FILE
 * -------------------------------------------
 * The tier registers every member at exactly one rank
 * (`tests/unit_tests/CMakeLists.txt`, `set(_beatnik_unit_ranks 1)`), and in
 * spack mode there is no build tree and therefore no `ctest` at all. The 1-6
 * sweep is `scripts/tuolumne/t4_fmm_vs_direct.flux`, which allocates two nodes
 * and loops this one binary over the six rank counts. The tier runner
 * (`scripts/tuolumne/unit_tests.flux`) picks this member up for free at one
 * rank because it discovers its tests rather than naming them.
 *
 * WHY 1-6 AND NOT 1 AND 4
 * -----------------------
 * Two independent reasons, both in the design document's risk list. **R3** —
 * a tag mismatch in the adapter's round trip does not crash; it produces a
 * velocity that is wrong by a factor that changes with the rank count, and a
 * sweep is the only thing that sees that. **R4** — Canopy's `SingleSolve`
 * three-component gradient bodies are known wrong at **exactly 4 ranks** under
 * `LaplaceKernel` (`canopy/README.md:562-588`). If this test fails at 4 ranks
 * and nowhere else, that is a decomposition-bug signature and the finding:
 * **do not widen the budget and do not drop 4 from the sweep.**
 *
 * THE TRAP THIS TEST IS SHAPED AROUND: A PASS THAT MEASURES NOTHING
 * ----------------------------------------------------------------
 * At \f$\theta=0.3\f$ Canopy's near field reaches \f$\sqrt3/\theta\f$ cell
 * widths. Beatnik's sources are a 2-manifold, so the occupied leaves form a
 * two-dimensional grid and that neighbourhood covers
 * \f$\pi(\sqrt3/\theta)^2\approx105\f$ of them. A solve whose occupied-leaf
 * count is not far above 105 is **entirely P2P**, agrees with `BRSolverDirect`
 * to round-off at every order, and reads as a success — risk **R1**'s cheapest
 * misreading, and exactly where T3's level-3 run at the default `ncrit` sits.
 * The liveness inequality is \f$N \gg \pi(\sqrt3/\theta)^2\,\texttt{ncrit}\f$;
 * at the default `ncrit = 64` that is \f$N\gg6720\f$, which neither
 * milestone-0 level meets.
 *
 * So this test pairs **subdivision level 4 (2562 vertices) with `ncrit = 8`** —
 * 320 occupied leaves against a 105-leaf near field, a factor of 3 — and then
 * **asserts** the far field is live rather than assuming it, through the
 * adapter's measured `p2p_pair_fraction`. No accuracy assertion in this file is
 * reached before that bound is checked.
 *
 * WHAT IS ASSERTED, GROUP BY GROUP
 * --------------------------------
 * 0. **The state is real.** Five direct-solver steps from the milestone-0
 *    initial condition, because `--initial-potential-strength 0` makes the
 *    sheet strength identically zero at step 0 and a step-0 comparison is
 *    vacuous. The global max sheet strength is asserted non-zero before
 *    anything is compared.
 * 1. **The round trip.** `diagnostics().global_particle_count` equals the
 *    global owned vertex count — the cheap independent discriminator for a
 *    dropped or duplicated source (**R3**) — and the FMM velocity is finite on
 *    every owned row.
 * 2. **The far field is live.** `p2p_pair_fraction` below `kP2PFractionBound`
 *    and `global_m2l_pair_count` strictly positive.
 * 3. **The adapter's softening and near-field floor**, which is the half of
 *    T1's exit criterion that first becomes runnable here:
 *    `diagnostics().softening == sqrt(blob())` and strictly positive (Canopy's
 *    \f$-1\f$ is an auto-softening *sentinel* and a different kernel, not an
 *    error), and `params().near_softening_factor == 0`. The `FmmConfig` members
 *    themselves are not observable from a test by construction — `FmmConfig` is
 *    a Canopy type and the conventions table confines those to the adapter — so
 *    the assertion is on the adapter's inputs and on what it reports, and an
 *    evaluation that completes at all is what shows the config it built was
 *    accepted.
 * 4. **The accuracy claim.** Max absolute and max relative velocity error
 *    against `kVelocityBudget`, whose derivation and full qualification list
 *    are on its declaration.
 * 5. **Negative case 1 — the order knob is live.** `order = 0` exceeds the
 *    budget, `order = 2` lands between 0 and 3, and \f$e_2/e_3\f$ is a decade
 *    or thereabouts. That ordering is what makes the error a **truncation**
 *    rather than a bias, and it is the property the whole basis choice rests
 *    on.
 * 6. **Negative case 2 — the basis selector is live.**
 *    `FarFieldBasis::SolidHarmonic` at `near_softening_factor = 0` misses the
 *    budget outright and sits a wide factor above the production arm *at the
 *    same order*, so the gap is the kernel and not the truncation — see
 *    `kBasisSeparation` for why that factor is not R2's "tens of percent".
 *    **A pass there would be the alarm**: it
 *    would mean the adapter is not selecting the basis it says it is
 *    (**R2**) — Canopy's basis template parameter defaults to `LaplaceKernel`,
 *    so an instantiation that omits it compiles, runs, and produces a bare
 *    \f$1/r\f$ far field.
 * 7. **Negative case 3 — the blob reaches the far field.** A second
 *    `BRSolverFMM` at a different `--eps` produces a different FMM velocity on
 *    the same state, and still tracks the direct solver *at that eps*. Under a
 *    bare-kernel far field the first would be nearly unchanged and the second
 *    would fail, which is the cheapest available proof that \f$b\f$ is inside
 *    \f$w\f$. It has to be a second solver: Canopy fixes the softening at
 *    `Solver` construction and pushes it into the M2L operator tables, so the
 *    adapter holds the value and **throws** if a later evaluation presents a
 *    different `blob()`. That guard is asserted here too.
 * 8. **A rank owning zero sources.** The mesh decomposition does not produce
 *    one at 2562 vertices over 1-6 ranks — the minimum owned count is reported
 *    so the log says so rather than leaving the case silently uncovered — so
 *    the case is constructed at the adapter's own interface instead: the last
 *    rank hands over its rows to rank 0 and evaluates on an empty source list.
 *    The global source set is unchanged, so rank 0's own rows must come back
 *    unchanged, and every rank must return from a collective Canopy has no test
 *    for.
 *
 * NOTHING HERE IS ASSERTED BITWISE, AND THAT IS MEASURED RATHER THAN ASSUMED
 * -------------------------------------------------------------------------
 * Two runs of an identical binary on an identical command line differ by
 * **1.15e-15 (np1)** and **1.88e-15 (np4)** of field RMS, localized to the
 * solve rather than to I/O (T3, `tasks/canopy/add-canopy-progress-log.md`). The
 * spun-up state this test compares on is downstream of five timesteps, so it
 * carries that noise. Every tolerance below clears it by at least nine decades
 * and the margin is stated on each one.
 *
 * THE `~canopy` BUILD
 * -------------------
 * This file carries the explicit instantiation of `FarFieldSolver` that closes
 * the gap T2 found and left here: `Beatnik_CreateBRSolver.hpp` preprocesses out
 * the `new BRSolverFMM<...>` line, which is the class's only construction site,
 * so a `~canopy` build instantiates `FarFieldSolver` nowhere and the guarded
 * half of the header is parsed but never definition-checked. In that build the
 * checks below reduce to the one thing that *is* true there — both evaluations
 * throw `std::runtime_error` naming the missing build option — so the member is
 * meaningful in both builds rather than vacuous in one.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp`.
 */

#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_BRSolverFMM.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#ifndef BEATNIK_ENABLE_CANOPY
//---------------------------------------------------------------------------//
// THE `~canopy` INSTANTIATION GAP, CLOSED PERMANENTLY.
//
// `Beatnik_CreateBRSolver.hpp` preprocesses out the `new BRSolverFMM<...>`
// line, which is the only construction site of the class, so nothing in the
// tree instantiates `FarFieldSolver` in a `~canopy` build -- T2 checked with
// `nm` and found zero symbols. The guarded half of the header is therefore
// parsed and definition-checked but never *instantiated*, which is weaker than
// it looks: a body that is ill-formed only on instantiation rots silently the
// next time someone edits it. This one line is what keeps that from happening,
// and it is guarded to `~canopy` because the `+canopy` build already
// instantiates every member through `BRSolverFMM` below.
//---------------------------------------------------------------------------//
template class Beatnik::FarFieldSolver<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
#endif

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The configuration, and why every number in it is the number it is.
//---------------------------------------------------------------------------//

/// Subdivision level 4, i.e. 2562 vertices. **Not** milestone-0's level 3.
/// Level 3's 642 vertices have NO solution to the liveness inequality at
/// \f$\theta=0.3\f$ under any `ncrit` — `ncrit = 8` gives 80 occupied leaves
/// against a 105-leaf near field, and lower values degenerate the tree — so a
/// comparison there is a comparison of two direct sums no matter what is done
/// to it. See the file header.
constexpr int kSubdivisions = 4;
constexpr long long kVertices = 2562;
constexpr long long kEdges = 7680;
constexpr long long kFaces = 5120;

/// Leaf occupancy for **this test only**. `FmmParams::ncrit` keeps the
/// reference's 64, which is right at production vertex counts and two orders
/// wrong at 2562: it would give 40 occupied leaves against a 105-leaf near
/// field. At 8 the surface occupies 320 leaves, a factor of 3 above the near
/// field — live, but not generous, which is why `kP2PFractionBound` is a bound
/// and not a hope. Canopy answered the same wall the same way, pairing 8640
/// particles with `ncrit = 8` rather than lowering the particle count
/// (`canopy/tests/tstCartesianTaylorSolve.hpp:96-107`).
constexpr int kNcrit = 8;

/// Steps of the **direct** solver before anything is compared.
/// `--initial-potential-strength 0` makes \f$\phi\f$, and therefore the sheet
/// vector \f$S=-\hat n\times\nabla_s\phi\f$, identically zero at step 0, so a
/// step-0 comparison compares two zero fields and passes at every order. Five
/// steps is enough for the forcing to fill \f$\phi\f$ and cheap at 2562
/// vertices; the strength is asserted non-zero rather than assumed.
constexpr int kSpinUpSteps = 5;

/// The production order, and the two scan orders the negative case uses.
constexpr int kProductionOrder = 3;
constexpr int kMonopoleOrder = 0;
constexpr int kBelowProductionOrder = 2;

/// `--eps` of the milestone-0 configuration, and the perturbed value the blob
/// negative case builds its second solver at. Under
/// `--kernel-blob-mode length` the blob is \f$b=\varepsilon^2\f$ and the
/// softening length Canopy is given is \f$\sqrt b=\varepsilon\f$, so these are
/// softening lengths directly. Doubling rather than nudging: the claim is that
/// the far field *responds* to the blob, and a nudge would leave the response
/// competing with the truncation error.
constexpr Real kEps = 0.025;
constexpr Real kEpsPerturbed = 0.050;

//---------------------------------------------------------------------------//
// Tolerances and bounds. Every one of them carries its derivation, because the
// conventions table forbids compiling a bare tolerance into a test.
//---------------------------------------------------------------------------//

/**
 * THE LIVENESS BOUND. `p2p_pair_fraction` is the adapter's
 * (global near-field pair count) / (global particle count)^2, so a solve with
 * no far field at all sits at 1.0.
 *
 * The a-priori expectation at this configuration: 320 occupied leaves, of which
 * about 105 fall inside any given leaf's near field, so roughly a third of the
 * pairs are P2P and two thirds go through M2L. 0.75 is chosen from that
 * arithmetic — comfortably above the expected ~0.33 so the bound is not a
 * measurement in disguise, and far enough below 1.0 that an all-P2P solve
 * cannot squeak past it. The realized fraction is noted at every evaluation, so
 * a later reader can see how much of the margin was used.
 */
constexpr double kP2PFractionBound = 0.75;

/**
 * THE BUDGET. Max velocity error against `BRSolverDirect`, relative to the
 * direct field's own max magnitude.
 *
 * **Qualification list, which the conventions table requires of every stated
 * tolerance and without which this number is unreadable:** it is on the
 * **velocity**, i.e. the *gradient* of Canopy's potential and not the potential
 * — the two differ by a full order at fixed `order`; source distribution is the
 * milestone-0 icosphere at subdivision level 4 (2562 vertices) after
 * `kSpinUpSteps` direct steps; rank counts 1, 2, 3, 4, 5 and 6; basis
 * `FarFieldBasis::CartesianTaylor`; `order = 3`; `ncrit = 8`; `max_depth = 10`;
 * `mac_theta = 0.3`; `softening = 0.025` (\f$=\sqrt{b}\f$ at `--eps 0.025`
 * under `--kernel-blob-mode length`); `near_softening_factor = 0`; and the
 * realized P2P pair fraction, which is reported at every evaluation and bounded
 * by `kP2PFractionBound` above.
 *
 * **Where the value comes from.** Not from this measurement, and not from
 * \f$\tau_A\f$ — **T5** measures \f$\tau_A\f$ and no T5 number exists yet.
 * Canopy measured \f$7.0718\times10^{-4}\f$ relative on the **gradient** at
 * \f$p=3\f$, \f$\theta=0.3\f$ on its volumetric cloud, against
 * \f$8.996\times10^{-3}\f$ at \f$p=2\f$; that pair is the provenance of
 * `FmmParams::order = 3` and it is what this budget is set from. 2.0e-3 is
 * about 2.8x above the \f$p=3\f$ figure — headroom for a 2-manifold rather than
 * a volumetric cloud, which is a different distribution and an unmeasured curve
 * — and 4.5x below the \f$p=2\f$ figure, which is the room the order-2 negative
 * case needs to sit between 3 and 0.
 *
 * **Floor.** Twelve decades above the 1.15e-15 / 1.88e-15 run-to-run noise
 * floor T3 measured on the same binary, so nothing here is a bitwise claim in
 * disguise.
 */
constexpr double kVelocityBudget = 2.0e-3;

/**
 * How far above the **production arm** the solid-harmonic arm must land for
 * negative case 2 to mean anything. Both arms run at `order = 3`, so any
 * difference between them is the kernel and not the truncation — which is what
 * makes a ratio the right form for this bound and an absolute level the wrong
 * one.
 *
 * **This is not R2's "tens of percent", and the reason is a finding rather than
 * a concession.** That figure describes a *self-contacting* sheet, where source
 * and target separations approach \f$\sqrt b\f$ and a far field with no blob in
 * its expansion is wrong by an O(1) factor. It does not describe this test's
 * geometry. Canopy accepts a pair only beyond \f$R/w>2\sqrt3/\theta=11.55\f$
 * half-widths; at 2562 vertices in a root box about 0.6 wide the occupied
 * leaves sit at depth 4-5, so \f$w\approx0.019\f$ to \f$0.0094\f$ and the
 * closest accepted separation is \f$R\approx0.11\f$ to \f$0.22\f$ — four to
 * nine times the softening length \f$\sqrt b=0.025\f$. The bare and softened
 * kernels differ there by \f$\tfrac32 b/R^2\f$, i.e. 2% to 8% on the closest
 * accepted pairs and less beyond, and those pairs carry a minority of a field
 * whose near half both bases evaluate identically through P2P. So the expected
 * whole-field separation is the \f$10^{-3}\f$-\f$10^{-2}\f$ band — about a
 * decade above the production arm — and **a bound of 100x the budget would
 * assert of a smooth sphere something that is only true of a roll-up.**
 *
 * 5.0 is a factor of two below that decade, the same shape of margin
 * `kOrderDecadeFloor` takes below Canopy's measured order ratio. A selector
 * that was not selecting would put the two arms on top of each other, which is
 * what this has to catch; the solid-harmonic arm additionally has to miss the
 * budget outright, asserted separately below, and that bound is independently
 * derived.
 */
constexpr double kBasisSeparation = 5.0;

/**
 * The truncation signature: \f$e_2/e_3\f$. Canopy's own pair is 12.7x
 * (8.996e-3 over 7.0718e-4), i.e. a decade. 3.0 is a factor of four of slack
 * below that, which still separates a truncation that responds to `order` from
 * a bias that does not — a bias would put \f$e_2\approx e_3\f$.
 */
constexpr double kOrderDecadeFloor = 3.0;

/**
 * How much the FMM velocity must change when the softening length doubles, for
 * negative case 3 to have teeth. Relative to the field scale. The change is
 * expected to be of order the blob's own effect on the field — percent-scale —
 * so 1e-3 is a floor, not a prediction, and it sits three decades above the
 * agreement two identical solvers would show.
 */
constexpr double kBlobResponseFloor = 1.0e-3;

/**
 * Agreement required between the baseline FMM field and the same global source
 * set redistributed so one rank owns nothing. The tree is built from the same
 * global particles, so the two differ only in accumulation order and in
 * Canopy's own partition — round-off, not truncation. 1e-9 is six decades below
 * the budget (so this can never be confused with an accuracy check) and six
 * decades above the 1e-15 floor.
 */
constexpr double kRedistributionTolerance = 1.0e-9;

//---------------------------------------------------------------------------//
// Global reductions. Every reported scalar goes through one of these over
// OWNED rows only (risk R9), so a six-rank run and a one-rank run compare the
// same quantity against the same literal.
//---------------------------------------------------------------------------//

double globalMax( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MAX, comm );
    return out;
}

double globalMin( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MIN, comm );
    return out;
}

long long globalCount( MPI_Comm comm, long long local )
{
    long long out = local;
    MPI_Allreduce( &local, &out, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

//---------------------------------------------------------------------------//
/// The milestone-0 command line as a `SolverParams`, at level 4 and with the
/// direct BR solver. Field for field
/// `Beatnik_Test_Milestone0Frozen.cpp::makeParams`, with three deliberate
/// departures, each required by what T4 is checking:
///
///   * `kSubdivisions` is 4 rather than that file's constant — see the header;
///   * no checkpoint directory, so nothing is written and the test needs no
///     `BEATNIK_TEST_SCRATCH` and no parallel filesystem;
///   * `time.steps` is `kSpinUpSteps`, and the loop is driven one step at a
///     time through `advanceOneStep()` rather than through `solve()`.
///
/// `--br-approximation direct` is what SPINS UP the state: the state both
/// solvers are then compared on must be one the FMM did not produce, or the
/// comparison is partly circular.
Beatnik::SolverParams makeSpinUpParams()
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere, --radius 0.25,
    // --center-z 0.25, --icosphere-subdivisions 4.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = kSubdivisions;
    p.initial.radius = 0.25;
    p.initial.center_z = 0.25;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0.
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --sigma 0
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = kEps;
    p.zmodel.sigma = 0.0;
    // --forcing-sign 1 --br-sign 1 --kernel-blob-mode length
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = Beatnik::KernelBlobMode::Length;
    // --viscosity-mode laplace-beltrami, --velocity-mode full,
    // --bernoulli-scalar-mode normal-speed, preserve_volume on.
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = Beatnik::VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = Beatnik::BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    // --br-approximation direct. See the note above.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex. MANDATORY, not decorative: ZModelParams
    // defaults to Face, whose generate() throws, and the far-field adapter
    // rejects anything but Vertex regardless -- source, target and output row
    // are one index on that path.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // The dt controls of the milestone-0 configuration.
    p.time.steps = kSpinUpSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --no-dynamic-remesh --refine-every 0. The connectivity is frozen, so the
    // adapter's forward distributor takes its reuse branch on every evaluation
    // after the first and the source set never changes shape under it.
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;
    p.cleanup.enabled = true;

    // No checkpoints at all: `CheckpointIO` is a no-op without a directory, so
    // this test writes nothing and needs no scratch path.
    p.checkpoint.every_steps = 0;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = "";

    return p;
}

//---------------------------------------------------------------------------//
/// `FmmParams` for one dispatch arm. Everything except `basis` and `order` is
/// the T1 default, with the single exception of `ncrit`, which this test must
/// lower to make the far field live at all (see `kNcrit`).
Beatnik::FmmParams makeFmmParams( Beatnik::FarFieldBasis basis, int order )
{
    Beatnik::FmmParams f;
    f.basis = basis;
    f.order = order;
    f.ncrit = kNcrit;
    return f;
}

//---------------------------------------------------------------------------//
/// Max magnitude of a `(N,3)` field over owned rows, reduced globally. The
/// field scale every error below is expressed against.
template <class ExecSpace, class ViewType>
double fieldScale( MPI_Comm comm, const ViewType& v, int n )
{
    Real worst = 0;
    Kokkos::parallel_reduce(
        "beatnik_t4_field_scale", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m ) {
            const Real mag = Kokkos::sqrt( v( i, 0 ) * v( i, 0 ) +
                                           v( i, 1 ) * v( i, 1 ) +
                                           v( i, 2 ) * v( i, 2 ) );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ) );
    return globalMax( comm, static_cast<double>( worst ) );
}

//---------------------------------------------------------------------------//
/// Max \f$|a_i - b_i|\f$ over owned rows, reduced globally, together with the
/// count of non-finite rows in `a`. One pass, because the finiteness check
/// (step 7) and the error (step 3) are the same walk.
struct FieldDifference
{
    double max_abs = 0.0;
    long long nonfinite = 0;
};

template <class ExecSpace, class ViewA, class ViewB>
FieldDifference fieldDifference( MPI_Comm comm, const ViewA& a, const ViewB& b,
                                 int n )
{
    Real worst = 0;
    long long bad = 0;
    Kokkos::parallel_reduce(
        "beatnik_t4_field_difference", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m, long long& nb ) {
            Real d[3];
            bool finite = true;
            for ( int c = 0; c < 3; ++c )
            {
                d[c] = a( i, c ) - b( i, c );
                if ( !Kokkos::isfinite( a( i, c ) ) )
                    finite = false;
            }
            if ( !finite )
            {
                ++nb;
                return;
            }
            const Real mag =
                Kokkos::sqrt( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ), bad );

    FieldDifference out;
    out.max_abs = globalMax( comm, static_cast<double>( worst ) );
    out.nonfinite = globalCount( comm, bad );
    return out;
}

//---------------------------------------------------------------------------//
/// Everything one FMM evaluation produced, so a caller can compare arms without
/// re-deriving any of it.
struct ArmResult
{
    double max_abs = 0.0;      ///< max |u_fmm - u_direct| over owned rows.
    double rel = 0.0;          ///< the same, over the direct field's scale.
    long long nonfinite = 0;   ///< non-finite rows in the FMM velocity.
    double p2p_fraction = 0.0; ///< realized, from the adapter's diagnostics.
    long long particles = 0;   ///< global count Canopy held.
    long long m2l_pairs = 0;
    long long m2l_fallback = 0;
    double softening = 0.0;
    Beatnik::FarFieldBasis basis = Beatnik::FarFieldBasis::CartesianTaylor;
    int order = 0;
};

//---------------------------------------------------------------------------//
/// One line in the log per FMM evaluation, carrying the whole qualification
/// list the conventions table requires beside any number taken from it.
void noteArm( Beatnik::Test::Recorder& rec, const std::string& label,
              const ArmResult& r, double scale )
{
    std::ostringstream os;
    os.precision( 6 );
    os << label << ": basis " << Beatnik::toString( r.basis ) << ", order "
       << r.order << ", ncrit " << kNcrit << ", mac_theta 0.3, softening "
       << r.softening << "; max|du| " << r.max_abs << " over field scale "
       << scale << " = " << r.rel << " relative (budget " << kVelocityBudget
       << "); p2p fraction " << r.p2p_fraction << " (bound "
       << kP2PFractionBound << "); particles " << r.particles << ", m2l pairs "
       << r.m2l_pairs << ", m2l fallback " << r.m2l_fallback;
    rec.note( os.str() );
}

//---------------------------------------------------------------------------//
// The body.
//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using solver_type = Beatnik::Solver<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using direct_type = Beatnik::BRSolverDirect<ExecSpace, MemSpace>;
    using fmm_type = Beatnik::BRSolverFMM<ExecSpace, MemSpace>;
    using vector_view = Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );

    {
        std::ostringstream os;
        os << "execution space " << ExecSpace::name() << ", ranks "
           << comm_size << ", icosphere subdivisions " << kSubdivisions
           << " (" << kVertices << " vertices), spin-up steps " << kSpinUpSteps
           << " with --br-approximation direct, ncrit " << kNcrit;
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // Group 0 -- the state. Five direct steps from the milestone-0 initial
    // condition, because at step 0 the sheet strength is identically zero and
    // a comparison there is vacuous.
    //-----------------------------------------------------------------------//
    Beatnik::SolverParams params = makeSpinUpParams();
    solver_type solver( comm, params );
    solver.setup();

    auto& mesh = solver.mesh();
    const auto& state = solver.state();

    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

    for ( int s = 0; s < kSpinUpSteps; ++s )
    {
        const bool ok = solver.advanceOneStep();
        if ( !ok )
        {
            rec.fail( "the direct spin-up went non-finite at step " +
                      std::to_string( s + 1 ) +
                      "; nothing downstream of this is meaningful" );
            return;
        }
    }
    BEATNIK_CHECK_EQ( rec, solver.step(), kSpinUpSteps );

    // The evaluation preconditions, in the order ZModelSolver's RHS
    // establishes them (`Beatnik_ZModelSolver.hpp`, steps 0-2): one whole-tuple
    // halo exchange, geometry at the CURRENT positions, then the sheet vector.
    // Both BR solvers are then handed the SAME mesh, geometry, state and
    // quadrature, which is what makes this a comparison of the two summations
    // and of nothing else.
    mesh.haloExchange();
    geometry_type geometry;
    geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                      mesh.faceVertices() );
    state.updateSheetVector( mesh, geometry );

    auto quadrature =
        Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
            Beatnik::SourceQuadrature::Vertex );

    const int n_owned = mesh.ownedVertexCount();
    const long long owned_total = globalCount( comm, n_owned );
    BEATNIK_CHECK_EQ( rec, owned_total, kVertices );

    // Can the mesh decomposition produce a rank owning zero sources at this
    // vertex count? Reported rather than assumed -- group 8 is what covers the
    // case when the answer is no.
    const double min_owned =
        globalMin( comm, static_cast<double>( n_owned ) );
    {
        std::ostringstream os;
        os << "owned vertices: this rank " << n_owned << ", global "
           << owned_total << ", minimum over ranks "
           << static_cast<long long>( min_owned );
        rec.note( os.str() );
    }

    // THE STATE IS REAL. A zero sheet strength makes every velocity zero and
    // every comparison below vacuous, so this is asserted before anything is
    // compared rather than inferred from the step count.
    typename Beatnik::SourceQuadratureBase<ExecSpace, MemSpace>::point_view
        points;
    typename Beatnik::SourceQuadratureBase<ExecSpace, MemSpace>::strength_view
        strengths;
    quadrature->generate( mesh, geometry, state, points, strengths );
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( points.extent( 0 ) ),
                      n_owned );
    const double strength_scale =
        fieldScale<ExecSpace>( comm, strengths, n_owned );
    {
        std::ostringstream os;
        os.precision( 6 );
        os << "after " << kSpinUpSteps << " direct steps: max |area * S| "
           << strength_scale << ", simulation time " << solver.time();
        rec.note( os.str() );
    }
    BEATNIK_CHECK_TRUE( rec, strength_scale > 0.0 );

    //-----------------------------------------------------------------------//
    // The direct velocity -- the reference every arm below is measured against.
    //-----------------------------------------------------------------------//
    direct_type direct( comm );
    vector_view u_direct( "beatnik_t4_u_direct", n_owned );
    direct.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                     params.zmodel, u_direct );
    const double scale = fieldScale<ExecSpace>( comm, u_direct, n_owned );
    {
        std::ostringstream os;
        os.precision( 6 );
        os << "direct velocity: max |u| " << scale;
        rec.note( os.str() );
    }
    // A near-zero direct field would make every relative error below a ratio of
    // two round-off quantities. 1e-6 is a sanity floor, not a reference value.
    BEATNIK_CHECK_TRUE( rec, scale > 1.0e-6 );

    //-----------------------------------------------------------------------//
    // One FMM arm, evaluated on that same state. Returns everything the
    // diagnostics carry so the caller asserts rather than assumes.
    //-----------------------------------------------------------------------//
    auto evaluateArm = [&]( Beatnik::FarFieldBasis basis, int order,
                            const Beatnik::ZModelParams& zparams,
                            vector_view& u_out ) -> ArmResult
    {
        fmm_type fmm( comm, makeFmmParams( basis, order ) );
        Kokkos::realloc( u_out, n_owned );
        fmm.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                      zparams, u_out );

        const auto& diag = fmm.farField().diagnostics();
        const FieldDifference d =
            fieldDifference<ExecSpace>( comm, u_out, u_direct, n_owned );

        ArmResult r;
        r.max_abs = d.max_abs;
        r.rel = d.max_abs / scale;
        r.nonfinite = d.nonfinite;
        r.p2p_fraction = diag.p2p_pair_fraction;
        r.particles = diag.global_particle_count;
        r.m2l_pairs = diag.global_m2l_pair_count;
        r.m2l_fallback = diag.global_m2l_fallback_pair_count;
        r.softening = static_cast<double>( diag.softening );
        r.basis = diag.basis;
        r.order = diag.order;
        return r;
    };

    //-----------------------------------------------------------------------//
    // Groups 1-4 -- THE PRODUCTION ARM. The round trip, the liveness of the
    // far field, the softening, and only then the accuracy claim.
    //-----------------------------------------------------------------------//
    vector_view u_fmm( "beatnik_t4_u_fmm", n_owned );
    const ArmResult production = evaluateArm(
        Beatnik::FarFieldBasis::CartesianTaylor, kProductionOrder,
        params.zmodel, u_fmm );
    noteArm( rec, "production arm", production, scale );

    // Group 1 -- THE ROUND TRIP (R3). A tag mismatch does not crash; it drops
    // or duplicates a source, which is a wrong velocity and not a slow one.
    // This count is the cheap independent discriminator, and it is an integer
    // reduction so it is exact at every rank count.
    BEATNIK_CHECK_EQ( rec, production.particles, kVertices );
    // And the result is finite everywhere. `checkClose` already refuses a
    // non-finite value, but a max over a view containing a NaN is itself NaN,
    // so the count is what localizes it.
    BEATNIK_CHECK_EQ( rec, production.nonfinite, 0 );

    // Group 2 -- THE FAR FIELD IS LIVE. Nothing below this line is evidence
    // about the expansion unless these two pass: a solve that is entirely P2P
    // agrees with the direct sum to round-off at every order and reads as a
    // success (R1).
    BEATNIK_CHECK_TRUE( rec, production.p2p_fraction < kP2PFractionBound );
    BEATNIK_CHECK_TRUE( rec, production.m2l_pairs > 0 );
    // A non-zero fallback count means the accuracy number below is a mixture of
    // two code paths -- the same mathematics evaluated pair by pair, slower and
    // bitwise different (R6). Asserted, not merely reported, because a mixed
    // number is not the number this test claims to have measured.
    BEATNIK_CHECK_EQ( rec, production.m2l_fallback, 0 );

    // Group 3 -- THE SOFTENING AND THE NEAR-FIELD FLOOR, i.e. the half of T1's
    // exit criterion that first becomes runnable here. `softening` is a LENGTH
    // and `blob()` is a squared length, so the comparison is against its square
    // root; a value of -1 would be Canopy's auto-softening sentinel and a
    // different kernel entirely, which is why positivity is asserted separately
    // rather than left implicit in the equality.
    {
        fmm_type probe( comm, makeFmmParams(
                                  Beatnik::FarFieldBasis::CartesianTaylor,
                                  kProductionOrder ) );
        BEATNIK_CHECK_EQ( rec, probe.farField().params().ncrit, kNcrit );
        BEATNIK_CHECK_CLOSE( rec, probe.farField().params().near_softening_factor,
                             0.0, 0.0 );
        BEATNIK_CHECK_CLOSE( rec, probe.farField().params().mac_theta, 0.3,
                             1.0e-15 );
    }
    BEATNIK_CHECK_TRUE( rec, production.softening > 0.0 );
    BEATNIK_CHECK_CLOSE( rec, production.softening,
                         std::sqrt( static_cast<double>(
                             params.zmodel.blob() ) ),
                         1.0e-15 );
    BEATNIK_CHECK_EQ( rec, static_cast<int>( production.basis ),
                      static_cast<int>(
                          Beatnik::FarFieldBasis::CartesianTaylor ) );
    BEATNIK_CHECK_EQ( rec, production.order, kProductionOrder );

    // Group 4 -- THE ACCURACY CLAIM. Both forms, as the exit criterion asks:
    // the max absolute error against the budget expressed in field units, and
    // the max relative error against the budget itself. They are the same
    // statement; both are written so a failure report names whichever the
    // reader is thinking in.
    BEATNIK_CHECK_TRUE( rec, production.max_abs <= kVelocityBudget * scale );
    BEATNIK_CHECK_TRUE( rec, production.rel <= kVelocityBudget );

    //-----------------------------------------------------------------------//
    // Group 5 -- NEGATIVE CASE 1: THE ORDER KNOB IS LIVE, AND THE ERROR IS A
    // TRUNCATION.
    //
    // Under `CartesianTaylorBasis` the error is a Taylor truncation at the
    // accepted separation ratio, going as (cw/R)^p on the gradient, so `order`
    // is the knob -- which is the opposite of what a bare-kernel far field
    // does, and is the property the whole basis choice rests on. Monopole-only
    // must blow the budget; order 2 must land between 0 and 3; and e2/e3 must
    // be a decade or thereabouts. A BIAS would put e2 ~ e3 and pass a check
    // that only looked at order 0.
    //-----------------------------------------------------------------------//
    vector_view u_p0( "beatnik_t4_u_p0", n_owned );
    const ArmResult monopole =
        evaluateArm( Beatnik::FarFieldBasis::CartesianTaylor, kMonopoleOrder,
                     params.zmodel, u_p0 );
    noteArm( rec, "negative 1a (order 0)", monopole, scale );

    vector_view u_p2( "beatnik_t4_u_p2", n_owned );
    const ArmResult below =
        evaluateArm( Beatnik::FarFieldBasis::CartesianTaylor,
                     kBelowProductionOrder, params.zmodel, u_p2 );
    noteArm( rec, "negative 1b (order 2)", below, scale );

    {
        std::ostringstream os;
        os.precision( 6 );
        os << "NEGATIVE CASE 1 -- THE ORDER KNOB IS LIVE AND THE ERROR IS A "
              "TRUNCATION: order 0 gives "
           << monopole.rel << " relative, order 2 gives " << below.rel
           << ", order 3 gives " << production.rel << "; budget "
           << kVelocityBudget << ", e2/e3 = "
           << ( production.rel > 0.0 ? below.rel / production.rel : 0.0 )
           << " (floor " << kOrderDecadeFloor
           << "). A Taylor truncation FALLS with `order`; a bias would not.";
        rec.note( os.str() );
    }
    BEATNIK_CHECK_TRUE( rec, monopole.rel > kVelocityBudget );
    BEATNIK_CHECK_TRUE( rec, below.rel < monopole.rel );
    BEATNIK_CHECK_TRUE( rec, production.rel < below.rel );
    BEATNIK_CHECK_TRUE( rec,
                        below.rel > kOrderDecadeFloor * production.rel );
    // The far field was live in both, or the ordering above says nothing about
    // the expansion.
    BEATNIK_CHECK_TRUE( rec, monopole.p2p_fraction < kP2PFractionBound );
    BEATNIK_CHECK_TRUE( rec, below.p2p_fraction < kP2PFractionBound );

    //-----------------------------------------------------------------------//
    // Group 6 -- NEGATIVE CASE 2: THE BASIS SELECTOR IS LIVE.
    //
    // THE ONE CASE WHOSE *PASSING* WOULD BE THE ALARM (R2). Canopy's basis
    // template parameter defaults to `LaplaceKernel`, so an instantiation that
    // omits it compiles, runs, and expands the BARE 1/r field -- with
    // `near_softening_factor = 0` the blob then reaches neither the expansion
    // nor a near-field floor, and the error is tens of percent rather than
    // small. If the solid-harmonic arm agreed with the production arm, the
    // adapter would not be selecting the basis it says it is.
    //-----------------------------------------------------------------------//
    vector_view u_sh( "beatnik_t4_u_sh", n_owned );
    const ArmResult harmonic =
        evaluateArm( Beatnik::FarFieldBasis::SolidHarmonic, kProductionOrder,
                     params.zmodel, u_sh );
    noteArm( rec, "negative 2 (solid-harmonic)", harmonic, scale );
    {
        std::ostringstream os;
        os.precision( 6 );
        os << "NEGATIVE CASE 2 -- THE BASIS IN FORCE IS '"
           << Beatnik::toString( harmonic.basis )
           << "' AT near_softening_factor 0, SO THE EXPANDED KERNEL IS THE "
              "BARE ONE: "
           << harmonic.rel << " relative, against the production basis '"
           << Beatnik::toString( production.basis ) << "' at "
           << production.rel << " -- a factor of "
           << ( production.rel > 0.0 ? harmonic.rel / production.rel : 0.0 )
           << " at the SAME order, so the gap is the kernel and not the "
              "truncation (required "
           << kBasisSeparation << "x, and it must miss the budget "
           << kVelocityBudget
           << " outright). A PASS HERE WOULD BE THE ALARM, not the result.";
        rec.note( os.str() );
    }
    BEATNIK_CHECK_EQ( rec, static_cast<int>( harmonic.basis ),
                      static_cast<int>(
                          Beatnik::FarFieldBasis::SolidHarmonic ) );
    BEATNIK_CHECK_TRUE( rec, harmonic.rel > kVelocityBudget );
    BEATNIK_CHECK_TRUE( rec,
                        harmonic.rel > kBasisSeparation * production.rel );
    BEATNIK_CHECK_TRUE( rec, harmonic.p2p_fraction < kP2PFractionBound );

    //-----------------------------------------------------------------------//
    // Group 7 -- NEGATIVE CASE 3: THE BLOB REACHES THE FAR FIELD.
    //
    // Two statements and a guard. (a) A second solver at twice the softening
    // length produces a DIFFERENT velocity on the same state -- under a
    // bare-kernel far field the blob would reach only the near field and the
    // response would be a fraction of this. (b) That second field still tracks
    // the direct solver AT THAT eps, inside the same budget, which is the
    // statement that the blob is inside w rather than merely adjacent to it.
    // (c) The adapter REFUSES a changed blob on a live solver -- Canopy fixes
    // the softening at `Solver` construction and pushes it into the M2L
    // operator tables, so a silent change would run one kernel in the far field
    // and another in the near. A guard, not a defect, and T7 meets it too.
    //-----------------------------------------------------------------------//
    Beatnik::ZModelParams zperturbed = params.zmodel;
    zperturbed.eps = kEpsPerturbed;

    direct_type direct_perturbed( comm );
    vector_view u_direct_perturbed( "beatnik_t4_u_direct_eps2", n_owned );
    direct_perturbed.computeInterfaceVelocity( mesh, geometry, state,
                                               *quadrature, zperturbed,
                                               u_direct_perturbed );

    vector_view u_fmm_perturbed( "beatnik_t4_u_fmm_eps2", n_owned );
    fmm_type fmm_perturbed(
        comm, makeFmmParams( Beatnik::FarFieldBasis::CartesianTaylor,
                             kProductionOrder ) );
    fmm_perturbed.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                            zperturbed, u_fmm_perturbed );
    const auto& perturbed_diag = fmm_perturbed.farField().diagnostics();

    const FieldDifference blob_response =
        fieldDifference<ExecSpace>( comm, u_fmm_perturbed, u_fmm, n_owned );
    const FieldDifference perturbed_error = fieldDifference<ExecSpace>(
        comm, u_fmm_perturbed, u_direct_perturbed, n_owned );
    const double scale_perturbed =
        fieldScale<ExecSpace>( comm, u_direct_perturbed, n_owned );

    {
        std::ostringstream os;
        os.precision( 6 );
        os << "NEGATIVE CASE 3 -- THE SOFTENING LENGTH IS INSIDE THE "
              "EXPANSION: sqrt(blob) went from "
           << static_cast<double>( production.softening ) << " to "
           << static_cast<double>( perturbed_diag.softening )
           << " and the FMM velocity changed by "
           << ( blob_response.max_abs / scale ) << " relative (floor "
           << kBlobResponseFloor << "), while still tracking the direct "
              "solver at that softening length to "
           << ( perturbed_error.max_abs / scale_perturbed )
           << " (budget " << kVelocityBudget
           << "). A bare-kernel far field would fail the second.";
        rec.note( os.str() );
    }
    BEATNIK_CHECK_CLOSE( rec, static_cast<double>( perturbed_diag.softening ),
                         std::sqrt( static_cast<double>( zperturbed.blob() ) ),
                         1.0e-15 );
    BEATNIK_CHECK_TRUE( rec, blob_response.max_abs >
                                 kBlobResponseFloor * scale );
    BEATNIK_CHECK_EQ( rec, perturbed_error.nonfinite, 0 );
    BEATNIK_CHECK_TRUE( rec, perturbed_error.max_abs <=
                                 kVelocityBudget * scale_perturbed );
    BEATNIK_CHECK_TRUE( rec, perturbed_diag.p2p_pair_fraction <
                                 kP2PFractionBound );

    // (c) The guard. A live solver presented a different blob must THROW: the
    // value it was constructed with is already baked into the operator tables.
    {
        bool threw = false;
        std::string what;
        try
        {
            vector_view scratch( "beatnik_t4_guard", n_owned );
            fmm_perturbed.computeInterfaceVelocity( mesh, geometry, state,
                                                    *quadrature, params.zmodel,
                                                    scratch );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            what = e.what();
        }
        rec.note( "softening-stability guard: " +
                  ( threw ? ( "threw -- " + what )
                          : std::string( "DID NOT THROW, so a changed blob "
                                         "would run two different kernels" ) ) );
        BEATNIK_CHECK_TRUE( rec, threw );
    }

    //-----------------------------------------------------------------------//
    // Group 8 -- A RANK THAT OWNS ZERO SOURCES.
    //
    // The mesh decomposition does not produce one at 2562 vertices over 1-6
    // ranks (the minimum owned count is reported in group 0), so the case is
    // constructed at the adapter's own interface instead of being left silently
    // uncovered: the last rank hands its rows to rank 0 and evaluates on an
    // EMPTY source list. The global source set is byte-identical, so this is a
    // pure redistribution -- rank 0's own rows must come back unchanged to
    // round-off, the global particle count must still be 2562, and every rank
    // must RETURN from a collective sequence Canopy has no test for.
    //
    // stdout is flushed first, deliberately: if a zero-particle rank hangs or
    // aborts inside Canopy, `report()` never runs and the notes above are all
    // the log will have.
    //-----------------------------------------------------------------------//
    std::fflush( stdout );
    if ( comm_size < 2 )
    {
        rec.note( "zero-source rank: not constructible at one rank; the case "
                  "is covered by the multi-rank launches of this same binary" );
    }
    else
    {
        const int last = comm_size - 1;
        auto h_points = Kokkos::create_mirror_view( points );
        auto h_strengths = Kokkos::create_mirror_view( strengths );
        Kokkos::deep_copy( h_points, points );
        Kokkos::deep_copy( h_strengths, strengths );

        int skew_n = n_owned;
        if ( rank == last )
            skew_n = 0;

        // Row-major buffers rather than a view-to-view copy: the device view's
        // layout is LayoutLeft here and the appended rows are not contiguous in
        // it, so the transfer is written elementwise and the layout cannot
        // silently transpose it.
        std::vector<double> sendbuf;
        if ( rank == last )
        {
            sendbuf.resize( static_cast<std::size_t>( n_owned ) * 6 );
            for ( int i = 0; i < n_owned; ++i )
                for ( int c = 0; c < 3; ++c )
                {
                    sendbuf[static_cast<std::size_t>( i ) * 6 + c] =
                        static_cast<double>( h_points( i, c ) );
                    sendbuf[static_cast<std::size_t>( i ) * 6 + 3 + c] =
                        static_cast<double>( h_strengths( i, c ) );
                }
            int n = n_owned;
            MPI_Send( &n, 1, MPI_INT, 0, 700, comm );
            MPI_Send( sendbuf.data(), n * 6, MPI_DOUBLE, 0, 701, comm );
        }
        std::vector<double> recvbuf;
        int extra = 0;
        if ( rank == 0 )
        {
            MPI_Recv( &extra, 1, MPI_INT, last, 700, comm, MPI_STATUS_IGNORE );
            recvbuf.resize( static_cast<std::size_t>( extra ) * 6 );
            MPI_Recv( recvbuf.data(), extra * 6, MPI_DOUBLE, last, 701, comm,
                      MPI_STATUS_IGNORE );
            skew_n = n_owned + extra;
        }

        typename Beatnik::SourceQuadratureBase<ExecSpace,
                                               MemSpace>::point_view
            skew_points( "beatnik_t4_skew_points", skew_n );
        typename Beatnik::SourceQuadratureBase<ExecSpace,
                                               MemSpace>::strength_view
            skew_strengths( "beatnik_t4_skew_strengths", skew_n );
        {
            auto h_sp = Kokkos::create_mirror_view( skew_points );
            auto h_ss = Kokkos::create_mirror_view( skew_strengths );
            const int mine = ( rank == last ) ? 0 : n_owned;
            for ( int i = 0; i < mine; ++i )
                for ( int c = 0; c < 3; ++c )
                {
                    h_sp( i, c ) = h_points( i, c );
                    h_ss( i, c ) = h_strengths( i, c );
                }
            for ( int i = 0; i < extra; ++i )
                for ( int c = 0; c < 3; ++c )
                {
                    h_sp( n_owned + i, c ) = static_cast<Real>(
                        recvbuf[static_cast<std::size_t>( i ) * 6 + c] );
                    h_ss( n_owned + i, c ) = static_cast<Real>(
                        recvbuf[static_cast<std::size_t>( i ) * 6 + 3 + c] );
                }
            Kokkos::deep_copy( skew_points, h_sp );
            Kokkos::deep_copy( skew_strengths, h_ss );
        }

        Beatnik::FarFieldSolver<ExecSpace, MemSpace> skewed(
            comm, makeFmmParams( Beatnik::FarFieldBasis::CartesianTaylor,
                                 kProductionOrder ) );
        vector_view u_skew( "beatnik_t4_u_skew", skew_n );
        skewed.evaluateVelocity( skew_points, skew_strengths, params.zmodel,
                                 u_skew );
        const auto& skew_diag = skewed.diagnostics();

        // Rank 0's own rows are still rows [0, n_owned) of what it handed in,
        // so they are directly comparable with the baseline field. The rows it
        // took over from the last rank, and every other rank's rows, are
        // covered by the global count and by the fact that the sequence
        // returned at all.
        const int compared = ( rank == 0 ) ? n_owned : 0;
        Real worst = 0;
        Kokkos::parallel_reduce(
            "beatnik_t4_skew_difference",
            Kokkos::RangePolicy<ExecSpace>( 0, compared ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                for ( int c = 0; c < 3; ++c )
                {
                    const Real e =
                        Kokkos::fabs( u_skew( i, c ) - u_fmm( i, c ) );
                    if ( e > m )
                        m = e;
                }
            },
            Kokkos::Max<Real>( worst ) );
        const double skew_diff = globalMax( comm, static_cast<double>( worst ) );

        std::ostringstream os;
        os.precision( 6 );
        os << "zero-source rank: rank " << last << " handed its " 
           << ( rank == 0 ? extra : ( rank == last ? n_owned : 0 ) )
           << " rows to rank 0 and evaluated on an empty source list; global "
              "particle count "
           << skew_diag.global_particle_count << ", p2p fraction "
           << skew_diag.p2p_pair_fraction
           << ", max |u_skew - u_fmm| on rank 0's own rows " << skew_diff
           << " over field scale " << scale << " = "
           << ( skew_diff / scale ) << " relative (tolerance "
           << kRedistributionTolerance << ")";
        rec.note( os.str() );

        BEATNIK_CHECK_EQ( rec, skew_diag.global_particle_count, kVertices );
        BEATNIK_CHECK_TRUE( rec, skew_diff <=
                                     kRedistributionTolerance * scale );
    }
}

//---------------------------------------------------------------------------//
/// The `~canopy` face of this test. `BRSolverFMM` still constructs -- the
/// adapter's constructor is a no-op there -- and both evaluations throw a
/// `std::runtime_error` naming the build option. That is the whole of what is
/// true in that build, and asserting it is what keeps the member meaningful
/// rather than vacuous.
template <class ExecSpace, class MemSpace>
void runChecksNoCanopy( Beatnik::Test::Recorder& rec )
{
    using fmm_type = Beatnik::BRSolverFMM<ExecSpace, MemSpace>;
    using vector_view =
        Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using state_type = Beatnik::SurfaceState<ExecSpace, MemSpace>;

    rec.note( "built WITHOUT Canopy: asserting that the far field refuses to "
              "run rather than comparing anything" );

    Beatnik::FmmParams fmm_params = makeFmmParams(
        Beatnik::FarFieldBasis::CartesianTaylor, kProductionOrder );
    fmm_type fmm( MPI_COMM_WORLD, fmm_params );
    BEATNIK_CHECK_EQ( rec, static_cast<int>( fmm.kind() ),
                      static_cast<int>( Beatnik::BRApproximation::Fmm ) );

    mesh_type mesh( MPI_COMM_WORLD );
    const Real center[3] = { 0.0, 0.0, 0.25 };
    mesh.generateIcosphere( 1, 0.25, center );
    geometry_type geometry;
    geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                      mesh.faceVertices() );
    state_type state( Beatnik::StateModel::Potential );
    auto quadrature = Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
        Beatnik::SourceQuadrature::Vertex );

    Beatnik::SolverParams params = makeSpinUpParams();
    vector_view velocity( "beatnik_t4_no_canopy", mesh.ownedVertexCount() );
    bool threw = false;
    std::string what;
    try
    {
        fmm.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                      params.zmodel, velocity );
    }
    catch ( const std::runtime_error& e )
    {
        threw = true;
        what = e.what();
    }
    rec.note( std::string( "evaluateVelocity: " ) +
              ( threw ? ( "threw -- " + what )
                      : std::string( "DID NOT THROW" ) ) );
    BEATNIK_CHECK_TRUE( rec, threw );
    BEATNIK_CHECK_TRUE(
        rec, what.find( "BEATNIK_ENABLE_CANOPY" ) != std::string::npos );
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_FmmVsDirect" );
        try
        {
            // One binary on the default execution space, no backend suffix --
            // the `unit` tier's convention; see tests/unit_tests/CMakeLists.txt.
            using ExecSpace = Kokkos::DefaultExecutionSpace;
#ifdef BEATNIK_ENABLE_CANOPY
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
#else
            runChecksNoCanopy<ExecSpace, typename ExecSpace::memory_space>(
                rec );
#endif
        }
        catch ( const std::exception& e )
        {
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();

    // ONE VERDICT ACROSS THE RANKS. Every rank reported, so the log names which
    // rank failed; the reduction is here and not inside the recorder because a
    // collective in there would deadlock exactly when one rank took an early
    // exception path. See Beatnik_TestAssert.hpp.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
