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
 * @file Beatnik_Test_BirkhoffRott.cpp
 * @brief **T2c's EXIT CRITERION, MECHANIZED** — the vertex source quadrature
 *        and the direct O(N^2) Birkhoff-Rott sum, against a hard-coded
 *        reference computed from the read-only Python.
 *
 * THIS IS THE SECOND MEMBER OF THE SHIP GATE.
 * `tasks/framework.md` states T2c's exit criterion as: *a `regression`-tier
 * test comparing the induced velocity on the default icosphere against a
 * hard-coded reference computed from the Python, to `1e-13` relative.* The tier
 * — not this file — supplies the rank sweep, per the convention T1c set, so the
 * criterion's ranks 1 and 4 are a verified subset of the gate's 1-6 on SERIAL
 * and HIP. The test reads its own comm size and adapts.
 *
 * WHY THE SOURCE STATE IS SYNTHETIC AND NOT THE INITIAL CONDITION
 * --------------------------------------------------------------
 * After `InitialCondition::build` the potential is identically zero on the
 * default configuration, so the sheet vector is zero and the induced velocity
 * is zero everywhere — which every implementation of this kernel reproduces,
 * including a wrong one. The source is therefore the same synthetic linear
 * potential T2b validated, \f$\phi = a\cdot p\f$ with
 * `a = (0.3, -0.7, 1.1)`, on the same mesh. That buys a cross-check on the
 * *source* before the kernel is compared at all: T2b published
 * `max|S| = 1.3193451648051979` and `sum|S| = 167.62467266803063` for exactly
 * this field on exactly this mesh, and both are asserted below. A disagreement
 * therefore localizes to the BR kernel rather than to the sheet strength that
 * feeds it.
 *
 * ORDER-INVARIANT SUMMARY SCALARS, AND ONE SIGNED ONE
 * --------------------------------------------------
 * Everything compared is a max, a min, or a sum over the whole surface, as at
 * T2b — so no vertex-order matching between Beatnik and the Python is needed,
 * and the same literals hold at every rank count. At least one is **signed**
 * (`sum u_x`, `sum u_y`, `sum u_z`, and the Riesz scalar's min): a
 * magnitude-only comparison cannot see a reversed \f$\delta\times S\f$, which
 * negates the whole velocity while leaving every \f$|u|\f$ unchanged, and that
 * is the single most likely error in this kernel.
 *
 * ONE COMPARISON IS BITWISE AND ONE IS NOT, AND THE DIFFERENCE IS MEASURED
 * -----------------------------------------------------------------------
 * The `br_sign = -1` negative case at the end asserts an exact equality on the
 * velocity and a `1e-13` bound on the Riesz scalar. That is not a tolerance
 * chosen to make it pass — the two paths genuinely differ, because the Riesz
 * path re-runs `surfaceGradient`, whose atomic face scatter is not bitwise
 * reproducible on HIP. The reasoning, the measurement that forced it and the
 * discriminator that keeps the check meaningful are at that block.
 *
 * HOW THE REFERENCE NUMBERS WERE OBTAINED
 * ---------------------------------------
 * Every `kPy*` literal below was computed by calling the **read-only**
 * reference (`~/research-bridges/zmodel-steve/zmodel3d-amr`) on
 * `mesh.icosphere_mesh( subdivisions=2, radius=0.25, center=(0,0,0.25) )` —
 * the Python's own defaults, i.e. the mesh T1a's gold file describes — with the
 * same `a`, through `potential_mesh_birkhoff_rott_velocity` and
 * `potential_surface_riesz_scalar` at `source_quadrature="vertex"`,
 * `br_approximation="direct"`, `eps=0.025`, `use_matlab_blob=False`,
 * `br_sign=1`. They are hard-coded and **must not be adjusted to make this test
 * pass**: a mismatch at `1e-13` is a real disagreement with the reference and
 * is the finding.
 *
 * RISK R9 — OWNED-ONLY SOURCES, CHECKED RATHER THAN ASSUMED
 * ---------------------------------------------------------
 * The failure mode this task can introduce is a quadrature that emits sources
 * for the whole local vertex set instead of `[0, ownedVertexCount())`. That is
 * smoothly wrong and scales with the ghost fraction, i.e. it changes with the
 * rank count — which is why this test is in the multi-rank gate. Following
 * T1c's template, the discriminating numbers are reported and not just the
 * verdict:
 *
 *   1. the owned sets partition the global sets (162 / 480 / 320), summed with
 *      a plain `MPI_Allreduce` rather than read from Tessera;
 *   2. the GLOBAL SOURCE COUNT is reduced and asserted to be exactly 162 —
 *      the direct detector, since a ghost-inclusive list makes it 200-400 here;
 *   3. every compared scalar is reported to 17 digits with its relative error
 *      at every rank count, so a reader can tell a last-ulp summation-order
 *      difference (R2) from a ghost-fraction-scaled one (R9).
 *
 * ARGUMENTS: none. Everything this test needs it computes.
 */

#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The mesh and the source field. The Python's defaults, from
// `run_adaptive_mesh_bubble.py::parse_args`: subdivision 2, radius 0.25, centre
// (0, 0, 0.25). Same mesh as T1a's gold file, T1b's test and T2b's test.
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;
constexpr long long kVertices = 162;
constexpr long long kEdges = 480;
constexpr long long kFaces = 320;

/// \f$\phi(p) = a\cdot p\f$. The same direction T2b used, deliberately generic:
/// an axis-aligned `a` would let the icosahedron's own symmetry planes cancel
/// errors this one exposes.
constexpr Real kA[3] = { 0.3, -0.7, 1.1 };

/// `--eps 0.025` with `--kernel-blob-mode length` (both codes' default), so the
/// kernel denominator offset is `eps^2 = 6.25e-4`.
constexpr Real kEps = 0.025;

//---------------------------------------------------------------------------//
// Reference values. See the file header for how they were produced and why they
// must not be edited.
//---------------------------------------------------------------------------//

// potential_sheet_vector -- the SOURCE, cross-checked before the kernel. These
// are T2b's published numbers, reproduced here from the same reference call.
constexpr double kPySheetMax = 1.31934516480519792e+00;
constexpr double kPySheetSum = 1.67624672668030655e+02;

// potential_mesh_birkhoff_rott_velocity, vertex quadrature, direct sum.
constexpr double kPyVelocityMax = 7.14122311532192522e-01;
constexpr double kPyVelocityMin = 2.14190901248706378e-01;
constexpr double kPyVelocitySum = 6.80151725261897440e+01;
// The SIGNED components. A reversed cross product negates all three and leaves
// every magnitude above untouched.
constexpr double kPyVelocitySumX = -1.38090917397758552e+01;
constexpr double kPyVelocitySumY = 3.22212140594769920e+01;
constexpr double kPyVelocitySumZ = -5.06333363791781466e+01;

// potential_surface_riesz_scalar, vertex quadrature, direct sum. `min` is
// signed and is negative, so it pins the -1/(4 pi^2) prefactor's sign.
constexpr double kPyRieszMax = 2.27174976735775941e-01;
constexpr double kPyRieszMin = -2.27174976735775969e-01;
constexpr double kPyRieszSumAbs = 1.84520520838543689e+01;

//---------------------------------------------------------------------------//
// Tolerances.
//---------------------------------------------------------------------------//

/// The criterion's number, relative. Beatnik and the Python agree on the mesh
/// positions to the last bits (T1b), and T2b measured the derived operators at
/// `1e-15` or better, so `1e-13` leaves two decades for the ring's summation
/// order (R2) and the on-node reduction tree. **Do not loosen this** — see the
/// file header.
constexpr double kPyTolerance = 1.0e-13;

/// For a quantity that is exactly zero in exact arithmetic, at a scale of order
/// one. Chosen a priori, not from the measurement.
constexpr double kExactZeroTolerance = 1.0e-13;

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using range = Kokkos::RangePolicy<ExecSpace>;
    using br_type = Beatnik::BRSolverDirect<ExecSpace, MemSpace>;
    // The SOLVER's view types, not hand-rolled ones. Its `device_type` is
    // `Kokkos::Device<ExecSpace, MemSpace>`, and a `View<Real*, MemSpace>` is a
    // distinct and unrelated type that will not bind to its out-parameters.
    using vector_view = typename br_type::vector_view;
    using scalar_view = typename br_type::scalar_view;

    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    //-----------------------------------------------------------------------//
    // The mesh, and the structural facts every later number rests on.
    //-----------------------------------------------------------------------//
    mesh_type mesh( MPI_COMM_WORLD );
    const Real center[3] = { 0.0, 0.0, kCenterZ };
    mesh.generateIcosphere( kSubdivisions, kRadius, center );

    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
    BEATNIK_CHECK_EQ( rec, mesh.haloDepth(),
                      ( mesh_type::halo_depth ) );

    const int n_owned = mesh.ownedVertexCount();
    const int n_local = mesh.totalVertexCount();

    //-----------------------------------------------------------------------//
    // R9 DISCRIMINATOR 1 -- do the owned sets PARTITION the global sets?
    //
    // Summed with a plain MPI_Allreduce over `ownedXCount()` rather than read
    // from Tessera's `globalOwnedX`, deliberately: two independent paths to the
    // same number, and owned-versus-local is exactly what R9 turns on. This is
    // the precondition the owned-only source list needs, and R9 says to check
    // it rather than assume it. (T1c's template.)
    //-----------------------------------------------------------------------//
    {
        long long owned[3] = { n_owned, mesh.ownedEdgeCount(),
                               mesh.ownedFaceCount() };
        long long total[3] = { 0, 0, 0 };
        MPI_Allreduce( owned, total, 3, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        std::ostringstream os;
        os << "owned partition: sum over ranks V " << total[0] << " E "
           << total[1] << " F " << total[2] << "; this rank owns V " << owned[0]
           << " of local V " << n_local << " (ghost fraction "
           << ( n_local > 0 ? double( n_local - n_owned ) / double( n_local )
                            : 0.0 )
           << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, total[0], kVertices );
        BEATNIK_CHECK_EQ( rec, total[1], kEdges );
        BEATNIK_CHECK_EQ( rec, total[2], kFaces );
    }

    //-----------------------------------------------------------------------//
    // The source state. THE ORDERING HERE IS PART OF WHAT IS BEING TESTED.
    //
    // `updateSheetVector` is a one-ring stencil on the potential, so an owned
    // vertex on a partition boundary reads the potential at vertices it does
    // not own: the potential's ghosts must be current BEFORE the call. T2d's
    // RHS is what will do this in production; it does not exist yet, so the
    // ordering is written out explicitly here rather than inherited.
    //-----------------------------------------------------------------------//
    Beatnik::SurfaceState<ExecSpace, MemSpace> state(
        Beatnik::StateModel::Potential );
    state.initializeFields( mesh );

    auto pos = mesh.positions();
    auto phi = mesh.potential();
    const Real a0 = kA[0], a1 = kA[1], a2 = kA[2];

    // OWNED rows only, so the exchange below has something to do and is not
    // trivially a no-op that a broken ordering would survive.
    Kokkos::parallel_for(
        "test_seed_potential", range( 0, n_owned ), KOKKOS_LAMBDA( const int i ) {
            phi( i ) = a0 * pos( i, 0 ) + a1 * pos( i, 1 ) + a2 * pos( i, 2 );
        } );
    Kokkos::fence();

    mesh.haloExchange();

    // The exchange did what the precondition needs: every GHOST row now holds
    // its owner's value, which for this field is the analytic a.p. Checked over
    // the whole local range, so at one rank it is vacuously the owned check and
    // at six it is the real one.
    {
        Real worst = 0.0;
        Kokkos::parallel_reduce(
            "test_ghost_potential", range( 0, n_local ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                const Real want = a0 * pos( i, 0 ) + a1 * pos( i, 1 ) +
                                  a2 * pos( i, 2 );
                const Real e = Kokkos::fabs( phi( i ) - want );
                if ( e > m )
                    m = e;
            },
            Kokkos::Max<Real>( worst ) );
        Real global_worst = 0.0;
        MPI_Allreduce( &worst, &global_worst, 1, MPI_DOUBLE, MPI_MAX,
                       mesh.comm() );
        std::ostringstream os;
        os.precision( 17 );
        os << "potential after haloExchange: max|phi - a.p| over the whole "
              "local range "
           << global_worst;
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, global_worst <= kExactZeroTolerance );
    }

    Beatnik::MeshGeometry<ExecSpace, MemSpace> geom;
    // The WHOLE LOCAL face set, per DISTRIBUTED ASSEMBLY in
    // Beatnik_MeshGeometry.hpp.
    geom.compute( pos, n_local, mesh.faceVertices() );

    state.updateSheetVector( mesh, geom );

    //-----------------------------------------------------------------------//
    // The SOURCE, before the kernel. T2b's published numbers.
    //
    // Reduced over the OWNED range only and then globally: `updateSheetVector`
    // leaves ghost rows holding PARTIAL sums by construction (the face-loop
    // assembly is complete on owned vertices only), so a whole-local-range
    // reduction here would be wrong for a reason that has nothing to do with
    // this task. The quadrature reads owned rows for the same reason, which is
    // why nothing needs an exchange after this call.
    //-----------------------------------------------------------------------//
    {
        auto sheet = mesh.sheetVector();
        Real mag_max = 0.0, mag_sum = 0.0;
        Kokkos::parallel_reduce(
            "test_sheet_summary", range( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m, Real& s ) {
                const Real mag = Kokkos::sqrt( sheet( i, 0 ) * sheet( i, 0 ) +
                                               sheet( i, 1 ) * sheet( i, 1 ) +
                                               sheet( i, 2 ) * sheet( i, 2 ) );
                if ( mag > m )
                    m = mag;
                s += mag;
            },
            Kokkos::Max<Real>( mag_max ), mag_sum );

        Real global_max = 0.0, global_sum = 0.0;
        MPI_Allreduce( &mag_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                       mesh.comm() );
        MPI_Allreduce( &mag_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                       mesh.comm() );

        std::ostringstream os;
        os.precision( 17 );
        os << "source max|S| " << global_max << " (python " << kPySheetMax
           << "), sum|S| " << global_sum << " (python " << kPySheetSum << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_CLOSE( rec, global_max, kPySheetMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, global_sum, kPySheetSum, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // The quadrature.
    //
    // R9 DISCRIMINATOR 2 -- the global source count. This is the DIRECT
    // detector for the failure mode this task can introduce: a rule emitting
    // the whole local vertex set instead of the owned one makes this 162 at one
    // rank and several hundred at six, while every velocity number below moves
    // smoothly and plausibly. Asserted, not merely reported.
    //-----------------------------------------------------------------------//
    auto quadrature =
        Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
            Beatnik::SourceQuadrature::Vertex );
    BEATNIK_CHECK_TRUE( rec, quadrature != nullptr );
    BEATNIK_CHECK_EQ( rec, static_cast<int>( quadrature->kind() ),
                      static_cast<int>( Beatnik::SourceQuadrature::Vertex ) );

    {
        typename Beatnik::SourceQuadratureBase<ExecSpace,
                                               MemSpace>::point_view points;
        typename Beatnik::SourceQuadratureBase<ExecSpace, MemSpace>::
            strength_view strengths;
        quadrature->generate( mesh, geom, state, points, strengths );

        const long long local_sources =
            static_cast<long long>( points.extent( 0 ) );
        long long global_sources = 0;
        MPI_Allreduce( &local_sources, &global_sources, 1, MPI_LONG_LONG,
                       MPI_SUM, mesh.comm() );
        std::ostringstream os;
        os << "quadrature emitted " << local_sources << " local sources ("
           << n_owned << " owned of " << n_local << " local vertices), "
           << global_sources << " globally";
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, local_sources,
                          static_cast<long long>( n_owned ) );
        BEATNIK_CHECK_EQ( rec, global_sources, kVertices );
        BEATNIK_CHECK_EQ( rec, quadrature->pointCount( n_owned,
                                                       mesh.ownedFaceCount() ),
                          n_owned );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( strengths.extent( 0 ) ),
                          static_cast<long long>( n_owned ) );
    }

    //-----------------------------------------------------------------------//
    // THE EXIT CRITERION -- the induced velocity.
    //-----------------------------------------------------------------------//
    Beatnik::ZModelParams params;
    params.eps = kEps;
    params.blob_mode = Beatnik::KernelBlobMode::Length;
    params.br_sign = 1.0;
    params.source_quadrature = Beatnik::SourceQuadrature::Vertex;
    params.br_approximation = Beatnik::BRApproximation::Direct;

    br_type br( mesh.comm() );
    BEATNIK_CHECK_EQ( rec, static_cast<int>( br.kind() ),
                      static_cast<int>( Beatnik::BRApproximation::Direct ) );

    vector_view velocity( "velocity", n_owned );
    br.computeInterfaceVelocity( mesh, geom, state, *quadrature, params,
                                 velocity );

    {
        Real mag_max = 0.0, mag_min = 1.0e300, mag_sum = 0.0;
        Real sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        Kokkos::parallel_reduce(
            "test_velocity_summary", range( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m_max, Real& m_min, Real& m_sum,
                           Real& sx, Real& sy, Real& sz ) {
                const Real u[3] = { velocity( i, 0 ), velocity( i, 1 ),
                                    velocity( i, 2 ) };
                const Real mag =
                    Kokkos::sqrt( u[0] * u[0] + u[1] * u[1] + u[2] * u[2] );
                if ( mag > m_max )
                    m_max = mag;
                if ( mag < m_min )
                    m_min = mag;
                m_sum += mag;
                sx += u[0];
                sy += u[1];
                sz += u[2];
            },
            Kokkos::Max<Real>( mag_max ), Kokkos::Min<Real>( mag_min ), mag_sum,
            sum_x, sum_y, sum_z );

        Real g_max = 0.0, g_min = 0.0;
        MPI_Allreduce( &mag_max, &g_max, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        MPI_Allreduce( &mag_min, &g_min, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
        Real local_sums[4] = { mag_sum, sum_x, sum_y, sum_z };
        Real g_sums[4] = { 0, 0, 0, 0 };
        MPI_Allreduce( local_sums, g_sums, 4, MPI_DOUBLE, MPI_SUM,
                       mesh.comm() );

        // Reported to 17 digits at every rank count WHETHER OR NOT they pass:
        // that measurement is what lets a later reader tell a last-ulp
        // summation-order difference (R2) from a ghost-fraction-scaled one
        // (R9), and it is re-measured on every gate run.
        std::ostringstream os;
        os.precision( 17 );
        os << "velocity max|u| " << g_max << " (python " << kPyVelocityMax
           << "), min|u| " << g_min << " (python " << kPyVelocityMin
           << "), sum|u| " << g_sums[0] << " (python " << kPyVelocitySum << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "velocity SIGNED sum u = (" << g_sums[1] << ", " << g_sums[2]
            << ", " << g_sums[3] << ") (python " << kPyVelocitySumX << ", "
            << kPyVelocitySumY << ", " << kPyVelocitySumZ << ")";
        rec.note( os2.str() );

        BEATNIK_CHECK_CLOSE( rec, g_max, kPyVelocityMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_min, kPyVelocityMin, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_sums[0], kPyVelocitySum, kPyTolerance );
        // The three signed components. These are what a reversed delta x S
        // fails and the three magnitudes above do not.
        BEATNIK_CHECK_CLOSE( rec, g_sums[1], kPyVelocitySumX, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_sums[2], kPyVelocitySumY, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_sums[3], kPyVelocitySumZ, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // The surface Riesz scalar. Nothing calls this until
    // `--bernoulli-scalar-mode surface-riesz`, but it is on T2c's fill-in list
    // and it shares the ring, the kernel and the quadrature with the velocity
    // -- so leaving it unvalidated would leave the factored ring exercised by
    // one caller only. Its prefactor is -1/(4 pi^2) and it carries NO
    // `br_sign`; `min psi < 0` is what pins that sign.
    //-----------------------------------------------------------------------//
    scalar_view riesz( "riesz", n_owned );
    br.computeSurfaceRieszScalar( mesh, geom, state, *quadrature, params,
                                 riesz );
    {
        Real psi_max = -1.0e300, psi_min = 1.0e300, abs_sum = 0.0;
        Kokkos::parallel_reduce(
            "test_riesz_summary", range( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m_max, Real& m_min, Real& s ) {
                const Real v = riesz( i );
                if ( v > m_max )
                    m_max = v;
                if ( v < m_min )
                    m_min = v;
                s += Kokkos::fabs( v );
            },
            Kokkos::Max<Real>( psi_max ), Kokkos::Min<Real>( psi_min ),
            abs_sum );

        Real g_max = 0.0, g_min = 0.0, g_sum = 0.0;
        MPI_Allreduce( &psi_max, &g_max, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        MPI_Allreduce( &psi_min, &g_min, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
        MPI_Allreduce( &abs_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );

        std::ostringstream os;
        os.precision( 17 );
        os << "riesz max " << g_max << " (python " << kPyRieszMax << "), min "
           << g_min << " (python " << kPyRieszMin << "), sum|psi| " << g_sum
           << " (python " << kPyRieszSumAbs << ")";
        rec.note( os.str() );

        BEATNIK_CHECK_CLOSE( rec, g_max, kPyRieszMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_min, kPyRieszMin, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, g_sum, kPyRieszSumAbs, kPyTolerance );
        BEATNIK_CHECK_TRUE( rec, g_min < 0.0 );
    }

    //-----------------------------------------------------------------------//
    // A NEGATIVE CASE, and it is a real one.
    //
    // T1b's lesson: a check that has only ever seen agreeing data has not been
    // tested. Here the cheapest genuine negative is `--br-sign -1`, which by
    // the convention in Beatnik_BRSolverBase.hpp must negate the velocity
    // EXACTLY and must leave the Riesz scalar alone. Both halves are asserted;
    // the second is what catches a `br_sign` applied in the shared kernel
    // instead of on the velocity path only.
    //
    // THE TWO HALVES ARE NOT ASSERTED THE SAME WAY, AND THE ASYMMETRY IS A
    // MEASURED PROPERTY OF THE TWO PATHS RATHER THAN A WEAKENING
    // ------------------------------------------------------------------
    // The velocity IS bitwise: `generate` reads `mesh.sheetVector()`, which was
    // assembled once above, and does no reduction of its own, so both calls sum
    // exactly the same source list in exactly the same ring order and
    // `br_sign` is a sign-bit flip on a completed sum. Measured 0 differing
    // components at every rank count on both backends.
    //
    // The Riesz scalar is NOT, on HIP. `generateGradient` re-runs
    // `SurfaceOperators::surfaceGradient`, whose face-loop assembly uses
    // `Kokkos::atomic_add` -- documented as not bitwise reproducible under
    // DETERMINISM in Beatnik_MeshGeometry.hpp. Two identical calls therefore
    // produce last-bit-different gradients, and the first version of this check
    // demanded bitwise equality and failed on HIP at all six rank counts (15 to
    // 60 of 162 values differing) while passing on SERIAL at all six. That is
    // the atomic scatter, not a `br_sign` leak, and the discriminator is the
    // SIZE: a leak makes the difference exactly 2|psi| (relative 2.0), while an
    // atomic reordering makes it ~1e-16. So the claim is made at `1e-13`
    // relative to max|psi| -- thirteen decades below what the bug it exists to
    // catch would produce -- and the measured number is reported either way.
    //-----------------------------------------------------------------------//
    {
        Beatnik::ZModelParams flipped = params;
        flipped.br_sign = -1.0;

        vector_view velocity_flipped( "velocity_flipped", n_owned );
        br.computeInterfaceVelocity( mesh, geom, state, *quadrature, flipped,
                                     velocity_flipped );
        scalar_view riesz_flipped( "riesz_flipped", n_owned );
        br.computeSurfaceRieszScalar( mesh, geom, state, *quadrature, flipped,
                                      riesz_flipped );

        int velocity_bad = 0;
        Real riesz_drift = 0.0;
        Real riesz_scale = 0.0;
        Kokkos::parallel_reduce(
            "test_br_sign", range( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, int& vb, Real& drift, Real& scale ) {
                for ( int d = 0; d < 3; ++d )
                    if ( velocity_flipped( i, d ) != -velocity( i, d ) )
                        ++vb;
                const Real e =
                    Kokkos::fabs( riesz_flipped( i ) - riesz( i ) );
                if ( e > drift )
                    drift = e;
                const Real m = Kokkos::fabs( riesz( i ) );
                if ( m > scale )
                    scale = m;
            },
            velocity_bad, Kokkos::Max<Real>( riesz_drift ),
            Kokkos::Max<Real>( riesz_scale ) );

        int g_velocity_bad = 0;
        MPI_Allreduce( &velocity_bad, &g_velocity_bad, 1, MPI_INT, MPI_SUM,
                       mesh.comm() );
        Real local_max[2] = { riesz_drift, riesz_scale };
        Real g_max[2] = { 0, 0 };
        MPI_Allreduce( local_max, g_max, 2, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        const double relative_drift =
            ( g_max[1] > Real( 0 ) ) ? double( g_max[0] / g_max[1] ) : 0.0;

        std::ostringstream os;
        os.precision( 17 );
        os << "br_sign = -1: velocity components not exactly negated "
           << g_velocity_bad << " (must be 0); riesz max|dpsi| " << g_max[0]
           << ", relative to max|psi| " << relative_drift
           << " (a br_sign leak would be 2.0)";
        rec.note( os.str() );

        BEATNIK_CHECK_EQ( rec, g_velocity_bad, 0 );
        BEATNIK_CHECK_TRUE( rec, relative_drift <= kPyTolerance );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_BirkhoffRott" );
        try
        {
            // BEATNIK_TEST_EXEC_SPACE is defined by the per-backend shim
            // tests/CMakeLists.txt generates, so the target name's `_SERIAL` /
            // `_HIP` suffix means what the gate's `-R <backend>` filter assumes
            // it means. Defaulting to the default space keeps the file
            // compilable on its own.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
        }
        catch ( const std::exception& e )
        {
            // Most likely a BEATNIK_NOT_IMPLEMENTED from a stub on a path this
            // test did not expect to touch. Reported as a named failure rather
            // than allowed to abort, so the tally line still appears in the log.
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();

    // ONE VERDICT ACROSS THE RANKS. Every rank printed its own tally above, so
    // the log names which rank failed; MPI_MAX then makes any rank's failure the
    // job's failure. Without this a launcher that reports only rank 0's status
    // would report success for a run that failed elsewhere.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
