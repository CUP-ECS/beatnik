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
 * @file Beatnik_Test_T2bOperators.cpp
 * @brief `unit`-tier test for T2b: the seven surface differential operators and
 *        `SurfaceState::updateSheetVector`.
 *
 * THIS IS T2b's EXIT CRITERION, MECHANIZED.
 * `tasks/framework.md` states it as: *a unit test (tier `unit`) confirming, on
 * the default icosphere, that `meanCurvatureNormal` returns
 * `≈ -2/R · n̂_out` (the Meyer-Desbrun-Schroeder-Barr identity — the definitive
 * sign check) and that `surfaceGradient` of a linear function reproduces its
 * tangential projection to `1e-12`.*
 *
 * THE SECOND HALF OF THE CRITERION CANNOT HOLD AS LITERALLY WRITTEN, AND THAT
 * IS A PROPERTY OF THE OPERATOR AND NOT OF THIS PORT
 * --------------------------------------------------------------------------
 * `surfaceGradient` is an **area-weighted average of the per-face in-plane
 * gradients, projected onto the vertex tangent plane**. For a linear
 * \f$\phi = a\cdot p\f$ the *face* gradient is exactly \f$P_f a\f$ — and that
 * exactness is checked below, at `1e-13` absolute. But the average of
 * \f$P_f a\f$ over faces that are tilted relative to each other is not
 * \f$P_v a\f$, and projecting it afterwards does not repair the difference. The
 * discrepancy is \f$O((h/R)^2)\f$, i.e. ~8% of \f$|a|\f$ at subdivision 2, and
 * no correct implementation makes it `1e-12`.
 *
 * The read-only Python reference agrees, and quantifies it exactly:
 *
 *     surface_gradient  max|g - P_v a| = 2.34168993652347258e-02
 *
 * on the default icosphere with `a = (0.3, -0.7, 1.1)`. So `1e-12` is used here
 * for the strongest statement that *is* true and that a wrong implementation
 * would fail — **Beatnik reproduces the Python's `surface_gradient` of the same
 * linear function on the same mesh, to `1e-12` relative**, that discrepancy
 * scalar included — plus the exact half of the claim, that the result carries no
 * normal component (`1e-13` absolute; the Python measures `2.2e-16`). Both
 * checks are tighter than the criterion in the respects where tightness is
 * meaningful, and neither was loosened to pass. See `tasks/framework.md` T2b's
 * completion note.
 *
 * HOW THE REFERENCE NUMBERS WERE OBTAINED
 * ---------------------------------------
 * Every `kPy*` literal below was computed by calling the **read-only** reference
 * (`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/mesh_solver.py`)
 * directly on `mesh.icosphere_mesh( subdivisions=2, radius=0.25,
 * center=(0,0,0.25) )` — the Python's own defaults, i.e. the same mesh T1a's
 * gold file describes. They are **order-invariant summary scalars** (a max, a
 * min, a sum of magnitudes) precisely so that this test does not have to match
 * Beatnik's vertex numbering to the Python's, which T1b established differs.
 *
 * They are hard-coded and **must not be adjusted to make this test pass**: a
 * mismatch at `1e-12` is a real disagreement with the reference and is the
 * finding. The same rule as `Beatnik_Test_MeshGeometry.cpp`.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp`.
 */

#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The mesh. The Python's defaults, from
// `run_adaptive_mesh_bubble.py::parse_args`: subdivision 2, radius 0.25,
// centre (0, 0, 0.25). Same mesh as T1b's test and T1a's gold file.
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;
constexpr int kVertices = 162;

/// The linear test function, \f$\phi(p) = a\cdot p\f$. Deliberately a generic
/// direction: an axis-aligned `a` would leave the icosahedron's own symmetry
/// planes to cancel errors this one exposes.
constexpr Real kA[3] = { 0.3, -0.7, 1.1 };

//---------------------------------------------------------------------------//
// Reference values from the read-only Python, on the mesh above. See the file
// header for how they were produced and why they must not be edited.
//---------------------------------------------------------------------------//

// _face_scalar_gradient of phi, against the exact per-face projection P_f a.
// Python: 1.27675647831893002e-15 -- i.e. exact to rounding.
constexpr double kPyFaceGradientMaxError = 1.27675647831893002e-15;

// surface_gradient of phi.
constexpr double kPySurfaceGradientMax = 1.31934516480519815e+00;
constexpr double kPySurfaceGradientSum = 1.67624672668030627e+02;
constexpr double kPySurfaceGradientProjectionError = 2.34168993652347258e-02;
constexpr double kPySurfaceGradientNormalPart = 2.22044604925031308e-16;

// mean_curvature_normal. 2/R = 8 exactly at R = 0.25; the spread below is the
// valence-5 versus valence-6 vertices of the icosphere.
constexpr double kPyCurvatureMagMin = 7.91848082705875189e+00;
constexpr double kPyCurvatureMagMax = 9.07600952626480151e+00;
constexpr double kPyCurvatureMagSum = 1.29887789652816309e+03;
constexpr double kPyCurvatureMagMean = 8.01776479338372283e+00;
/// max |cos(angle between Delta_LB x and the OUTWARD radial direction) + 1|,
/// i.e. how far from exactly antiparallel the worst vertex is.
constexpr double kPyCurvatureCosPlusOne = 1.39072087439129355e-04;

// cotangent_laplacian_scalars of phi, and its energy quadratic form.
constexpr double kPyCotLaplacianMax = 1.18326447314336871e+01;
constexpr double kPyCotLaplacianQuadForm = -9.19601207727919090e-01;

// graph_laplacian_scalars of phi and graph_laplacian_vectors of the positions.
constexpr double kPyGraphLaplacianMax = 1.68330765456009245e-02;
constexpr double kPyGraphLaplacianSum = 1.33470857333871074e+00;
constexpr double kPyGraphLaplacianVecMax = 1.26637503746173247e-02;
constexpr double kPyGraphLaplacianVecSum = 1.95937462564235254e+00;

// potential_sheet_vector, S = -n x grad_s(phi).
constexpr double kPySheetVectorMax = 1.31934516480519792e+00;
constexpr double kPySheetVectorSum = 1.67624672668030655e+02;

//---------------------------------------------------------------------------//
// Tolerances.
//---------------------------------------------------------------------------//

/// Against a Python reference scalar, relative. The criterion's number. Beatnik
/// and the Python agree on the mesh positions to the last bits (T1b: the
/// icosphere tables are the same literals and differ only by a
/// reciprocal-multiply), so a derived summary scalar should agree to ~1e-15;
/// 1e-12 leaves three decades of margin for the non-reproducible atomic scatter
/// (DETERMINISM in `Beatnik_MeshGeometry.hpp`). Do NOT loosen this.
constexpr double kPyTolerance = 1.0e-12;

/// For a Python reference formed by cancellation near a limit: `cos + 1` with
/// `cos` ~ -0.99986 loses about four significant digits, so the two codes
/// cannot agree on it to better than ~1e-12 even if they agree on `cos` to
/// 1e-16. Three further decades of margin.
constexpr double kPyCancellationTolerance = 1.0e-9;

/// For quantities that are exactly zero in exact arithmetic and are computed at
/// a scale of order 1 (|a| = 1.34, |phi| < 0.5). A machine-precision bound with
/// room for an atomic scatter, chosen a priori and not from the measurement.
constexpr double kExactZeroTolerance = 1.0e-13;

/// For the ANALYTIC half of criterion 1: the discrete mean curvature converges
/// to 2/R at O((h/R)^2), and h/R = 0.276 at subdivision 2, so ~7.6% is the
/// expected size of the discretization error. 10% is that bound rounded up. It
/// is a statement about the discretization, NOT a comparison tolerance -- the
/// per-vertex extremes are pinned against the Python at 1e-12 instead, which is
/// why nothing here needs a tolerance fitted to the icosphere's valence spread.
constexpr double kDiscretizationTolerance = 1.0e-1;

//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using Ops = Beatnik::SurfaceOperators;
    using scalar_view = Kokkos::View<Real*, MemSpace>;
    using vector_view = Kokkos::View<Real* [3], MemSpace>;
    using range = Kokkos::RangePolicy<ExecSpace>;

    int comm_size = 1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    // Every check below is a WHOLE-SURFACE summary scalar compared against a
    // whole-surface Python reference, so this test is meaningful at one rank
    // only -- which is what the `unit` tier registers (see
    // tests/unit_tests/CMakeLists.txt). Asserted rather than silently skipped:
    // a version of this that quietly reported PASS after checking nothing at
    // four ranks would be worse than one that fails.
    BEATNIK_CHECK_EQ( rec, comm_size, 1 );
    if ( comm_size != 1 )
    {
        rec.note( "not run: the T2b summary scalars are whole-surface "
                  "statements and this tier is registered at one rank" );
        return;
    }

    mesh_type mesh( MPI_COMM_WORLD );
    const Real center[3] = { 0.0, 0.0, kCenterZ };
    mesh.generateIcosphere( kSubdivisions, kRadius, center );

    const int nv = mesh.totalVertexCount();
    const int nf = mesh.totalFaceCount();
    BEATNIK_CHECK_EQ( rec, nv, kVertices );

    auto pos = mesh.positions();
    auto face_verts = mesh.faceVertices();
    auto one_ring = mesh.vertexOneRing();

    Beatnik::MeshGeometry<ExecSpace, MemSpace> geom;
    // The whole local face set, per DISTRIBUTED ASSEMBLY.
    geom.compute( pos, nv, face_verts );
    auto vn = geom.vertex_normal;
    auto va = geom.vertex_area;

    // The linear test field, phi = a . p, over the whole local range.
    scalar_view phi( "phi", nv );
    const Real a0 = kA[0], a1 = kA[1], a2 = kA[2];
    Kokkos::parallel_for(
        "test_linear_field", range( 0, nv ), KOKKOS_LAMBDA( const int i ) {
            phi( i ) = a0 * pos( i, 0 ) + a1 * pos( i, 1 ) + a2 * pos( i, 2 );
        } );
    Kokkos::fence();

    // The exact analytic outward normal. Every icosphere vertex lies exactly on
    // the sphere, so (p - c)/|p - c| is the true outward normal there and is
    // independent of anything under test -- which is what makes it usable as the
    // reference direction for the curvature sign check.
    vector_view radial( "radial", nv );
    const Real cz = kCenterZ;
    Kokkos::parallel_for(
        "test_radial", range( 0, nv ), KOKKOS_LAMBDA( const int i ) {
            Real r[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) - cz };
            const Real len =
                Kokkos::sqrt( r[0] * r[0] + r[1] * r[1] + r[2] * r[2] );
            for ( int d = 0; d < 3; ++d )
                radial( i, d ) = ( len > Real( 0 ) ) ? r[d] / len : Real( 0 );
        } );
    Kokkos::fence();

    //-----------------------------------------------------------------------//
    // Group 1 -- faceScalarGradient of a linear function is EXACT per face.
    //
    // The one part of the gradient chain that is exact rather than
    // second-order, so it is checked as such: max |g_f - P_f a| against zero at
    // machine precision, not against a tolerance.
    //-----------------------------------------------------------------------//
    vector_view face_gradient( "face_gradient", nf );
    Ops::faceScalarGradient( pos, face_verts, phi, face_gradient );
    {
        auto fn = geom.face_normal;
        Real worst = 0.0;
        Kokkos::parallel_reduce(
            "test_face_gradient_exact", range( 0, nf ),
            KOKKOS_LAMBDA( const int f, Real& m ) {
                // P_f a = a - (a . n_f) n_f.
                const Real an = a0 * fn( f, 0 ) + a1 * fn( f, 1 ) +
                                a2 * fn( f, 2 );
                const Real e0 = face_gradient( f, 0 ) - ( a0 - an * fn( f, 0 ) );
                const Real e1 = face_gradient( f, 1 ) - ( a1 - an * fn( f, 1 ) );
                const Real e2 = face_gradient( f, 2 ) - ( a2 - an * fn( f, 2 ) );
                const Real e = Kokkos::sqrt( e0 * e0 + e1 * e1 + e2 * e2 );
                if ( e > m )
                    m = e;
            },
            Kokkos::Max<Real>( worst ) );
        std::ostringstream os;
        os.precision( 17 );
        os << "faceScalarGradient max|g_f - P_f a| " << worst << " (python "
           << kPyFaceGradientMaxError << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, worst <= kExactZeroTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 2 -- surfaceGradient. CRITERION 2.
    //
    // Three order-invariant summaries against the Python at 1e-12, plus the
    // exact statement (no normal component). See the file header for why the
    // criterion's literal "reproduces its tangential projection to 1e-12" is
    // realized as "reproduces the reference's discrepancy from that projection
    // to 1e-12" rather than as a 1e-12 bound on the discrepancy itself.
    //-----------------------------------------------------------------------//
    vector_view gradient( "gradient", nv );
    Ops::surfaceGradient( pos, face_verts, phi, vn, gradient );
    {
        Real mag_max = 0.0, mag_sum = 0.0, proj_err = 0.0, normal_part = 0.0;
        Kokkos::parallel_reduce(
            "test_surface_gradient", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m_max, Real& m_sum, Real& m_proj,
                           Real& m_norm ) {
                const Real g[3] = { gradient( i, 0 ), gradient( i, 1 ),
                                    gradient( i, 2 ) };
                const Real mag =
                    Kokkos::sqrt( g[0] * g[0] + g[1] * g[1] + g[2] * g[2] );
                if ( mag > m_max )
                    m_max = mag;
                m_sum += mag;

                // P_v a = a - (a . n_v) n_v, with the SAME vertex normal the
                // routine projected with.
                const Real an =
                    a0 * vn( i, 0 ) + a1 * vn( i, 1 ) + a2 * vn( i, 2 );
                const Real e0 = g[0] - ( a0 - an * vn( i, 0 ) );
                const Real e1 = g[1] - ( a1 - an * vn( i, 1 ) );
                const Real e2 = g[2] - ( a2 - an * vn( i, 2 ) );
                const Real e = Kokkos::sqrt( e0 * e0 + e1 * e1 + e2 * e2 );
                if ( e > m_proj )
                    m_proj = e;

                const Real gn = Kokkos::fabs( g[0] * vn( i, 0 ) +
                                              g[1] * vn( i, 1 ) +
                                              g[2] * vn( i, 2 ) );
                if ( gn > m_norm )
                    m_norm = gn;
            },
            Kokkos::Max<Real>( mag_max ), mag_sum, Kokkos::Max<Real>( proj_err ),
            Kokkos::Max<Real>( normal_part ) );

        std::ostringstream os;
        os.precision( 17 );
        os << "surfaceGradient max|g| " << mag_max << " (python "
           << kPySurfaceGradientMax << "), sum|g| " << mag_sum << " (python "
           << kPySurfaceGradientSum << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "surfaceGradient max|g - P_v a| " << proj_err << " (python "
            << kPySurfaceGradientProjectionError << "), max|g.n| "
            << normal_part << " (python " << kPySurfaceGradientNormalPart
            << ")";
        rec.note( os2.str() );

        BEATNIK_CHECK_CLOSE( rec, mag_max, kPySurfaceGradientMax,
                             kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mag_sum, kPySurfaceGradientSum,
                             kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, proj_err,
                             kPySurfaceGradientProjectionError, kPyTolerance );
        // The exact half: the projection at the end of the routine leaves no
        // normal component. Dropping that projection is the plausible-looking
        // bug this catches, and it would put a spurious normal circulation in
        // the sheet vector.
        BEATNIK_CHECK_TRUE( rec, normal_part <= kExactZeroTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 3 -- meanCurvatureNormal. CRITERION 1, the definitive sign check.
    //
    // On a sphere of radius R the MDSB identity gives Delta_LB x = -2/R n_out,
    // so the vector points INWARD with magnitude 2/R = 8 at R = 0.25. Getting
    // the sign backwards inflates the bubble under surface tension and is the
    // one error that looks like physics.
    //-----------------------------------------------------------------------//
    vector_view curvature( "curvature", nv );
    Ops::meanCurvatureNormal( pos, face_verts, va, curvature );
    {
        int wrong_sign = 0;
        Real mag_min = 1.0e300, mag_max = 0.0, mag_sum = 0.0, cos_dev = 0.0;
        Kokkos::parallel_reduce(
            "test_mean_curvature", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, int& bad, Real& m_min, Real& m_max,
                           Real& m_sum, Real& m_cos ) {
                const Real h[3] = { curvature( i, 0 ), curvature( i, 1 ),
                                    curvature( i, 2 ) };
                const Real mag =
                    Kokkos::sqrt( h[0] * h[0] + h[1] * h[1] + h[2] * h[2] );
                const Real dot = h[0] * radial( i, 0 ) + h[1] * radial( i, 1 ) +
                                 h[2] * radial( i, 2 );
                // THE SIGN CHECK: strictly inward at every single vertex,
                // against the exact analytic outward normal.
                if ( !( dot < Real( 0 ) ) )
                    ++bad;
                if ( mag < m_min )
                    m_min = mag;
                if ( mag > m_max )
                    m_max = mag;
                m_sum += mag;
                if ( mag > Real( 0 ) )
                {
                    const Real dev = Kokkos::fabs( dot / mag + Real( 1 ) );
                    if ( dev > m_cos )
                        m_cos = dev;
                }
            },
            wrong_sign, Kokkos::Min<Real>( mag_min ), Kokkos::Max<Real>( mag_max ),
            mag_sum, Kokkos::Max<Real>( cos_dev ) );

        const Real mag_mean = mag_sum / static_cast<Real>( nv );
        std::ostringstream os;
        os.precision( 17 );
        os << "meanCurvatureNormal |H| min " << mag_min << " max " << mag_max
           << " mean " << mag_mean << " (2/R = " << ( 2.0 / kRadius ) << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "meanCurvatureNormal inward-sign violations " << wrong_sign
            << ", max|cos + 1| " << cos_dev << " (python "
            << kPyCurvatureCosPlusOne << ")";
        rec.note( os2.str() );

        BEATNIK_CHECK_EQ( rec, wrong_sign, 0 );
        // Antiparallel to the outward radial direction. Not exact -- the
        // one-ring of a valence-5 vertex is not centrally symmetric -- but the
        // reference says how far off, so that is what is compared to.
        BEATNIK_CHECK_TRUE( rec, cos_dev <= 1.0e-3 );
        BEATNIK_CHECK_CLOSE( rec, cos_dev, kPyCurvatureCosPlusOne,
                             kPyCancellationTolerance );
        // The analytic magnitude, at the discretization tolerance.
        BEATNIK_CHECK_CLOSE( rec, mag_mean, 2.0 / kRadius,
                             kDiscretizationTolerance );
        // The per-vertex extremes, against the reference rather than against a
        // tolerance fitted to the icosphere's valence spread.
        BEATNIK_CHECK_CLOSE( rec, mag_min, kPyCurvatureMagMin, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mag_max, kPyCurvatureMagMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mag_sum, kPyCurvatureMagSum, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mag_mean, kPyCurvatureMagMean, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 4 -- cotangentLaplacianScalar.
    //-----------------------------------------------------------------------//
    {
        // 4a. Of a constant field: identically zero, and EXACTLY so. Every
        //     contribution is w * (c - c), which is a product with an exact
        //     zero however large the cotangent weight is -- so this is an
        //     equality and not a tolerance. It catches a stencil that forgot to
        //     difference (e.g. accumulating w * values(q) alone), which every
        //     non-constant test would pass.
        scalar_view constant( "constant", nv );
        Kokkos::deep_copy( constant, Real( 3.25 ) );
        scalar_view lap_const( "lap_const", nv );
        Ops::cotangentLaplacianScalar( pos, face_verts, constant, va,
                                       lap_const );
        Real worst = 0.0;
        Kokkos::parallel_reduce(
            "test_cot_const", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                const Real v = Kokkos::fabs( lap_const( i ) );
                if ( v > m )
                    m = v;
            },
            Kokkos::Max<Real>( worst ) );
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "cotangentLaplacianScalar(const) max|.| " << worst;
            rec.note( os.str() );
        }
        BEATNIK_CHECK_TRUE( rec, worst == 0.0 );
    }
    {
        // 4b. Of the linear field: against the Python, and the DISSIPATIVE
        //     SIGN. The energy quadratic form sum_i A_i phi_i (Delta phi)_i is
        //     negative definite for the correct sign convention, so
        //     mu * Delta_s phi with mu > 0 drives phi toward its local average.
        //     The opposite sign is anti-diffusive; this is the cheap way to
        //     detect it without integrating anything.
        scalar_view lap( "lap", nv );
        Ops::cotangentLaplacianScalar( pos, face_verts, phi, va, lap );
        Real worst = 0.0, quad = 0.0;
        Kokkos::parallel_reduce(
            "test_cot_linear", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m, Real& q ) {
                const Real v = Kokkos::fabs( lap( i ) );
                if ( v > m )
                    m = v;
                q += va( i ) * phi( i ) * lap( i );
            },
            Kokkos::Max<Real>( worst ), quad );
        std::ostringstream os;
        os.precision( 17 );
        os << "cotangentLaplacianScalar(linear) max|.| " << worst << " (python "
           << kPyCotLaplacianMax << "), sum A phi Lphi " << quad << " (python "
           << kPyCotLaplacianQuadForm << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_CLOSE( rec, worst, kPyCotLaplacianMax, kPyTolerance );
        BEATNIK_CHECK_TRUE( rec, quad < 0.0 );
        BEATNIK_CHECK_CLOSE( rec, quad, kPyCotLaplacianQuadForm, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 5 -- the two graph Laplacians, on the vertex one-ring CSR.
    //
    // These exist here to pin the T2b signature decision: the operators average
    // over the UNIQUE neighbour set, and a per-face scatter would visit every
    // interior neighbour twice. The Python's values are what discriminates, so
    // they are what is compared to.
    //-----------------------------------------------------------------------//
    {
        scalar_view constant( "constant", nv );
        Kokkos::deep_copy( constant, Real( 3.25 ) );
        scalar_view gl( "graph_lap", nv );
        Ops::graphLaplacianScalar( one_ring, constant, gl );
        scalar_view gl_phi( "graph_lap_phi", nv );
        Ops::graphLaplacianScalar( one_ring, phi, gl_phi );

        Real const_worst = 0.0, worst = 0.0, sum = 0.0;
        Kokkos::parallel_reduce(
            "test_graph_lap", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& mc, Real& m, Real& s ) {
                const Real c = Kokkos::fabs( gl( i ) );
                if ( c > mc )
                    mc = c;
                const Real v = Kokkos::fabs( gl_phi( i ) );
                if ( v > m )
                    m = v;
                s += v;
            },
            Kokkos::Max<Real>( const_worst ), Kokkos::Max<Real>( worst ), sum );
        std::ostringstream os;
        os.precision( 17 );
        os << "graphLaplacianScalar(const) max|.| " << const_worst
           << ", (linear) max|.| " << worst << " (python "
           << kPyGraphLaplacianMax << "), sum|.| " << sum << " (python "
           << kPyGraphLaplacianSum << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, const_worst == 0.0 );
        BEATNIK_CHECK_CLOSE( rec, worst, kPyGraphLaplacianMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, sum, kPyGraphLaplacianSum, kPyTolerance );
    }
    {
        // The vector form, applied to the positions -- which is the umbrella
        // smoothing vector, and the field the Python's own reference value was
        // taken on.
        vector_view glv( "graph_lap_vec", nv );
        Ops::graphLaplacianVector( one_ring, pos, glv );
        Real worst = 0.0, sum = 0.0;
        Kokkos::parallel_reduce(
            "test_graph_lap_vec", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m, Real& s ) {
                const Real mag = Kokkos::sqrt( glv( i, 0 ) * glv( i, 0 ) +
                                               glv( i, 1 ) * glv( i, 1 ) +
                                               glv( i, 2 ) * glv( i, 2 ) );
                if ( mag > m )
                    m = mag;
                s += mag;
            },
            Kokkos::Max<Real>( worst ), sum );
        std::ostringstream os;
        os.precision( 17 );
        os << "graphLaplacianVector(x) max|.| " << worst << " (python "
           << kPyGraphLaplacianVecMax << "), sum|.| " << sum << " (python "
           << kPyGraphLaplacianVecSum << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_CLOSE( rec, worst, kPyGraphLaplacianVecMax,
                             kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, sum, kPyGraphLaplacianVecSum, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 6 -- projectTangent leaves no normal component.
    //
    // Applied to the CONSTANT field a, whose normal component is large and
    // varies over the sphere, so a routine that silently did nothing would fail
    // rather than pass on an already-tangential input.
    //-----------------------------------------------------------------------//
    {
        vector_view field( "field", nv );
        Kokkos::parallel_for(
            "test_fill_constant_vector", range( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                field( i, 0 ) = a0;
                field( i, 1 ) = a1;
                field( i, 2 ) = a2;
            } );
        Kokkos::fence();
        Ops::projectTangent( field, vn );

        Real normal_part = 0.0, mismatch = 0.0;
        Kokkos::parallel_reduce(
            "test_project_tangent", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m_norm, Real& m_mis ) {
                const Real gn =
                    Kokkos::fabs( field( i, 0 ) * vn( i, 0 ) +
                                  field( i, 1 ) * vn( i, 1 ) +
                                  field( i, 2 ) * vn( i, 2 ) );
                if ( gn > m_norm )
                    m_norm = gn;
                const Real an =
                    a0 * vn( i, 0 ) + a1 * vn( i, 1 ) + a2 * vn( i, 2 );
                const Real e0 = field( i, 0 ) - ( a0 - an * vn( i, 0 ) );
                const Real e1 = field( i, 1 ) - ( a1 - an * vn( i, 1 ) );
                const Real e2 = field( i, 2 ) - ( a2 - an * vn( i, 2 ) );
                const Real e = Kokkos::sqrt( e0 * e0 + e1 * e1 + e2 * e2 );
                if ( e > m_mis )
                    m_mis = e;
            },
            Kokkos::Max<Real>( normal_part ), Kokkos::Max<Real>( mismatch ) );
        std::ostringstream os;
        os.precision( 17 );
        os << "projectTangent max|v.n| " << normal_part << ", max|v - P_v a| "
           << mismatch;
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, normal_part <= kExactZeroTolerance );
        BEATNIK_CHECK_TRUE( rec, mismatch <= kExactZeroTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 7 -- SurfaceState::updateSheetVector, S = -n x grad_s(phi).
    //
    // The magnitudes against the Python, plus the sign, which is what the
    // declaration says is load-bearing. Since grad is tangential,
    // (grad x S) . n = -|grad|^2 identically for the correct sign and
    // +|grad|^2 for the wrong one -- so one signed identity pins the direction
    // of rotation within the tangent plane, which a magnitude check cannot see.
    //-----------------------------------------------------------------------//
    {
        auto mesh_phi = mesh.potential();
        Kokkos::parallel_for(
            "test_seed_potential", range( 0, nv ),
            KOKKOS_LAMBDA( const int i ) { mesh_phi( i ) = phi( i ); } );
        Kokkos::fence();

        Beatnik::SurfaceState<ExecSpace, MemSpace> state(
            Beatnik::StateModel::Potential );
        state.updateSheetVector( mesh, geom );
        auto sheet = mesh.sheetVector();

        Real mag_max = 0.0, mag_sum = 0.0, normal_part = 0.0, rot_err = 0.0;
        Real rot_max = -1.0e300;
        Kokkos::parallel_reduce(
            "test_sheet_vector", range( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m_max, Real& m_sum, Real& m_norm,
                           Real& m_rot_err, Real& m_rot_max ) {
                const Real s[3] = { sheet( i, 0 ), sheet( i, 1 ),
                                    sheet( i, 2 ) };
                const Real mag =
                    Kokkos::sqrt( s[0] * s[0] + s[1] * s[1] + s[2] * s[2] );
                if ( mag > m_max )
                    m_max = mag;
                m_sum += mag;
                const Real sn = Kokkos::fabs( s[0] * vn( i, 0 ) +
                                              s[1] * vn( i, 1 ) +
                                              s[2] * vn( i, 2 ) );
                if ( sn > m_norm )
                    m_norm = sn;

                const Real g[3] = { gradient( i, 0 ), gradient( i, 1 ),
                                    gradient( i, 2 ) };
                const Real cr[3] = { g[1] * s[2] - g[2] * s[1],
                                     g[2] * s[0] - g[0] * s[2],
                                     g[0] * s[1] - g[1] * s[0] };
                const Real rot = cr[0] * vn( i, 0 ) + cr[1] * vn( i, 1 ) +
                                 cr[2] * vn( i, 2 );
                const Real g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                const Real e = Kokkos::fabs( rot + g2 );
                if ( e > m_rot_err )
                    m_rot_err = e;
                if ( rot > m_rot_max )
                    m_rot_max = rot;
            },
            Kokkos::Max<Real>( mag_max ), mag_sum,
            Kokkos::Max<Real>( normal_part ), Kokkos::Max<Real>( rot_err ),
            Kokkos::Max<Real>( rot_max ) );

        std::ostringstream os;
        os.precision( 17 );
        os << "updateSheetVector max|S| " << mag_max << " (python "
           << kPySheetVectorMax << "), sum|S| " << mag_sum << " (python "
           << kPySheetVectorSum << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "updateSheetVector max|S.n| " << normal_part
            << ", max|(g x S).n + |g|^2| " << rot_err << ", max (g x S).n "
            << rot_max;
        rec.note( os2.str() );

        BEATNIK_CHECK_CLOSE( rec, mag_max, kPySheetVectorMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mag_sum, kPySheetVectorSum, kPyTolerance );
        BEATNIK_CHECK_TRUE( rec, normal_part <= kExactZeroTolerance );
        // The sign. `rot_max < 0` alone would catch a flipped sign; the
        // identity also catches a rotation that is not a rotation.
        BEATNIK_CHECK_TRUE( rec, rot_max < 0.0 );
        BEATNIK_CHECK_TRUE( rec, rot_err <= kExactZeroTolerance );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_T2bOperators" );
        try
        {
            // One binary on the default execution space, no backend suffix --
            // the `unit` tier's convention; see tests/unit_tests/CMakeLists.txt.
            using ExecSpace = Kokkos::DefaultExecutionSpace;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
        }
        catch ( const std::exception& e )
        {
            // Most likely a BEATNIK_NOT_IMPLEMENTED from a stub on a path this
            // test did not expect to touch. Reported as a named failure so the
            // tally line still appears in the log.
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();
    MPI_Finalize();
    return rc;
}
