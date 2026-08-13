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
 * @file Beatnik_SurfaceState.hpp
 * @brief The evolved solution state on the interface, in either of the two
 *        formulations the reference supports.
 *
 * THE TWO STATE MODELS
 * --------------------
 * Port of mesh_solver.py::MeshZModelState (lines 59-119) and
 * ::MeshPotentialZModelState (lines 122-192)
 *
 * **`Potential` (the default).** The evolved unknown is the scalar potential
 * jump \f$\phi\f$ at vertices. Directed edge differences
 * \f$\phi_j - \phi_i\f$ *are* the edge-integrated circulation 1-form, so
 * circulation is exactly conserved by construction on every closed loop of the
 * mesh — the formulation is discretely curl-free. The sheet vector is derived:
 * \f[
 *   S \;=\; -\,\hat n \times \nabla_s \phi .
 * \f]
 * \f$\phi\f$ is defined only up to an additive constant, which is pinned by
 * subtracting its area-weighted mean, both on construction and from its time
 * derivative.
 *
 * **`SheetVector`.** The evolved unknown is the tangential vector density
 * \f$S\f$ itself. In a local chart it corresponds to
 * \f$(w_2 z_a - w_1 z_b)/\|z_a\times z_b\|\f$, i.e. the structured-grid model's
 * \f$w\f$ written intrinsically, so the Birkhoff-Rott source is \f$S\,dA\f$.
 * Nothing constrains \f$S\f$ to remain curl-free, so this model can develop
 * spurious circulation that the potential model cannot. It is retained because
 * it is the direct descendant of the structured-grid `solver.py` formulation
 * and is the fallback when the potential model's re-centring interferes with a
 * diagnostic.
 *
 * WHY THIS IS ONE CLASS AND NOT TWO
 * ---------------------------------
 * The Python has two dataclasses with heavy duplication, and the driver
 * duck-types over `hasattr(state, "potential")` at a dozen call sites. Here the
 * model is a runtime tag, so the RHS and the integrator are not templated on
 * the model — which would double the instantiations of every Kokkos kernel for
 * no benefit, since the model is fixed for the whole run.
 *
 * THE FIELDS LIVE IN THE MESH — THIS CLASS HOLDS NO STORAGE
 * --------------------------------------------------------
 * **T1c CHANGE, and it is the one M1 booked and deferred to here.** The pre-T1c
 * revision declared three `Kokkos::View`s of its own for the potential, the
 * sheet vector and the material position. Under the M1 vertex user field pack
 * those three *are* slots in Tessera's vertex AoSoA
 * (`Beatnik::VertexFieldId`), and M1 recorded why a Beatnik-side copy cannot
 * work: `refine()` interpolates only fields it owns, `migrate()` ships only
 * tuples it owns, and `haloExchange()` syncs only the pack. A per-vertex field
 * held outside the mesh is therefore **silently dropped by refinement and
 * silently stale after migration** — and, before T1c, those three views were
 * simply never allocated at all.
 *
 * So every accessor is gone and every method below takes the mesh:
 *
 * | Was | Now |
 * | --- | --- |
 * | `potential()` | `mesh.potential()` |
 * | `sheetVector()` | `mesh.sheetVector()` |
 * | `materialPosition()` | `mesh.materialPosition()` |
 * | `resize( vertex_count )` | `initializeFields( mesh )` |
 * | `remap( edit )` | **deleted** |
 *
 * `resize` becomes `initializeFields` because Tessera allocates and only
 * *initialization* is left to do; `remap` is deleted because Tessera transfers
 * the pack itself.
 *
 * What remains here is the state *model* and the operations that are specific
 * to it. The class is deliberately still a class rather than a free-function
 * namespace: `Solver` holds one and hands it around, and a later task may want
 * per-model scratch that genuinely is Beatnik's.
 */

#ifndef BEATNIK_SURFACESTATE_HPP
#define BEATNIK_SURFACESTATE_HPP

#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <limits>
#include <utility>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief The state model, and the operations that depend on which one is in
 *        force. **Holds no per-vertex storage** — see the file header.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class SurfaceState
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /// @param model Which unknown is evolved. Fixed for the run.
    explicit SurfaceState( StateModel model )
        : _model( model )
    {
    }

    /// Which unknown is evolved.
    StateModel model() const { return _model; }

    /**
     * @brief Which npz key each `VertexFieldId` slot carries, in slot order.
     *
     * The declaration `CheckpointIO::write` emits as
     * `/beatnik/vertex_field_names` and `compare_output.py` cross-checks its
     * `FIELD_MAP` against. It lives here rather than in the IO adapter because
     * it is a statement about *the state*, and the whole point of the
     * cross-check is that the writer and the comparator name the slots
     * independently — see the `u0`/`u1`/`u2` hazard in
     * `Beatnik_IOInterface.hpp`.
     *
     * Order must match `Beatnik::VertexFieldId` and the schema table in that
     * header. `VertexFieldId::Count` is asserted against it at the use site.
     */
    static constexpr const char* vertex_field_names[3] = {
        "potential", "sheet_vector", "remesh_material_position" };

    /**
     * @brief Zero the three mesh-resident vertex fields.
     *
     * Port of mesh_solver.py::sphere_potential_mesh_state (lines 355-361) —
     * `potential=np.zeros(surface.vertex_count)`
     *
     * **T1c CHANGE — this replaces `resize( vertex_count )`.** Under the M1
     * field pack Tessera owns the allocation (the vertex AoSoA is sized by
     * `buildIcosphere` / `distribute` / `refine`), so there is nothing left for
     * Beatnik to allocate and nothing that a vertex count would be used for.
     * What *is* left is initialization: a freshly built mesh's user fields are
     * uninitialized storage, so reading one before it is written is a genuine
     * bug that would show up as a plausible-looking non-zero potential.
     *
     * Written over the **whole local range** (owned + ghost), not just the
     * owned one, so the ghost rows are defined before the first
     * `haloExchange()` rather than holding garbage that a kernel might read.
     *
     * @warning **Cold start only. Do NOT call this after a mesh edit.** The
     *          pre-T1c `resize` was documented as "called after every mesh
     *          edit", which was correct when Beatnik owned the storage and
     *          reallocation was Beatnik's job. It is now actively wrong:
     *          `refine()` and `splitEdges()` interpolate the pack through the
     *          `RefinePolicy` and `migrate()` ships it, so zeroing afterwards
     *          would **destroy the solution**. Nothing needs calling after an
     *          edit; that is the whole benefit of the fields being in the mesh.
     */
    template <class MeshType>
    void initializeFields( MeshType& mesh ) const
    {
        auto phi = mesh.potential();
        auto sheet = mesh.sheetVector();
        auto material = mesh.materialPosition();
        const int n = mesh.totalVertexCount();
        Kokkos::parallel_for(
            "beatnik_state_initialize_fields",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                phi( i ) = Real( 0 );
                for ( int d = 0; d < 3; ++d )
                {
                    sheet( i, d ) = Real( 0 );
                    material( i, d ) = Real( 0 );
                }
            } );
        Kokkos::fence();
    }

    /**
     * @brief Recompute the sheet vector from the potential.
     *
     * Port of mesh_solver.py::potential_sheet_vector (lines 364-367)
     *
     * \f[
     *   S \;=\; -\,\hat n_v \times \nabla_s \phi
     * \f]
     * with \f$\nabla_s\phi\f$ the per-vertex surface gradient
     * (`SurfaceOperators::surfaceGradient`) and \f$\hat n_v\f$ the
     * area-weighted vertex normal.
     *
     * **The minus sign is not cosmetic.** \f$-\hat n\times\nabla_s\phi\f$ is a
     * 90-degree rotation of the gradient within the tangent plane; the opposite
     * sign rotates the other way and reverses the induced circulation, which
     * makes the bubble roll up inward instead of outward. `--br-sign` exists to
     * flip the *whole* induced velocity if the overall orientation convention
     * turns out inverted; it is not a substitute for getting this sign right.
     *
     * A no-op under `StateModel::SheetVector`.
     *
     * **Implemented at T2b**, which is where `SurfaceOperators::surfaceGradient`
     * landed — T1c left this throwing because its body *is* that call and
     * because at 0 timesteps the sheet vector is never read. Written over the
     * **whole local range**, so ghost rows are consistent with their owners
     * without an exchange: `surfaceGradient` assembles from the whole local face
     * set, so a *ghost* vertex's gradient is a partial sum and its sheet vector
     * is correspondingly partial — but the halo-exchange-free consumers
     * (`Beatnik_SourceQuadrature.hpp`, owned entities only) read owned rows, and
     * a caller that needs ghost rows exact follows with `haloExchange()`.
     *
     * @pre The **potential's** ghost values must be current — one
     *      `mesh.haloExchange()` before this call. The gradient is a one-ring
     *      stencil, so an owned vertex on a partition boundary reads the
     *      potential at vertices it does not own. This is the *only* exchange
     *      needed: the face-loop assembly is complete on every owned vertex
     *      without a scatter-add (see DISTRIBUTED ASSEMBLY in
     *      `Beatnik_MeshGeometry.hpp`), and a second exchange does not widen the
     *      one-ring — the depth-2 halo built once in `SurfaceMesh` does (R8).
     *
     * **T2d — this is now a `const` member.** It writes the *mesh*, never the
     * state (which holds no storage at all), and the RHS receives its state by
     * `const&` because the source quadrature does. Every other method here was
     * already `const` for the same reason; this one was the outlier.
     */
    template <class MeshType, class GeometryType>
    void updateSheetVector( MeshType& mesh, const GeometryType& geometry ) const
    {
        if ( _model != StateModel::Potential )
            return;

        auto pos = mesh.positions();
        auto faces = mesh.faceVertices();
        auto phi = mesh.potential();
        auto sheet = mesh.sheetVector();
        const int n = mesh.totalVertexCount();

        Kokkos::View<Real* [3], memory_space> gradient(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_sheet_vector_gradient" ),
            n );
        SurfaceOperators::surfaceGradient( pos, faces, phi,
                                          geometry.vertex_normal, gradient );

        auto vn = geometry.vertex_normal;
        Kokkos::parallel_for(
            "beatnik_update_sheet_vector",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                // S = -n x grad_s(phi). The minus sign is load-bearing; see the
                // declaration. Written out rather than negating a cross-product
                // helper so the component order is inspectable.
                sheet( i, 0 ) = -( vn( i, 1 ) * gradient( i, 2 ) -
                                   vn( i, 2 ) * gradient( i, 1 ) );
                sheet( i, 1 ) = -( vn( i, 2 ) * gradient( i, 0 ) -
                                   vn( i, 0 ) * gradient( i, 2 ) );
                sheet( i, 2 ) = -( vn( i, 0 ) * gradient( i, 1 ) -
                                   vn( i, 1 ) * gradient( i, 0 ) );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Per-face sheet vector, from the per-face potential gradient.
     *
     * Port of mesh_solver.py::face_potential_sheet_vector (lines 370-374)
     *
     * \f$S_f = -\hat n_f \times \nabla_f\phi\f$, using the exact in-plane face
     * gradient rather than the averaged vertex one.
     *
     * Two distinct uses, and they are easy to conflate:
     *   1. The `Face` and `Triangle3` source quadratures build the BR source
     *      from this, not from the vertex sheet vector.
     *   2. `max_sheet_strength` takes the max over *both* the vertex and the
     *      face sheet vectors, because "the vertex-gradient diagnostic can miss
     *      triangle-scale potential jumps on irregular adaptive meshes. The
     *      face-gradient value is the quantity that sees those spikes first"
     *      (`run_adaptive_mesh_bubble.py:904-910`). Using only the vertex value
     *      lets the dt throttle miss the blow-up it exists to catch.
     *
     * **T2d — implemented for use 2 only.** `max_sheet_strength` needs it, and
     * `TimeIntegrator::chooseStepSize`'s `--max-sheet-dt-product` clamp needs
     * that, so leaving it throwing would have left a documented clamp that
     * aborts the run the moment it is enabled. Use 1 is the `Face` and
     * `Triangle3` quadratures, which are still T5-era or later.
     *
     * @param face_sheet `(Nf,3)`; **its extent chooses the face range**. Pass
     *        the OWNED range for anything that reduces to a global scalar
     *        (risk R9); the whole local range is right only for a per-face
     *        assembly, and there is none here.
     *
     * @pre \f$\phi\f$'s ghost values are current — the per-face gradient reads
     *      all three corners of a face this rank may only partly own.
     */
    template <class MeshType, class GeometryType, class VectorView>
    void faceSheetVector( MeshType& mesh, const GeometryType& geometry,
                          VectorView& face_sheet ) const
    {
        const int nf = static_cast<int>( face_sheet.extent( 0 ) );
        auto pos = mesh.positions();
        auto faces = mesh.faceVertices();
        auto phi = mesh.potential();
        auto fn = geometry.face_normal;
        auto out = face_sheet;
        Kokkos::parallel_for(
            "beatnik_face_sheet_vector",
            Kokkos::RangePolicy<execution_space>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                Real g[3];
                SurfaceOperators::faceGradient( pos, faces, phi, f, g );
                // S_f = -n_f x grad_f(phi). The same minus sign, and the same
                // component order, as the vertex form above.
                out( f, 0 ) = -( fn( f, 1 ) * g[2] - fn( f, 2 ) * g[1] );
                out( f, 1 ) = -( fn( f, 2 ) * g[0] - fn( f, 0 ) * g[2] );
                out( f, 2 ) = -( fn( f, 0 ) * g[1] - fn( f, 1 ) * g[0] );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Subtract the area-weighted mean of the potential.
     *
     * Port of mesh_solver.py::MeshPotentialZModelState.__post_init__
     * (lines 155-162)
     *
     * Pins the arbitrary additive constant. Applied on every state
     * construction; the RHS separately re-centres `potential_dot`. A no-op
     * under `SheetVector`.
     *
     * **T1c CHANGE — it takes the mesh as well as the geometry**, because the
     * potential it re-centres now lives in the mesh (see the file header) and
     * because the owned/ghost split it must respect is the mesh's.
     *
     * @note MPI. Collective, and **reduce-both-then-divide**. The two partial
     *       sums come from `areaWeightedMeanPartials` over the **owned** vertex
     *       range only (risk R9 — a ghost vertex is owned somewhere else), and
     *       both are reduced before the division. `areaWeightedMean`'s
     *       single-`Real` form is deliberately *not* used: an `allReduceSum` of
     *       per-rank means is not the global mean, and subtracting a per-rank
     *       mean would give the potential a piecewise-constant jump across
     *       every partition boundary — whose surface gradient, the sheet
     *       vector, would carry a delta function there. Invisible at one
     *       rank; see that function's note. The two sums are batched into one
     *       `MPI_Allreduce`.
     */
    template <class MeshType, class GeometryType>
    void centerPotential( MeshType& mesh, const GeometryType& geometry ) const
    {
        if ( _model != StateModel::Potential )
            return;

        auto phi = mesh.potential();
        const int n_owned = mesh.ownedVertexCount();
        const int n_local = mesh.totalVertexCount();

        // The owned range of both arrays. `phi` is a Cabana slice with no
        // extent, so the partials helper is handed plain owned-range views.
        Kokkos::View<Real*, memory_space> phi_owned(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_center_phi_owned" ),
            n_owned );
        Kokkos::parallel_for(
            "beatnik_center_gather",
            Kokkos::RangePolicy<execution_space>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) { phi_owned( i ) = phi( i ); } );
        Kokkos::fence();

        auto area_owned = Kokkos::subview( geometry.vertex_area,
                                           std::make_pair( 0, n_owned ) );

        Real weighted = 0, area = 0;
        SurfaceOperators::areaWeightedMeanPartials( phi_owned, area_owned,
                                                    weighted, area );

        // One collective for the numerator and the denominator together.
        Real pair[2] = { weighted, area };
        Real reduced[2] = { 0, 0 };
        MPI_Allreduce( pair, reduced, 2, MPI_DOUBLE, MPI_SUM, mesh.comm() );

        // The Python's fallback when the total area is non-positive is the
        // UNWEIGHTED mean (`_area_weighted_scalar_mean`, lines 241-243). Vertex
        // areas are floored at 1e-300, so a non-positive total means an empty
        // surface; reduce that path too rather than letting ranks disagree.
        Real mean = 0;
        if ( reduced[1] > Real( 0 ) )
        {
            mean = reduced[0] / reduced[1];
        }
        else
        {
            Real local_sum = 0;
            Kokkos::parallel_reduce(
                "beatnik_center_unweighted",
                Kokkos::RangePolicy<execution_space>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i, Real& acc ) {
                    acc += phi_owned( i );
                },
                local_sum );
            const Real total_sum = Comm::allReduceSum( mesh.comm(), local_sum );
            const long long total_n = mesh.globalVertexCount();
            mean = ( total_n > 0 ) ? total_sum / static_cast<Real>( total_n )
                                   : Real( 0 );
        }

        // Subtracted over the WHOLE local range, ghosts included: the shift is
        // one globally agreed constant, so applying it to ghosts keeps the mesh
        // halo-consistent and saves an exchange. Every rank computed the same
        // `mean` from the same collective, so the ghost and owner copies stay
        // bitwise equal.
        Kokkos::parallel_for(
            "beatnik_center_potential",
            Kokkos::RangePolicy<execution_space>( 0, n_local ),
            KOKKOS_LAMBDA( const int i ) { phi( i ) -= mean; } );
        Kokkos::fence();
    }

    /**
     * @brief Re-project the sheet vector onto the tangent plane.
     *
     * Port of mesh_solver.py::MeshZModelState.__post_init__ (line 93)
     *
     * Applied on every state construction under `SheetVector`. See
     * `SurfaceOperators::projectTangent` for why this cannot be skipped.
     *
     * **T2d — it takes the mesh too**, for the reason recorded on
     * `centerPotential`: the sheet vector it projects lives in the mesh. A no-op
     * under `Potential`, where the sheet vector is derived and already
     * tangential by construction.
     */
    template <class MeshType, class GeometryType>
    void projectSheetTangent( MeshType& mesh,
                              const GeometryType& geometry ) const
    {
        if ( _model != StateModel::SheetVector )
            return;
        auto sheet = mesh.sheetVector();
        SurfaceOperators::projectTangent( sheet, geometry.vertex_normal );
    }

    /**
     * @brief Largest sheet-vector magnitude anywhere on the surface.
     *
     * Port of run_adaptive_mesh_bubble.py::max_sheet_strength (lines 904-920)
     *
     * \f$\max\big(\max_v\|S_v\|,\ \max_f\|S_f\|\big)\f$ under `Potential`;
     * \f$\max_v\|S_v\|\f$ under `SheetVector`. Non-finite entries are dropped;
     * if *nothing* is finite the result is \f$+\infty\f$, which the dt clamp
     * and the filter threshold both handle by declining to act.
     *
     * @note MPI. `Comm::allReduceMax`. Must be global — see that function.
     *
     * **T2d — the "nothing is finite" rule is a GLOBAL statement.** The Python
     * drops non-finite candidates and returns \f$+\infty\f$ only if the whole
     * list is non-finite. Distributed, a rank whose owned range happens to be
     * entirely non-finite must not report \f$+\infty\f$ on its own, so the count
     * of finite candidates is reduced alongside the maximum and the verdict is
     * taken after the reduction.
     *
     * Reduced over **owned** vertices and **owned** faces (risk R9): this is a
     * global scalar, and although a max is idempotent under double counting,
     * the owned range is the contract everywhere else here.
     */
    template <class MeshType, class GeometryType>
    Real maxSheetStrength( MeshType& mesh, const GeometryType& geometry ) const
    {
        auto sheet = mesh.sheetVector();
        const int nv = mesh.ownedVertexCount();

        Real local_max = 0;
        long long local_finite = 0;
        Kokkos::parallel_reduce(
            "beatnik_max_sheet_vertex",
            Kokkos::RangePolicy<execution_space>( 0, nv ),
            KOKKOS_LAMBDA( const int i, Real& m, long long& c ) {
                const Real mag = Kokkos::sqrt( sheet( i, 0 ) * sheet( i, 0 ) +
                                               sheet( i, 1 ) * sheet( i, 1 ) +
                                               sheet( i, 2 ) * sheet( i, 2 ) );
                if ( Kokkos::isfinite( mag ) )
                {
                    ++c;
                    if ( mag > m )
                        m = mag;
                }
            },
            Kokkos::Max<Real>( local_max ), local_finite );

        if ( _model == StateModel::Potential )
        {
            // The FACE gradient too: "the vertex-gradient diagnostic can miss
            // triangle-scale potential jumps on irregular adaptive meshes"
            // (run_adaptive_mesh_bubble.py:904-910). Using only the vertex value
            // lets the dt throttle miss the blow-up it exists to catch.
            const int nf = mesh.ownedFaceCount();
            Kokkos::View<Real* [3], memory_space> face_sheet(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_max_sheet_face" ),
                nf );
            faceSheetVector( mesh, geometry, face_sheet );

            Real face_max = 0;
            long long face_finite = 0;
            Kokkos::parallel_reduce(
                "beatnik_max_sheet_face_reduce",
                Kokkos::RangePolicy<execution_space>( 0, nf ),
                KOKKOS_LAMBDA( const int f, Real& m, long long& c ) {
                    const Real mag = Kokkos::sqrt(
                        face_sheet( f, 0 ) * face_sheet( f, 0 ) +
                        face_sheet( f, 1 ) * face_sheet( f, 1 ) +
                        face_sheet( f, 2 ) * face_sheet( f, 2 ) );
                    if ( Kokkos::isfinite( mag ) )
                    {
                        ++c;
                        if ( mag > m )
                            m = mag;
                    }
                },
                Kokkos::Max<Real>( face_max ), face_finite );

            if ( face_max > local_max )
                local_max = face_max;
            local_finite += face_finite;
        }

        long long global_finite = 0;
        MPI_Allreduce( &local_finite, &global_finite, 1, MPI_LONG_LONG, MPI_SUM,
                       mesh.comm() );
        const Real global_max = Comm::allReduceMax( mesh.comm(), local_max );

        // The Python's `return np.inf` when nothing finite survived. Both the dt
        // clamp and the filter threshold handle +inf by declining to act.
        if ( global_finite == 0 )
            return std::numeric_limits<Real>::infinity();
        return global_max;
    }

    /**
     * @brief True iff every stored value is finite, globally.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1413-1416)
     *
     * The reference checks `state.vertices` and `state.sheet_vector` — note
     * that under the potential model `state.sheet_vector` is a *property* that
     * recomputes the surface gradient, so the check implicitly covers the
     * potential too. This port checks the stored fields directly and the
     * vertices separately, which is stricter and does not depend on a stub.
     *
     * Scanned over the **owned** range only. A ghost is an owned vertex on
     * another rank, so its owner reports it; including ghosts here would make
     * one rank's NaN counted twice, which changes nothing for a logical OR but
     * would report the failure on a rank that did not cause it.
     *
     * @note MPI. `Comm::allReduceAllFinite`. Must be unanimous or the run
     *       deadlocks — see that function. Note Tessera's `globalAllFinite`
     *       takes a **verdict, not data**, which is why the `isfinite` sweep is
     *       Beatnik's and only the bool is reduced.
     */
    template <class MeshType>
    bool allFinite( MeshType& mesh ) const
    {
        auto pos = mesh.positions();
        auto phi = mesh.potential();
        auto sheet = mesh.sheetVector();
        auto material = mesh.materialPosition();
        const int n = mesh.ownedVertexCount();
        const bool check_scalar = ( _model == StateModel::Potential );

        int bad = 0;
        Kokkos::parallel_reduce(
            "beatnik_state_all_finite",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i, int& acc ) {
                for ( int d = 0; d < 3; ++d )
                {
                    if ( !Kokkos::isfinite( pos( i, d ) ) )
                        ++acc;
                    if ( !Kokkos::isfinite( material( i, d ) ) )
                        ++acc;
                    if ( !check_scalar && !Kokkos::isfinite( sheet( i, d ) ) )
                        ++acc;
                }
                if ( check_scalar && !Kokkos::isfinite( phi( i ) ) )
                    ++acc;
            },
            bad );

        return Comm::allReduceAllFinite( mesh.comm(), bad == 0 );
    }

    /**
     * @brief Seed the material position from the current vertex positions.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1227, 1240, 1208-1209)
     * — `remesh_material_position = np.asarray(state.vertices).copy()`
     *
     * Done once at t=0, and on a restart from a checkpoint that predates the
     * material-position field.
     *
     * Copied over the **whole local range** so ghost rows are seeded too and
     * the mesh is halo-consistent for this field without an exchange — the
     * ghost positions are already current when this is called (every
     * construction entry point ends in a `haloExchange`), so the copy is exact
     * rather than approximately right.
     */
    template <class MeshType>
    void seedMaterialPosition( MeshType& mesh ) const
    {
        auto pos = mesh.positions();
        auto material = mesh.materialPosition();
        const int n = mesh.totalVertexCount();
        Kokkos::parallel_for(
            "beatnik_seed_material_position",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                    material( i, d ) = pos( i, d );
            } );
        Kokkos::fence();
    }

  private:
    StateModel _model;
};

} // namespace Beatnik

#endif // BEATNIK_SURFACESTATE_HPP
