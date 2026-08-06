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
 * model is a runtime tag and both storage arrays exist, with only the active
 * one allocated. That keeps the RHS and the integrator from being templated on
 * the model — which would double the instantiations of every Kokkos kernel for
 * no benefit, since the model is fixed for the whole run.
 */

#ifndef BEATNIK_SURFACESTATE_HPP
#define BEATNIK_SURFACESTATE_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Per-vertex solution fields on the interface.
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

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;

    /// @param model Which unknown is evolved. Fixed for the run.
    explicit SurfaceState( StateModel model )
        : _model( model )
    {
    }

    /// Which unknown is evolved.
    StateModel model() const { return _model; }

    /**
     * @brief `(Nv,)` velocity potential jump. Allocated only under
     *        `StateModel::Potential`.
     *
     * Units: velocity x length (a circulation per unit length of contour), so
     * its surface gradient has units of velocity.
     */
    scalar_view potential() const { return _potential; }

    /**
     * @brief `(Nv,3)` tangential sheet vector.
     *
     * Under `SheetVector` this is the evolved unknown; under `Potential` it is
     * a **cache** refreshed by `updateSheetVector` and must not be written
     * directly. Units: velocity.
     */
    vector_view sheetVector() const { return _sheet_vector; }

    /**
     * @brief `(Nv,3)` carried Lagrangian ("material") coordinate.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1227, 1240) and
     * ::dynamic_remesh_state_with_material (lines 1080-1110)
     *
     * Initialized to the vertex positions at t=0 and advected as an ordinary
     * per-vertex field through every remesh — interpolated at edge midpoints on
     * a split, carried on a collapse — but **never** integrated in time. So it
     * records, for each current vertex, roughly which piece of the *initial*
     * sheet it came from.
     *
     * Its only consumer is the nonlocal-proximity material exclusion: two faces
     * that are geometrically close but close in material coordinate too are the
     * same piece of sheet folded over on itself at the mesh scale, not two
     * approaching sheets, and must not trigger proximity refinement. Without
     * it, a tightly rolled spiral refines against its own immediate
     * neighborhood forever.
     */
    vector_view materialPosition() const { return _material_position; }

    /**
     * @brief Allocate the fields for a given vertex count.
     *
     * Called after every mesh edit. Only the array selected by `model()` is
     * allocated, plus the sheet-vector cache and the material position.
     */
    void resize( int vertex_count )
    {
        (void)vertex_count;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "resize" );
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
     */
    template <class MeshType, class GeometryType>
    void updateSheetVector( const MeshType& mesh, const GeometryType& geometry )
    {
        (void)mesh;
        (void)geometry;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "updateSheetVector" );
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
     */
    template <class MeshType, class GeometryType, class VectorView>
    void faceSheetVector( const MeshType& mesh, const GeometryType& geometry,
                          VectorView& face_sheet ) const
    {
        (void)mesh;
        (void)geometry;
        (void)face_sheet;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "faceSheetVector" );
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
     * @note MPI. Collective — see `SurfaceOperators::areaWeightedMean`.
     */
    template <class GeometryType>
    void centerPotential( const GeometryType& geometry )
    {
        (void)geometry;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "centerPotential" );
    }

    /**
     * @brief Re-project the sheet vector onto the tangent plane.
     *
     * Port of mesh_solver.py::MeshZModelState.__post_init__ (line 93)
     *
     * Applied on every state construction under `SheetVector`. See
     * `SurfaceOperators::projectTangent` for why this cannot be skipped.
     */
    template <class GeometryType>
    void projectSheetTangent( const GeometryType& geometry )
    {
        (void)geometry;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "projectSheetTangent" );
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
     */
    template <class MeshType, class GeometryType>
    Real maxSheetStrength( const MeshType& mesh,
                           const GeometryType& geometry ) const
    {
        (void)mesh;
        (void)geometry;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "maxSheetStrength" );
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
     * vertices separately.
     *
     * @note MPI. `Comm::allReduceAllFinite`. Must be unanimous or the run
     *       deadlocks — see that function.
     */
    template <class MeshType>
    bool allFinite( const MeshType& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "allFinite" );
    }

    /**
     * @brief Transfer all fields through a mesh edit's parent/weight map.
     *
     * Uses `MeshEditResult` from `Beatnik_MeshInterface.hpp`:
     * \f$f_{\text{new}}[i] = w_a[i]\,f_{\text{old}}[p_a[i]] +
     * w_b[i]\,f_{\text{old}}[p_b[i]]\f$.
     *
     * Applies to the potential (or sheet vector) **and** the material position;
     * forgetting the latter silently disables the proximity material exclusion,
     * which shows up much later as runaway refinement in the roll-up.
     */
    template <class EditResult>
    void remap( const EditResult& edit )
    {
        (void)edit;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "remap" );
    }

    /**
     * @brief Seed the material position from the current vertex positions.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1227, 1240, 1208-1209)
     *
     * Done once at t=0, and on a restart from a checkpoint that predates the
     * material-position field.
     */
    template <class MeshType>
    void seedMaterialPosition( const MeshType& mesh )
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceState", "seedMaterialPosition" );
    }

  private:
    StateModel _model;
    scalar_view _potential;
    vector_view _sheet_vector;
    vector_view _material_position;
};

} // namespace Beatnik

#endif // BEATNIK_SURFACESTATE_HPP
