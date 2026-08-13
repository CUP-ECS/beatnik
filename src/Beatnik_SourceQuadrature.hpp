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
 * @file Beatnik_SourceQuadrature.hpp
 * @brief Abstraction and factory for the surface quadrature that turns the
 *        continuous Birkhoff-Rott integral into a finite source list.
 *
 * THE INTEGRAL BEING DISCRETIZED
 * ------------------------------
 * The Birkhoff-Rott velocity is a surface integral,
 * \f[
 *   u(x) \;=\; \frac{1}{4\pi}\int_\Sigma
 *     \frac{(x-y)\times S(y)}{(b + |x-y|^2)^{3/2}}\; dS_y ,
 * \f]
 * and a quadrature rule replaces \f$\int_\Sigma f\,dS\f$ by
 * \f$\sum_s \omega_s f(y_s)\f$. This class produces the pairs
 * \f$(y_s,\ \omega_s S(y_s))\f$ — points and **area-weighted** strengths — and
 * nothing else. The kernel sum itself is `Beatnik_BRSolverBase.hpp`; the
 * normalization \f$1/4\pi\f$ belongs to the kernel, not to the weights.
 *
 * THE THREE RULES
 * ---------------
 * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet
 * (lines 768-792) and ::potential_mesh_birkhoff_rott_velocity (lines 804-840)
 *
 * | Rule | Points | Weight | Strength sampled at |
 * |------|--------|--------|---------------------|
 * | `Vertex`    | \f$N_v\f$   | \f$A_v\f$ (lumped vertex area) | the vertex |
 * | `Face`      | \f$N_f\f$   | \f$A_f\f$                      | the face centroid |
 * | `Triangle3` | \f$3N_f\f$  | \f$A_f/3\f$                    | 3 interior points |
 *
 * `Triangle3` uses the symmetric 3-point barycentric rule
 * \f$(2/3,1/6,1/6)\f$ and its two cyclic permutations, which is exact for
 * quadratics on the triangle. The points are strictly interior, which is the
 * reason to prefer it near self-contact: a vertex-centred rule places a source
 * exactly on a target, where the regularized kernel is at its stiffest.
 *
 * A SUBTLETY IN THE POTENTIAL MODEL
 * ---------------------------------
 * Under `StateModel::Potential` the three rules do **not** merely resample one
 * sheet-vector field at different points — they compute it differently:
 *
 *   - `Face` and `Triangle3` build the strength from the **exact per-face**
 *     potential gradient, \f$S_f = -\hat n_f\times\nabla_f\phi\f$
 *     (`mesh_solver.py:811, 826`).
 *   - `Vertex` builds it from the **area-averaged per-vertex** gradient,
 *     \f$S_v = -\hat n_v\times\nabla_s\phi\f$ (line 838).
 *
 * These differ at \f$O(h)\f$ on an irregular mesh, so switching quadrature
 * changes the answer by more than the quadrature error alone. That is faithful
 * to the reference and must not be "cleaned up".
 *
 * IMPLEMENTATION SCOPE IN THIS PORT
 * ---------------------------------
 * All three are selectable at the CLI so a Python command line runs unchanged.
 * **Only `Vertex` needs to be implemented** — see README "Source quadrature".
 * `Face` and `Triangle3` throw from `generate` until someone needs them.
 */

#ifndef BEATNIK_SOURCEQUADRATURE_HPP
#define BEATNIK_SOURCEQUADRATURE_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Interface for a surface quadrature rule.
 *
 * The virtuals take concrete types rather than being function templates,
 * because a virtual cannot be a template. They are concrete only *given* the
 * class template parameters — the underlying container choice is still open.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class SourceQuadratureBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using point_view = Kokkos::View<Real* [3], device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using strength_view = Kokkos::View<Real* [3], device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using geometry_type = MeshGeometry<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    virtual ~SourceQuadratureBase() = default;

    /// Which rule this is.
    virtual SourceQuadrature kind() const = 0;

    /**
     * @brief Number of quadrature points this rule produces on a rank.
     *
     * @param vertex_count Owned vertices on this rank.
     * @param face_count   Owned faces on this rank.
     */
    virtual int pointCount( int vertex_count, int face_count ) const = 0;

    /**
     * @brief Emit the quadrature points and area-weighted sheet strengths.
     *
     * @param mesh     Surface geometry.
     * @param geometry Precomputed areas and normals.
     * @param state    Solution state supplying \f$\phi\f$ or \f$S\f$.
     * @param[out] points     `(Ns,3)` quadrature positions \f$y_s\f$.
     * @param[out] strengths  `(Ns,3)` \f$\omega_s S(y_s)\f$, units of
     *                        velocity x length^2 (circulation x length).
     *
     * @note MPI. Purely local: each rank emits sources for its **owned**
     *       entities only. Emitting for ghosts double-counts them in the global
     *       sum, which shows up as a velocity field that changes magnitude with
     *       the rank count — a distinctive and easily missed failure.
     */
    virtual void generate( mesh_type& mesh,
                           const geometry_type& geometry,
                           const state_type& state, point_view& points,
                           strength_view& strengths ) const = 0;

    /**
     * @brief Emit points and area-weighted **surface gradients** \f$G_s\f$.
     *
     * Port of mesh_solver.py::surface_riesz_scalar_from_sheet (lines 843-871)
     * and ::potential_surface_riesz_scalar (lines 874-909)
     *
     * The Riesz-scalar evaluation needs \f$\nabla_s\phi\,dS\f$, not
     * \f$S\,dS\f$. For the sheet-vector model the two are related by
     * \f$G = \hat n \times S\f$ (line 852) — the inverse of the rotation that
     * produced \f$S\f$ from the gradient — while for the potential model the
     * gradient is available directly.
     *
     * Only needed under `--bernoulli-scalar-mode surface-riesz`.
     */
    virtual void generateGradient( mesh_type& mesh,
                                   const geometry_type& geometry,
                                   const state_type& state, point_view& points,
                                   strength_view& gradients ) const = 0;
};

//---------------------------------------------------------------------------//
/**
 * @brief One source per vertex, weighted by the lumped vertex area.
 *
 * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet
 * (lines 775-777) and ::potential_mesh_birkhoff_rott_velocity (lines 835-840)
 *
 * \f$y_s = v_s\f$, \f$\omega_s = A_{v_s}\f$,
 * \f$S(y_s) = -\hat n_{v_s}\times\nabla_s\phi\,(v_s)\f$.
 *
 * **This is the rule the C++ port implements.** It is also the cheapest
 * (\f$N_v\f$ sources against \f$N_v\f$ targets, versus \f$3N_f \approx 6N_v\f$
 * for `Triangle3`) and the one whose sources coincide with the targets, which
 * makes the direct \f$O(N^2)\f$ reference sum a single symmetric kernel and
 * simplifies validation of the FMM against it.
 */
template <class ExecutionSpace, class MemorySpace>
class VertexQuadrature : public SourceQuadratureBase<ExecutionSpace, MemorySpace>
{
  public:
    using base_type = SourceQuadratureBase<ExecutionSpace, MemorySpace>;
    using device_type = typename base_type::device_type;
    using point_view = typename base_type::point_view;
    using strength_view = typename base_type::strength_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;

    SourceQuadrature kind() const override { return SourceQuadrature::Vertex; }

    int pointCount( int vertex_count, int face_count ) const override
    {
        (void)face_count;
        return vertex_count;
    }

    /**
     * @brief Emit one source per **owned** vertex.
     *
     * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet
     * (lines 775-777) — `source_points = vertices`,
     * `source_strength = vertex_area[:, None] * sheet_vector`
     *
     * `state` is unused, and deliberately so: under both state models the
     * vertex sheet vector lives in the mesh (`mesh.sheetVector()`, an M1 vertex
     * user field). Under `Potential` it is derived, and the caller refreshes it
     * with `SurfaceState::updateSheetVector` before each evaluation; under
     * `SheetVector` it is the evolved unknown. Either way this rule reads the
     * same slice, so there is nothing model-dependent to branch on here — the
     * `Face` and `Triangle3` rules are where the models diverge (see the file
     * header).
     *
     * **Risk R9.** The range is `[0, ownedVertexCount())`, never the local
     * range. A ghost vertex is an owned vertex on another rank and is already
     * in that rank's source list; emitting it here as well double-counts it in
     * the global sum, which shows up as a velocity field that changes magnitude
     * with the rank count and the partition shape.
     *
     * `points` and `strengths` are reallocated to the owned count if they do
     * not already have it, so a caller may pass default-constructed views.
     */
    void generate( mesh_type& mesh, const geometry_type& geometry,
                   const state_type& state, point_view& points,
                   strength_view& strengths ) const override
    {
        (void)state;

        const int n = mesh.ownedVertexCount();
        if ( static_cast<int>( points.extent( 0 ) ) != n )
            Kokkos::realloc( points, n );
        if ( static_cast<int>( strengths.extent( 0 ) ) != n )
            Kokkos::realloc( strengths, n );

        auto pos = mesh.positions();
        auto sheet = mesh.sheetVector();
        auto va = geometry.vertex_area;
        auto p = points;
        auto w = strengths;

        Kokkos::parallel_for(
            "beatnik_vertex_quadrature_generate",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                {
                    p( i, d ) = pos( i, d );
                    // The weight is the lumped vertex area; the 1/4pi is the
                    // kernel's and is not applied here.
                    w( i, d ) = va( i ) * sheet( i, d );
                }
            } );
        Kokkos::fence();
    }

    /**
     * @brief Emit one gradient source per **owned** vertex.
     *
     * Port of mesh_solver.py::potential_surface_riesz_scalar (lines 897-901,
     * the `vertex` branch) and ::surface_riesz_scalar_from_sheet
     * (lines 850-853, the `vertex` branch)
     *
     * **The two state models build \f$G\f$ by different expressions, and this
     * is faithful rather than tidy.** Under `Potential` the reference takes the
     * per-vertex surface gradient **directly** — `surface_gradient(state,
     * state.potential)`, line 900 — while under `SheetVector` the gradient is
     * recovered from the sheet vector by the inverse rotation
     * \f$G = \hat n_v \times S\f$ (line 852). On a tangential gradient the two
     * agree to roundoff (the read-only reference measures `5.6e-17` between
     * them on the default icosphere), but they are not the same expression and
     * are not written as one here.
     *
     * Same R9 discipline and same realloc contract as `generate`.
     */
    void generateGradient( mesh_type& mesh, const geometry_type& geometry,
                           const state_type& state, point_view& points,
                           strength_view& gradients ) const override
    {
        const int n = mesh.ownedVertexCount();
        if ( static_cast<int>( points.extent( 0 ) ) != n )
            Kokkos::realloc( points, n );
        if ( static_cast<int>( gradients.extent( 0 ) ) != n )
            Kokkos::realloc( gradients, n );

        auto pos = mesh.positions();
        auto va = geometry.vertex_area;
        auto vn = geometry.vertex_normal;
        auto p = points;
        auto w = gradients;

        if ( state.model() == StateModel::Potential )
        {
            // The gradient is assembled from the WHOLE LOCAL face set, which
            // scatters into ghost rows, so the scratch view must span the whole
            // local range even though only the owned rows are emitted. (Nv comes
            // from the output view; see `surfaceGradient`.)
            const int n_local = mesh.totalVertexCount();
            Kokkos::View<Real* [3], device_type> gradient(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_vertex_quadrature_gradient" ),
                n_local );
            auto faces = mesh.faceVertices();
            auto phi = mesh.potential();
            SurfaceOperators::surfaceGradient( pos, faces, phi, vn, gradient );

            Kokkos::parallel_for(
                "beatnik_vertex_quadrature_gradient_potential",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                    {
                        p( i, d ) = pos( i, d );
                        w( i, d ) = va( i ) * gradient( i, d );
                    }
                } );
            Kokkos::fence();
        }
        else
        {
            auto sheet = mesh.sheetVector();
            Kokkos::parallel_for(
                "beatnik_vertex_quadrature_gradient_sheet",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    // G = n x S, the inverse of S = -n x G within the tangent
                    // plane. Written out so the component order is inspectable.
                    const Real g[3] = {
                        vn( i, 1 ) * sheet( i, 2 ) - vn( i, 2 ) * sheet( i, 1 ),
                        vn( i, 2 ) * sheet( i, 0 ) - vn( i, 0 ) * sheet( i, 2 ),
                        vn( i, 0 ) * sheet( i, 1 ) - vn( i, 1 ) * sheet( i, 0 ) };
                    for ( int d = 0; d < 3; ++d )
                    {
                        p( i, d ) = pos( i, d );
                        w( i, d ) = va( i ) * g[d];
                    }
                } );
            Kokkos::fence();
        }
    }
};

//---------------------------------------------------------------------------//
/**
 * @brief One source per face at its centroid, weighted by the face area.
 *
 * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet
 * (lines 778-781) and ::potential_mesh_birkhoff_rott_velocity (lines 808-817)
 *
 * \f$y_s = \tfrac13(a+b+c)\f$, \f$\omega_s = A_f\f$, with the strength from the
 * **exact face** gradient under the potential model and from the arithmetic
 * mean of the three vertex sheet vectors under the sheet-vector model.
 *
 * The Python default. Accepted here for CLI compatibility; not implemented in
 * this port.
 */
template <class ExecutionSpace, class MemorySpace>
class FaceQuadrature : public SourceQuadratureBase<ExecutionSpace, MemorySpace>
{
  public:
    using base_type = SourceQuadratureBase<ExecutionSpace, MemorySpace>;
    using point_view = typename base_type::point_view;
    using strength_view = typename base_type::strength_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;

    SourceQuadrature kind() const override { return SourceQuadrature::Face; }

    int pointCount( int vertex_count, int face_count ) const override
    {
        (void)vertex_count;
        return face_count;
    }

    void generate( mesh_type& mesh, const geometry_type& geometry,
                   const state_type& state, point_view& points,
                   strength_view& strengths ) const override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)points;
        (void)strengths;
        BEATNIK_NOT_IMPLEMENTED( "FaceQuadrature", "generate" );
    }

    void generateGradient( mesh_type& mesh, const geometry_type& geometry,
                           const state_type& state, point_view& points,
                           strength_view& gradients ) const override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)points;
        (void)gradients;
        BEATNIK_NOT_IMPLEMENTED( "FaceQuadrature", "generateGradient" );
    }
};

//---------------------------------------------------------------------------//
/**
 * @brief Three interior sources per face, barycentric (2/3, 1/6, 1/6).
 *
 * Port of mesh_solver.py::_triangle3_points (lines 618-624),
 * ::_triangle3_interpolate (lines 627-640) and ::_triangle3_area_weights
 * (lines 643-644)
 *
 * Points \f$y_{f,q} = \sum_a \lambda_{qa} p_{fa}\f$ with
 * \f$\lambda \in \{(2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)\}\f$; weights
 * \f$A_f/3\f$ each. Exact for quadratic integrands.
 *
 * Under the potential model the strength is the **face-constant** \f$S_f\f$
 * repeated at all three points (`mesh_solver.py:827`), not interpolated — the
 * face gradient of a piecewise-linear potential *is* constant on the face, so
 * this is exact, not an approximation. Under the sheet-vector model the vertex
 * sheet vectors *are* barycentrically interpolated (line 783).
 *
 * Accepted here for CLI compatibility; not implemented in this port.
 */
template <class ExecutionSpace, class MemorySpace>
class Triangle3Quadrature
    : public SourceQuadratureBase<ExecutionSpace, MemorySpace>
{
  public:
    using base_type = SourceQuadratureBase<ExecutionSpace, MemorySpace>;
    using point_view = typename base_type::point_view;
    using strength_view = typename base_type::strength_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;

    /// Barycentric coordinates of the three interior points, row-major.
    /// Port of mesh_solver.py::_TRIANGLE3_BARYCENTRIC (lines 608-615)
    static constexpr Real barycentric[3][3] = {
        { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 },
        { 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0 },
        { 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0 } };

    SourceQuadrature kind() const override
    {
        return SourceQuadrature::Triangle3;
    }

    int pointCount( int vertex_count, int face_count ) const override
    {
        (void)vertex_count;
        return 3 * face_count;
    }

    void generate( mesh_type& mesh, const geometry_type& geometry,
                   const state_type& state, point_view& points,
                   strength_view& strengths ) const override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)points;
        (void)strengths;
        BEATNIK_NOT_IMPLEMENTED( "Triangle3Quadrature", "generate" );
    }

    void generateGradient( mesh_type& mesh, const geometry_type& geometry,
                           const state_type& state, point_view& points,
                           strength_view& gradients ) const override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)points;
        (void)gradients;
        BEATNIK_NOT_IMPLEMENTED( "Triangle3Quadrature", "generateGradient" );
    }
};

//---------------------------------------------------------------------------//
/**
 * @brief Build the quadrature rule named by `--source-quadrature`.
 *
 * @param kind Rule to construct.
 * @return Owning pointer to the rule. Never null: an unrecognized value is a
 *         parse error at the CLI, not a silent fallback here.
 */
template <class ExecutionSpace, class MemorySpace>
std::unique_ptr<SourceQuadratureBase<ExecutionSpace, MemorySpace>>
createSourceQuadrature( SourceQuadrature kind )
{
    switch ( kind )
    {
    case SourceQuadrature::Vertex:
        return std::unique_ptr<
            SourceQuadratureBase<ExecutionSpace, MemorySpace>>(
            new VertexQuadrature<ExecutionSpace, MemorySpace>() );
    case SourceQuadrature::Face:
        return std::unique_ptr<
            SourceQuadratureBase<ExecutionSpace, MemorySpace>>(
            new FaceQuadrature<ExecutionSpace, MemorySpace>() );
    case SourceQuadrature::Triangle3:
        return std::unique_ptr<
            SourceQuadratureBase<ExecutionSpace, MemorySpace>>(
            new Triangle3Quadrature<ExecutionSpace, MemorySpace>() );
    }
    BEATNIK_NOT_IMPLEMENTED( "SourceQuadrature", "createSourceQuadrature" );
}

} // namespace Beatnik

#endif // BEATNIK_SOURCEQUADRATURE_HPP
