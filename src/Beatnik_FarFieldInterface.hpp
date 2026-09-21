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
 * @file Beatnik_FarFieldInterface.hpp
 * @brief ADAPTER (3 of 3). Far-field summation of the regularized
 *        Birkhoff-Rott kernel, backed by **Canopy**'s fast multipole method.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Canopy type, include a Canopy header or
 * hold a Canopy object. `Beatnik_BRSolverFMM.hpp` calls only through
 * `FarFieldSolver` below, and the runtime `(basis, order)` pair becomes a
 * Canopy template argument here and nowhere else. Everything Canopy-facing
 * sits behind `BEATNIK_ENABLE_CANOPY`: a `~canopy` build still compiles this
 * header, and the two evaluations throw `std::runtime_error` naming the
 * missing build option rather than failing to compile.
 *
 * THE KERNEL
 * ----------
 * Both the velocity and the Riesz-scalar evaluations are sums over the same
 * regularized \f$1/r^2\f$ field. Writing \f$\delta = x_t - y_s\f$ and
 * \f$r^2 = |\delta|^2\f$, with blob parameter \f$b\f$ (see
 * `ZModelParams::blob()`), the shared kernel is
 *
 * \f[
 *   K(x_t, y_s) \;=\; \frac{\delta}{(b + r^2)^{3/2}} .
 * \f]
 *
 * Note the kernel is **not** \f$\delta/|\delta|^3\f$: the denominator carries
 * the additive blob, which both removes the singularity at \f$r=0\f$ (a target
 * *is* a source in this problem — every vertex appears on both sides, including
 * its own self-interaction, which contributes exactly zero to the velocity
 * because \f$\delta = 0\f$ there) and sets the sheet thickness. An FMM
 * expansion of this kernel is therefore an expansion of a *softened* Coulomb
 * field, not of the bare one; that difference matters for the error estimate at
 * separations comparable to \f$\sqrt{b}\f$.
 *
 * The two evaluations Beatnik needs from it:
 *
 * **Velocity** (`evaluateVelocity`), the Birkhoff-Rott integral —
 * Port of mesh_solver.py::_source_velocity_direct_unsigned (lines 437-454):
 * \f[
 *   u(x_t) \;=\; \frac{\sigma_{BR}}{4\pi} \sum_s K(x_t, y_s) \times S_s ,
 * \f]
 * where \f$S_s\f$ is the area-weighted sheet strength at source \f$s\f$. The
 * \f$1/4\pi\f$ and `ZModelParams::br_sign` are applied **once, here**; no
 * caller re-applies either.
 *
 * **Riesz scalar** (`evaluateRieszScalar`) —
 * Port of mesh_solver.py::_source_riesz_scalar_direct (lines 457-489):
 * \f[
 *   \Psi(x_t) \;=\; -\frac{1}{4\pi^2} \sum_s \frac{\delta \cdot G_s}{(b+r^2)^{3/2}} ,
 * \f]
 * where \f$G_s\f$ is the area-weighted surface gradient at source \f$s\f$. Note
 * the different normalization: \f$-1/(4\pi^2)\f$, **not** \f$1/(4\pi)\f$,
 * negative, and **not** multiplied by `br_sign` — the reference returns this
 * one unsigned. On a flat periodic patch this discretizes
 * \f$\mathcal{F}^{-1}(i k_j \hat w_j / |k|)/(2\pi)\f$. Both prefactors are the
 * ones `BRSolverDirect` applies
 * (`Beatnik_BRSolverDirect.hpp:126-127` and `:206-208`) and this adapter must
 * agree with that file, not merely with the formulae above.
 *
 * ONE CANOPY SOLVE, TWO CONTRACTIONS
 * ----------------------------------
 * Canopy's `NComps` is its number of simultaneous independent charge
 * components. With `NComps = 3` and `compute_gradient = true` one traversal
 * produces, per target \f$i\f$, the \f$3\times3\f$ tensor
 *
 * \f[
 *   T_{cj}(i) \;=\; \texttt{gradient}(i,c,j) \;=\;
 *     -\sum_s q_{c,s}\,\frac{\delta_j}{(r^2+\varepsilon^2)^{3/2}},
 *   \qquad \delta = x_i - y_s ,
 * \f]
 *
 * (`DownwardSweep::gradient()`, accumulated for near pairs by
 * `canopy/src/Canopy_P2P.hpp:883-896`), which is Beatnik's kernel with
 * \f$\varepsilon^2 = b\f$ **up to sign**. Loading the three charge components
 * with the three components of the area-weighted source vector makes both
 * contractions Beatnik needs local post-processing of that tensor, with no
 * second traversal and no communication:
 *
 *   - **Velocity.** With \f$q_c = \omega_s S_{c,s}\f$, the Birkhoff-Rott sum
 *     \f$\sum_s(\delta\times S_s)\omega_s K\f$ is \f$-\epsilon_{ijk}T_{kj}\f$,
 *     i.e. \f$u_0 = T_{12}-T_{21}\f$, \f$u_1 = T_{20}-T_{02}\f$,
 *     \f$u_2 = T_{01}-T_{10}\f$. Reversing either index of a pair negates the
 *     whole field, which is the same failure as reversing
 *     `BRSolverDirect`'s \f$\delta\times S\f$ order.
 *   - **Riesz scalar.** With \f$q_c = \omega_s G_{c,s}\f$,
 *     \f$\sum_s(\delta\cdot G_s)\omega_s K\f$ is \f$-\operatorname{tr}T\f$.
 *
 * The two do **not** share a tensor — they contract against different source
 * fields — so they are two `solve()` calls over one tree, which Canopy
 * supports directly: `solve()` re-reads the charge slice and zeroes its
 * outputs on every call.
 *
 * TWO DECOMPOSITIONS, AND THE ROUND TRIP BETWEEN THEM
 * ---------------------------------------------------
 * Beatnik's sources are owned mesh vertices under Tessera's decomposition;
 * Canopy owns its own decomposition and **permutes and migrates the caller's
 * array** on `setup` and on every maintenance path, with the within-AoSoA
 * order after migration explicitly unspecified. Nothing in Canopy carries a
 * caller-supplied identity through that, so the round trip is Beatnik's job.
 * It is the **tag-reverse handshake** proven on `origin/develop-canopy`
 * (`src/FmmBRSolver.hpp:198-296` and `:325-480` on that branch, verified at 1
 * and 4 ranks): a tag travels with each particle through every Canopy
 * migration, and a `Cabana::Distributor` built from the tags currently held
 * routes fresh data in and results out.
 *
 * It is simpler here than there. Under `SourceQuadrature::Vertex` the sources
 * are exactly the owned vertices in owned order, and the targets are the same
 * rows of the same array, so **source index, target index and output row are
 * one integer** and the tag is `(origin_rank, owned_index)` rather than
 * develop-canopy's `(rank, i, j)`. That equality is what the `Vertex` guard in
 * both evaluations protects: under any other rule the sources stop being the
 * vertices and the one-integer tag silently becomes wrong.
 *
 * WHY THIS IS AN ADAPTER AND NOT JUST "CALL CANOPY"
 * -------------------------------------------------
 * Canopy's `Solver` is a persistent object whose tree, partition and
 * communication plan are the thing being reused across evaluations, and whose
 * order **and far-field basis** are compile-time template parameters against
 * Beatnik's runtime `FmmParams`. Holding that state, and erasing that type,
 * are both Canopy facts, so both live here and `BRSolverFMM` stays the thirty
 * lines that turn a `(mesh, geometry, state, quadrature)` tuple into a call.
 */

#ifndef BEATNIK_FARFIELDINTERFACE_HPP
#define BEATNIK_FARFIELDINTERFACE_HPP

#include <Beatnik_Config.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#ifdef BEATNIK_ENABLE_CANOPY
#include <Cabana_Core.hpp>

#include <Canopy_CartesianTaylorBasis.hpp>
#include <Canopy_LaplaceKernel.hpp>
#include <Canopy_Solver.hpp>
#endif

#include <mpi.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What the far field actually did on the last evaluation.
 *
 * A Beatnik POD naming no Canopy type, declared **outside**
 * `BEATNIK_ENABLE_CANOPY` so a `~canopy` build still compiles it. Every field
 * traces to an accessor Canopy already exposes, and this struct is the only
 * route by which anything above the adapter can read a Canopy number: the
 * `FarFieldSolver` is private to `BRSolverFMM` and `BRSolverBase`'s virtuals
 * return `void`, so a test that could not reach this would have to name a
 * Canopy type to learn anything.
 *
 * **Global versus rank-local is part of each field's meaning**, and the two
 * are not interchangeable: a fallback count of zero on rank 0 says nothing
 * about rank 3. The `global_` fields are `MPI_SUM`-reduced over the
 * communicator inside the evaluation; the `local_` fields are this rank's own
 * and are left unreduced because the quantity they are read against —
 * Canopy's 32768-key operator cap — is itself per-rank.
 *
 * The conventions table requires every accuracy figure to name the basis, the
 * order, the softening and the realized P2P pair fraction, which is why those
 * four are carried here rather than re-derived by each consumer.
 */
struct FarFieldDiagnostics
{
    /// Which Canopy maintenance path the evaluation took. Mirrors
    /// `Canopy::Solver::MaintenanceAction` with the first-call `Setup` case
    /// added, which that enum has no member for because `setup()` is a
    /// separate entry point there.
    enum class Maintenance
    {
        Setup,     ///< First evaluation, or a rebuilt round trip: `setup()`.
        Migrate,   ///< Cheapest path: positions moved, topology unchanged.
        Rebalance, ///< Topology changed; repartition and rebuild the plan.
        Rebuild,   ///< A particle escaped the bounding box; full do-over.
    };

    /// The maintenance path taken by the most recent evaluation.
    Maintenance maintenance = Maintenance::Setup;

    /// The basis actually instantiated, not the one requested — they are the
    /// same by construction, since an unsupported pair throws.
    FarFieldBasis basis = FarFieldBasis::CartesianTaylor;

    /// The order actually instantiated (Canopy's `P_ORDER`).
    int order = 0;

    /// The softening **length** in force, i.e. \f$\sqrt{\texttt{blob()}}\f$.
    /// A length, not the squared length `ZModelParams::blob()` returns.
    Real softening = 0.0;

    /// Total particles Canopy holds across the communicator, reduced from
    /// `Solver::num_local_particles()`. Equals the global source count when
    /// the round trip is intact, and is the cheap independent discriminator
    /// for a dropped or duplicated source.
    long long global_particle_count = 0;

    /// Global count of near-field (P2P) source-target pairs, computed by the
    /// adapter — Canopy reports no such counter. See
    /// `FarFieldSolver::p2pPairCountLocal` for how and why.
    long long global_p2p_pair_count = 0;

    /// `global_p2p_pair_count` over the square of `global_particle_count`.
    /// **A solve that is entirely P2P agrees with `BRSolverDirect` to
    /// round-off at any order and reads as success**, so no accuracy figure
    /// taken from this path is interpretable without this number beside it.
    double p2p_pair_fraction = 0.0;

    /// Global M2L pair count, reduced from `DownwardSweep`'s
    /// `total_m2l_pair_count()`.
    long long global_m2l_pair_count = 0;

    /// Global count of M2L pairs that missed the operator table and took the
    /// per-pair translate instead, reduced from
    /// `DownwardSweep::total_fallback_pair_count()`. Non-zero means the
    /// accuracy number is a mixture of two code paths: the fallback is the
    /// same mathematics evaluated pair by pair, slower and bitwise different.
    long long global_m2l_fallback_pair_count = 0;

    /// **Rank-local.** `DownwardSweep::m2l_n_unique_ops()` — distinct M2L
    /// operator keys realized by the last interaction-list build, against
    /// Canopy's per-rank 32768-key cap.
    int local_m2l_unique_op_count = 0;

    /// **Rank-local.** `DownwardSweep::m2l_op_cache_size()` — operators
    /// currently resident in the persistent cache.
    int local_m2l_op_cache_size = 0;

    /// **Rank-local.** `DownwardSweep::m2l_op_keys_built_count()` — cumulative
    /// operators ever built. Climbing by the full cache size at every build is
    /// the signature of a cache that retains nothing, which is what a
    /// level-keyed basis on a drifting bounding box does.
    long long local_m2l_op_keys_built = 0;
};

/// Spelling for `FarFieldDiagnostics::Maintenance`, so a diagnostic that
/// echoes the resolved action and the enum cannot disagree. One table per
/// enum, as in `Beatnik_Types.hpp`.
inline const char* toString( FarFieldDiagnostics::Maintenance v )
{
    switch ( v )
    {
    case FarFieldDiagnostics::Maintenance::Setup:
        return "setup";
    case FarFieldDiagnostics::Maintenance::Migrate:
        return "migrate";
    case FarFieldDiagnostics::Maintenance::Rebalance:
        return "rebalance";
    case FarFieldDiagnostics::Maintenance::Rebuild:
        return "rebuild";
    }
    return "setup";
}

//---------------------------------------------------------------------------//
/**
 * @brief Member layout of the particle tuple that travels through Canopy.
 *
 * An enum rather than four bare integers, so a slice is named at every use
 * site and the layout has one definition. develop-canopy's `FmmField`
 * namespace (`src/FmmBRSolver.hpp:55-61` on that branch) is the precedent;
 * two tag components suffice here because the source list is one-dimensional
 * — see "Two decompositions" in the file header.
 *
 * Unscoped deliberately: `Cabana::slice<M>` takes a `std::size_t` non-type
 * template argument, and a scoped enumerator would need a cast at every call.
 */
namespace FarFieldMember
{
enum Index
{
    Position = 0, ///< `Real[3]`, the source/target position \f$y_s = x_t\f$.
    Charge = 1,   ///< `Real[3]`, the area-weighted source vector.
    Output = 2,   ///< `Real[3]`, the contracted result. Component 0 only for
                  ///< the Riesz scalar.
    Tag = 3,      ///< `int[2]`, `(origin_rank, owned_index)`.
};

/// Tag components, so `tag( p, 0 )` is never written bare.
enum TagComponent
{
    OriginRank = 0,
    OwnedIndex = 1,
};
} // namespace FarFieldMember

//---------------------------------------------------------------------------//
/**
 * @brief Far-field kernel summation. Canopy lives behind this.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space holding sources and results.
 */
template <class ExecutionSpace, class MemorySpace>
class FarFieldSolver
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /// `(N,3)` source positions. Concrete, not templated: this is already the
    /// type `SourceQuadratureBase::point_view` and `BRSolverBase::vector_view`
    /// resolve to, and the far field is the last consumer that had a reason to
    /// stay generic.
    using point_view = Kokkos::View<Real* [3], device_type>;
    /// `(N,3)` vector field — source strengths in, velocity out.
    using vector_view = Kokkos::View<Real* [3], device_type>;
    /// `(N,)` scalar field.
    using scalar_view = Kokkos::View<Real*, device_type>;

    /**
     * @param comm   Communicator over which sources and targets are
     *               distributed. The sum is global: every target sees every
     *               source on every rank.
     * @param params FMM tunables. `basis` and `order` are resolved to a
     *               Canopy instantiation **here, in the constructor**, so an
     *               unsupported pair fails at solver construction rather than
     *               at the first Runge-Kutta stage.
     *
     * @throws std::runtime_error if `(params.basis, params.order)` is not one
     *         of the six built instantiations, or if `ncrit` or `max_depth` is
     *         outside what Canopy accepts.
     */
    FarFieldSolver( MPI_Comm comm, const FmmParams& params )
        : _comm( comm )
        , _params( params )
    {
        MPI_Comm_rank( _comm, &_rank );
#ifdef BEATNIK_ENABLE_CANOPY
        validateTreeParams( params );
        _impl = makeImpl( params );
        _diagnostics.basis = params.basis;
        _diagnostics.order = params.order;
#endif
    }

    /**
     * @brief Evaluate the Birkhoff-Rott velocity at every source point.
     *
     * \f$ u(x_t) = \frac{\sigma_{BR}}{4\pi}\sum_s
     *     \frac{(x_t - y_s) \times S_s}{(b + |x_t-y_s|^2)^{3/2}} \f$
     *
     * **There is no `targets` argument, because Canopy has no target list.**
     * `Solver::solve` sizes its potential and gradient views to the local
     * particle count and evaluates the field at the particle positions only,
     * so evaluating anywhere else would mean padding the particle set with
     * zero-charge targets. Under `SourceQuadrature::Vertex` the targets *are*
     * the sources, so one array serves both sides and one integer indexes
     * source, target and output row alike.
     *
     * @param sources   `(Ns,3)` source positions \f$y_s\f$, **owned rows
     *                  only**. A ghost emitted here is an owned source on
     *                  another rank and would be double-counted globally,
     *                  which presents as a velocity whose magnitude changes
     *                  with the rank count (`Beatnik_SourceQuadrature.hpp`
     *                  risk R9).
     * @param strengths `(Ns,3)` area-weighted sheet strengths \f$S_s\f$, in
     *                  units of (velocity x length), i.e. circulation.
     * @param params    Supplies `blob()` — whose **square root** is the
     *                  softening length Canopy is given, since `blob()` is a
     *                  squared length and `FmmConfig::softening` is a length —
     *                  together with `br_sign` and `source_quadrature`. One
     *                  object rather than a bare `Real blob`, so there is no
     *                  second source of truth for the softening and no
     *                  separate sign argument to forget.
     * @param[out] velocity `(Ns,3)` result, **overwritten** (not accumulated),
     *                  in velocity units, on the owned rows in owned order.
     *                  Reallocated if it does not already have that extent.
     *                  \f$1/4\pi\f$ and `br_sign` are already applied.
     *
     * @note MPI. Collective, and every rank must enter it — including a rank
     *       that owns zero sources. The sequence branches only on values that
     *       are the result of a reduction, never on a local count.
     *
     * @throws std::runtime_error if built without Canopy, if
     *         `params.source_quadrature` is not `Vertex`, if the softening
     *         length is not strictly positive, or if the round trip failed to
     *         return exactly one result per owned source.
     */
    void evaluateVelocity( const point_view& sources,
                           const vector_view& strengths,
                           const ZModelParams& params, vector_view& velocity )
    {
#ifndef BEATNIK_ENABLE_CANOPY
        (void)sources;
        (void)strengths;
        (void)params;
        (void)velocity;
        throwNoCanopy( "evaluateVelocity" );
#else
        const int ns = static_cast<int>( sources.extent( 0 ) );
        if ( static_cast<int>( velocity.extent( 0 ) ) != ns )
            Kokkos::realloc( velocity, ns );
        Kokkos::deep_copy( velocity, Real( 0 ) );

        // 1/4pi ONCE, here, and `br_sign` on the velocity only. The same
        // expression BRSolverDirect.hpp:126-127 uses, deliberately spelled
        // the same way.
        const Real coefficient =
            params.br_sign / ( Real( 4 ) * static_cast<Real>( M_PI ) );

        scalar_view unused;
        evaluate( sources, strengths, params, Contraction::Curl, coefficient,
                  velocity, unused );
#endif
    }

    /**
     * @brief Evaluate the surface Riesz scalar at every source point.
     *
     * \f$ \Psi(x_t) = -\frac{1}{4\pi^2}\sum_s
     *     \frac{(x_t - y_s) \cdot G_s}{(b + |x_t-y_s|^2)^{3/2}} \f$
     *
     * Used only when `--bernoulli-scalar-mode surface-riesz` is selected. The
     * Python supports this mode for `direct`, `local` and `clustered` but
     * **not** for `treecode` (`mesh_solver.py:605` raises), so a Python run
     * cannot combine `surface-riesz` with the default far-field path. Beatnik
     * removes that restriction by routing it through the same FMM, which also
     * means there is no gold file for the combination.
     *
     * Same kernel, same round trip and same tree as `evaluateVelocity`; only
     * the contraction (a trace instead of a cross product) and the prefactor
     * differ. It is a **second** `solve()` because it contracts against a
     * different source field, not a re-read of the velocity's tensor.
     *
     * @param sources   `(Ns,3)` source positions, owned rows only.
     * @param gradients `(Ns,3)` area-weighted surface gradients \f$G_s\f$.
     * @param params    As `evaluateVelocity`, except that `br_sign` is **not**
     *                  applied here — the reference returns this one unsigned
     *                  (`mesh_solver.py:581-587`), an asymmetry with the
     *                  velocity path that is reproduced deliberately.
     * @param[out] scalar `(Ns,)` result, **overwritten**. The
     *                  \f$-1/4\pi^2\f$ is already applied.
     *
     * @note MPI. Collective, as `evaluateVelocity`.
     */
    void evaluateRieszScalar( const point_view& sources,
                              const vector_view& gradients,
                              const ZModelParams& params, scalar_view& scalar )
    {
#ifndef BEATNIK_ENABLE_CANOPY
        (void)sources;
        (void)gradients;
        (void)params;
        (void)scalar;
        throwNoCanopy( "evaluateRieszScalar" );
#else
        const int ns = static_cast<int>( sources.extent( 0 ) );
        if ( static_cast<int>( scalar.extent( 0 ) ) != ns )
            Kokkos::realloc( scalar, ns );
        Kokkos::deep_copy( scalar, Real( 0 ) );

        // -1/(4 pi^2). NOT multiplied by `br_sign`, matching
        // BRSolverDirect.hpp:206-208.
        const Real coefficient =
            Real( -1 ) / ( Real( 4 ) * static_cast<Real>( M_PI ) *
                           static_cast<Real>( M_PI ) );

        vector_view unused;
        evaluate( sources, gradients, params, Contraction::Trace, coefficient,
                  unused, scalar );
#endif
    }

    /**
     * @brief What the far field did on the most recent evaluation.
     *
     * In a `~canopy` build this is the default-constructed member: the two
     * evaluations throw, so nothing ever writes it.
     */
    const FarFieldDiagnostics& diagnostics() const { return _diagnostics; }

    /// FMM tunables in force.
    const FmmParams& params() const { return _params; }

  private:
#ifdef BEATNIK_ENABLE_CANOPY
    /// Which contraction of Canopy's 3x3 gradient tensor an evaluation wants.
    /// An enum rather than a bool, per the conventions table, because the two
    /// differ in output rank as well as in arithmetic.
    enum class Contraction
    {
        Curl, ///< \f$u_i = -\epsilon_{ijk}T_{kj}\f$, into `Output(p, 0..2)`.
        Trace ///< \f$\Psi = -\operatorname{tr}T\f$, into `Output(p, 0)`.
    };

    /// Canopy's number of simultaneous charge components. Three, because the
    /// source is a vector field and both contractions are of the resulting
    /// 3x3 tensor — see "One Canopy solve, two contractions" in the header.
    static constexpr int NCOMPS = 3;

    using particle_member_types =
        Cabana::MemberTypes<Real[3], Real[3], Real[3], int[2]>;
    using aosoa_type = Cabana::AoSoA<particle_member_types, MemorySpace>;

    /// Canopy's gradient output, `(num_local, NComps, 3)`. Identical for every
    /// dispatch arm — it depends on `Scalar` and `NComps` only, not on the
    /// basis or the order — which is what lets the contraction, the pack and
    /// the whole round trip live once here rather than six times in `ImplFor`.
    using gradient_view_type = Kokkos::View<Real* [NCOMPS][3], MemorySpace>;

    using builder_type = Canopy::TreeBuilder<MemorySpace, ExecutionSpace>;
    using comm_plan_type =
        Canopy::CommunicationPlan<MemorySpace, ExecutionSpace>;

    //-----------------------------------------------------------------------//
    /**
     * @brief The arm-independent face of one Canopy `Solver` instantiation.
     *
     * Canopy's basis is a template-*template* parameter and its order a
     * non-type template parameter, so every `(basis, order)` arm is a distinct
     * C++ type and `FarFieldSolver` cannot name the `Solver` as a typed
     * member. It holds a `std::unique_ptr<Impl>` instead, constructed once in
     * the constructor and persistent thereafter — which is what makes the
     * tree, the partition and the communication plan reusable across
     * evaluations at all.
     *
     * Only the operations that genuinely depend on the arm are virtual. The
     * tree builder and the communication plan do **not**: they are templated
     * on the spaces alone, so the P2P pair count is written once against the
     * concrete types below. `downward()` is the exception — its type carries
     * the basis — which is why `readDiagnostics` is here rather than outside.
     */
    class Impl
    {
      public:
        virtual ~Impl() = default;

        /// Construct the Canopy solver. Deferred out of `FarFieldSolver`'s
        /// constructor because the softening comes from `ZModelParams`, which
        /// only the evaluation has.
        virtual void createSolver( MPI_Comm comm,
                                   const Canopy::FmmConfig& cfg ) = 0;

        /// True once `createSolver` has run.
        virtual bool haveSolver() const = 0;

        /// `Solver::setup`. `num_before` is the local count **before**
        /// migration, which is Canopy's stated contract for this argument.
        virtual void setup( aosoa_type& particles, int num_before ) = 0;

        /// `Solver::auto_maintain`, mapped onto the Beatnik enum.
        virtual FarFieldDiagnostics::Maintenance
        autoMaintain( aosoa_type& particles ) = 0;

        /// `Solver::solve` with `compute_gradient = true`.
        virtual void solve( aosoa_type& particles ) = 0;

        virtual int numLocalParticles() const = 0;
        virtual gradient_view_type gradient() const = 0;
        virtual const builder_type& builder() const = 0;
        virtual const comm_plan_type& commPlan() const = 0;

        /// Fill the arm-dependent (downward-sweep) half of the diagnostics.
        /// The values written are rank-local; the caller reduces the two that
        /// are meaningful globally.
        virtual void readDiagnostics( FarFieldDiagnostics& d ) const = 0;
    };

    //-----------------------------------------------------------------------//
    /**
     * @brief One dispatch arm: Canopy's solver at a fixed basis and order.
     *
     * @tparam FarField Canopy far-field basis, **named explicitly at every
     *         instantiation**. Canopy defaults this parameter to
     *         `LaplaceKernel`, so an instantiation that omits it compiles,
     *         runs, and produces the bare-\f$1/r\f$ far field with no
     *         diagnostic — at `near_softening_factor = 0` that is wrong by
     *         tens of percent, not by a little. The `static_assert` below is
     *         what turns the omission into a compile error: `kernel_type` is
     *         what the `Solver` actually instantiated, and comparing it
     *         against `FarField` fails if the argument never reached the slot.
     * @tparam P_ORDER Canopy's order knob: the Taylor order \f$p\f$ under
     *         `CartesianTaylorBasis`, \f$P\f$ under `LaplaceKernel`.
     */
    template <template <class, int, int> class FarField, int P_ORDER>
    class ImplFor : public Impl
    {
      public:
        using solver_type = Canopy::Solver<MemorySpace, ExecutionSpace, Real,
                                           P_ORDER, NCOMPS, FarField>;

        static_assert(
            std::is_same<typename solver_type::kernel_type,
                         FarField<Real, P_ORDER, NCOMPS>>::value,
            "FarFieldSolver: this arm's Canopy Solver did not instantiate the "
            "basis the arm names. Canopy's basis template parameter is "
            "defaulted to LaplaceKernel, so a dropped argument silently "
            "produces a bare-1/r far field instead of failing." );

        void createSolver( MPI_Comm comm,
                           const Canopy::FmmConfig& cfg ) override
        {
            _solver = Canopy::createSolver<MemorySpace, ExecutionSpace, Real,
                                           P_ORDER, NCOMPS, FarField>( comm,
                                                                       cfg );
        }

        bool haveSolver() const override { return static_cast<bool>( _solver ); }

        void setup( aosoa_type& particles, int num_before ) override
        {
            _solver->template setup<FarFieldMember::Position,
                                    FarFieldMember::Charge>( particles,
                                                             num_before );
        }

        FarFieldDiagnostics::Maintenance
        autoMaintain( aosoa_type& particles ) override
        {
            const auto action =
                _solver->template auto_maintain<FarFieldMember::Position,
                                                FarFieldMember::Charge>(
                    particles );
            using A = typename solver_type::MaintenanceAction;
            switch ( action )
            {
            case A::Migrate:
                return FarFieldDiagnostics::Maintenance::Migrate;
            case A::Rebalance:
                return FarFieldDiagnostics::Maintenance::Rebalance;
            case A::Rebuild:
                return FarFieldDiagnostics::Maintenance::Rebuild;
            }
            return FarFieldDiagnostics::Maintenance::Rebuild;
        }

        void solve( aosoa_type& particles ) override
        {
            _solver->template solve<FarFieldMember::Position,
                                    FarFieldMember::Charge>(
                particles, /*compute_gradient=*/true );
        }

        int numLocalParticles() const override
        {
            return _solver->num_local_particles();
        }

        gradient_view_type gradient() const override
        {
            return _solver->gradient();
        }

        const builder_type& builder() const override
        {
            return _solver->builder();
        }

        const comm_plan_type& commPlan() const override
        {
            return _solver->comm_plan();
        }

        void readDiagnostics( FarFieldDiagnostics& d ) const override
        {
            const auto& down = _solver->downward();
            d.local_m2l_unique_op_count = down.m2l_n_unique_ops();
            d.local_m2l_op_cache_size = down.m2l_op_cache_size();
            d.local_m2l_op_keys_built = down.m2l_op_keys_built_count();
            d.global_m2l_pair_count = down.total_m2l_pair_count();
            d.global_m2l_fallback_pair_count = down.total_fallback_pair_count();
        }

      private:
        std::shared_ptr<solver_type> _solver;
    };

    //-----------------------------------------------------------------------//
    /**
     * @brief Resolve the runtime `(basis, order)` pair onto a built arm.
     *
     * **Six arms**, and nothing rounds or substitutes: an unsupported pair
     * throws naming the supported set.
     *
     *   - `CartesianTaylor` at **0** — the monopole-only negative case, whose
     *     *passing* an accuracy test would be the alarm;
     *   - `CartesianTaylor` at **2** — one order below production, where
     *     Canopy measured \f$8.996\times10^{-3}\f$ on the gradient;
     *   - `CartesianTaylor` at **3** — the production path;
     *   - `CartesianTaylor` at **4** and **5** — **scan-only**. Canopy's
     *     derivative-ladder oracle is validated to \f$|k|=6\f$, which is
     *     \f$2p\f$ at \f$p=3\f$ and no further, while the M2L reaches
     *     \f$b_{p+q}\f$ at \f$|p+q|=2p\f$. These two are therefore measurable
     *     but **not adoptable as the production order**; raising the
     *     production order is an upstream request for another degree of
     *     oracle, not a Beatnik change.
     *   - `SolidHarmonic` at **3** — the contrast arm, at *equal* order to the
     *     production path so the comparison is of bases and not of orders.
     *
     * **What an extra arm costs.** This header reaches every translation unit
     * that creates a BR solver, through `Beatnik_CreateBRSolver.hpp`, and
     * every arm below is instantiated in all of them because the switch
     * compiles all its branches. Each one therefore instantiates Canopy's
     * whole pipeline — `TreeBuilder`, `TreePartitioner`, `CommunicationPlan`,
     * `UpwardSweep`, `DownwardSweep`, `P2P` — in every such TU. Adding a
     * second `SolidHarmonic` arm is one line, and that is the price of it.
     */
    static std::unique_ptr<Impl> makeImpl( const FmmParams& params )
    {
        if ( params.basis == FarFieldBasis::CartesianTaylor )
        {
            switch ( params.order )
            {
            case 0:
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::CartesianTaylorBasis, 0>() );
            case 2:
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::CartesianTaylorBasis, 2>() );
            case 3:
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::CartesianTaylorBasis, 3>() );
            case 4:
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::CartesianTaylorBasis, 4>() );
            case 5:
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::CartesianTaylorBasis, 5>() );
            default:
                break;
            }
        }
        else if ( params.basis == FarFieldBasis::SolidHarmonic )
        {
            if ( params.order == 3 )
                return std::unique_ptr<Impl>(
                    new ImplFor<Canopy::LaplaceKernel, 3>() );
        }

        throw std::runtime_error(
            std::string( "FarFieldSolver: no far-field instantiation for "
                         "basis '" ) +
            toString( params.basis ) + "' at order " +
            std::to_string( params.order ) +
            ". Built arms are cartesian-taylor at orders 0, 2, 3, 4 and 5 "
            "(3 is the production order; 4 and 5 are scan-only, above "
            "Canopy's validated derivative ladder) and solid-harmonic at "
            "order 3. Set --br-treecode-order to a supported value." );
    }

    /// The two `FmmConfig` members Canopy leaves without a default
    /// initializer, checked before anything builds a tree from them. Canopy's
    /// `TreeBuilder` throws above 19 because that is what its `uint64_t`
    /// Morton key holds; the lower bounds have no Canopy check at all.
    static void validateTreeParams( const FmmParams& params )
    {
        if ( params.ncrit <= 0 )
            throw std::runtime_error(
                "FarFieldSolver: FmmParams::ncrit must be positive, got " +
                std::to_string( params.ncrit ) );
        if ( params.max_depth <= 0 )
            throw std::runtime_error(
                "FarFieldSolver: FmmParams::max_depth must be positive, got " +
                std::to_string( params.max_depth ) );
        if ( params.max_depth > 19 )
            throw std::runtime_error(
                "FarFieldSolver: FmmParams::max_depth must be at most 19 "
                "(Canopy's Morton key holds no more), got " +
                std::to_string( params.max_depth ) );
    }

    /**
     * @brief Build Canopy's `FmmConfig` from `FmmParams` and the blob.
     *
     * Fourteen of `FmmParams`' sixteen members map straight across; `basis`
     * and `order` do not, because Canopy's far field and order are *template*
     * parameters rather than config fields — which is exactly why the dispatch
     * above has to name them.
     *
     * **`softening` is set explicitly and checked for positivity here**, and
     * the check cannot be left to Canopy. `FmmConfig::softening` defaults to
     * \f$-1\f$, which is not an error but a *selector*: it switches on
     * distribution-based auto-softening at the first `setup()`, an effective
     * \f$\varepsilon\f$ that moves with the particle distribution, is not
     * Beatnik's blob, and additionally disables `near_softening_factor`.
     * Canopy aborts on `softening <= 0` inside `build_m2l_operators`, which
     * catches zero and negative values but never the sentinel, because the
     * sentinel has already been replaced by then.
     *
     * The value is \f$\sqrt{\texttt{blob()}}\f$ — a **length**, where
     * `blob()` is a squared length — derived at this one call site so both
     * `--kernel-blob-mode` settings are right without a second source of
     * truth.
     */
    Canopy::FmmConfig buildConfig( const ZModelParams& zparams ) const
    {
        const Real softening = std::sqrt( zparams.blob() );
        if ( !( softening > Real( 0 ) ) )
            throw std::runtime_error(
                "FarFieldSolver: the softening length sqrt(blob()) must be "
                "strictly positive, got " +
                std::to_string( softening ) +
                ". Canopy's default of -1 is not a fallback but a switch to "
                "distribution-based auto-softening, so it cannot be left to "
                "the default; raise --eps." );

        Canopy::FmmConfig cfg;
        cfg.ncrit = _params.ncrit;
        cfg.max_depth = _params.max_depth;
        cfg.xmin_tol = _params.xmin_tol;
        cfg.xmax_tol = _params.xmax_tol;
        cfg.ymin_tol = _params.ymin_tol;
        cfg.ymax_tol = _params.ymax_tol;
        cfg.zmin_tol = _params.zmin_tol;
        cfg.zmax_tol = _params.zmax_tol;
        cfg.ncrit_tol = _params.ncrit_tol;
        cfg.replication_depth = _params.replication_depth;
        cfg.imbalance_tolerance = _params.imbalance_tolerance;
        cfg.mac_theta = _params.mac_theta;
        cfg.softening = softening;
        cfg.near_softening_factor = _params.near_softening_factor;
        cfg.m2l_op_table_byte_budget = _params.m2l_op_table_byte_budget;
        return cfg;
    }

    /**
     * @brief Pack the caller's owned rows into a mesh-ordered AoSoA.
     *
     * Step 2 of the round trip. The tag is `(origin_rank, owned_index)` with
     * `owned_index` the caller's row, which under `SourceQuadrature::Vertex`
     * is simultaneously the source index, the target index and the output row.
     */
    void packSources( const point_view& sources, const vector_view& charges,
                      int ns )
    {
        _mesh_particles.resize( static_cast<std::size_t>( ns ) );

        auto pos = Cabana::slice<FarFieldMember::Position>( _mesh_particles );
        auto q = Cabana::slice<FarFieldMember::Charge>( _mesh_particles );
        auto out = Cabana::slice<FarFieldMember::Output>( _mesh_particles );
        auto tag = Cabana::slice<FarFieldMember::Tag>( _mesh_particles );
        const int rank = _rank;

        Kokkos::parallel_for(
            "beatnik_far_field_pack",
            Kokkos::RangePolicy<ExecutionSpace>( 0, ns ),
            KOKKOS_LAMBDA( const int p ) {
                for ( int d = 0; d < 3; ++d )
                {
                    pos( p, d ) = sources( p, d );
                    q( p, d ) = charges( p, d );
                    out( p, d ) = Real( 0 );
                }
                tag( p, FarFieldMember::OriginRank ) = rank;
                tag( p, FarFieldMember::OwnedIndex ) = p;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Route this evaluation's fresh tuples into Canopy's ordering.
     *
     * Step 3 of the round trip, the reuse branch, and the five-step
     * tag-reverse handshake from `origin/develop-canopy`'s
     * `FmmBRSolver::buildForwardDistributor` (`src/FmmBRSolver.hpp:198-296` on
     * that branch), reduced from a 2-D grid index to one integer:
     *
     *   1. on every rank, for each tuple Canopy currently holds, pack a claim
     *      `(owned_index, my_rank)` destined for `tag.origin_rank`;
     *   2. migrate the claims back to the ranks that own those indices;
     *   3. fill `owned_index -> canopy_rank` from the claims received;
     *   4. read one destination rank per mesh-ordered row out of that map;
     *   5. build the forward `Distributor` from those destinations.
     *
     * Rebuilt every evaluation. Caching it across `Migrate`-action evaluations
     * is a named future optimization and is not done here.
     *
     * **It also validates**, because Beatnik's source set is not fixed the way
     * develop-canopy's grid was: refinement, coarsening and mesh load
     * balancing all change how many owned vertices a rank has and what a given
     * index means. A stale map does not fail loudly — an unclaimed row gets
     * `-1`, which `Cabana::Distributor` reads as "drop this element", so the
     * source would silently vanish from the sum. The out-parameter therefore
     * reports whether every row was claimed exactly once and no claim was out
     * of range; the caller reduces that over the communicator and falls back
     * to a full `setup()` when any rank says no.
     *
     * @param ns Owned source count for **this** evaluation.
     * @param[out] valid_local 1 if this rank's map is complete, else 0.
     */
    Cabana::Distributor<MemorySpace>
    buildForwardDistributor( int ns, int& valid_local )
    {
        using claim_member_types = Cabana::MemberTypes<int[2]>;
        using claim_aosoa = Cabana::AoSoA<claim_member_types, MemorySpace>;

        const int num_canopy = static_cast<int>( _canopy_particles.size() );

        claim_aosoa claims_canopy( "beatnik_far_field_claims_canopy",
                                   num_canopy );
        Kokkos::View<int*, MemorySpace> claim_dests(
            Kokkos::ViewAllocateWithoutInitializing(
                "beatnik_far_field_claim_dests" ),
            num_canopy );
        {
            auto tag =
                Cabana::slice<FarFieldMember::Tag>( _canopy_particles );
            auto claim = Cabana::slice<0>( claims_canopy );
            const int my_rank = _rank;
            Kokkos::parallel_for(
                "beatnik_far_field_pack_claims",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_canopy ),
                KOKKOS_LAMBDA( const int p ) {
                    claim( p, 0 ) = tag( p, FarFieldMember::OwnedIndex );
                    claim( p, 1 ) = my_rank;
                    claim_dests( p ) = tag( p, FarFieldMember::OriginRank );
                } );
            Kokkos::fence();
        }

        Cabana::Distributor<MemorySpace> claim_dist( _comm, claim_dests );
        claim_aosoa claims_mesh( "beatnik_far_field_claims_mesh",
                                 claim_dist.totalNumImport() );
        Cabana::migrate( claim_dist, claims_canopy, claims_mesh );

        Kokkos::View<int*, MemorySpace> dest_ranks(
            "beatnik_far_field_forward_dests", ns );
        Kokkos::deep_copy( dest_ranks, -1 );

        const int num_claims = static_cast<int>( claims_mesh.size() );
        int out_of_range = 0;
        {
            auto claim = Cabana::slice<0>( claims_mesh );
            Kokkos::parallel_reduce(
                "beatnik_far_field_fill_dests",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_claims ),
                KOKKOS_LAMBDA( const int c, int& bad ) {
                    const int i = claim( c, 0 );
                    if ( i >= 0 && i < ns )
                        dest_ranks( i ) = claim( c, 1 );
                    else
                        ++bad;
                },
                out_of_range );
            Kokkos::fence();
        }

        int filled = 0;
        Kokkos::parallel_reduce(
            "beatnik_far_field_count_dests",
            Kokkos::RangePolicy<ExecutionSpace>( 0, ns ),
            KOKKOS_LAMBDA( const int i, int& n ) {
                if ( dest_ranks( i ) >= 0 )
                    ++n;
            },
            filled );
        Kokkos::fence();

        valid_local = ( num_claims == ns && filled == ns && out_of_range == 0 )
                          ? 1
                          : 0;

        return Cabana::Distributor<MemorySpace>( _comm, dest_ranks );
    }

    /**
     * @brief The global near-field pair count this rank contributes.
     *
     * Step 7 of T2: **Canopy reports no such counter**, and its own accuracy
     * test settles for an `m2l_n_unique_ops() > 0` liveness guard, which is
     * weaker than a fraction. The ingredients are host-side and already
     * public: `CommunicationPlan::p2p_plan().neighbor_lists` maps each of this
     * rank's owned leaves to its near-field neighbour leaves, and
     * `TreeBuilder::cells()` is a globally replicated cell list carrying
     * `global_count` per cell. A leaf's particles live on exactly one rank
     * after partitioning, so summing
     * \f$n_T\sum_{S\in\mathrm{nbrs}(T)}n_S\f$ over owned leaves and reducing
     * gives the global pair count with no double counting.
     *
     * A few thousand host-side hash lookups per evaluation, computed
     * unconditionally rather than behind a flag, because it is the number
     * every accuracy claim on this path has to carry: a solve that is entirely
     * P2P agrees with `BRSolverDirect` to round-off and reads as success.
     */
    long long p2pPairCountLocal() const
    {
        const auto& cells = _impl->builder().cells();
        std::unordered_map<Canopy::MortonKey, long long> occupancy;
        occupancy.reserve( cells.size() * 2 );
        for ( const auto& c : cells )
            occupancy[c.key] = static_cast<long long>( c.global_count );

        long long pairs = 0;
        const auto& neighbors = _impl->commPlan().p2p_plan().neighbor_lists;
        for ( const auto& entry : neighbors )
        {
            const auto target = occupancy.find( entry.first );
            if ( target == occupancy.end() )
                continue;
            long long sources = 0;
            for ( const auto& key : entry.second )
            {
                const auto source = occupancy.find( key );
                if ( source != occupancy.end() )
                    sources += source->second;
            }
            pairs += target->second * sources;
        }
        return pairs;
    }

    /**
     * @brief The whole round trip, for either contraction.
     *
     * The six steps of "Two decompositions" in the file header, in order. Both
     * evaluations share every one of them; only `contraction` and
     * `coefficient` differ, so there is one place where a step can be got
     * wrong rather than two that can drift apart.
     *
     * **Collective structure.** Every rank runs the same sequence the same
     * number of times, including a rank that owns zero sources: the only
     * branch is on `reuse`, which is the result of an `MPI_LAND` reduction,
     * never on a local count. Canopy has no test for a zero-particle rank and
     * Beatnik's decomposition can produce one, so the discipline matters here
     * rather than being inherited.
     *
     * **Maintenance runs before every solve, unconditionally.** Canopy's
     * `solve()` uses the leaf membership, communication plan and P2P
     * neighbour lists cached by the last setup or maintenance call, so a
     * particle that has moved out of its leaf still contributes to its old
     * leaf's multipole and still gets its old leaf's near-field list, and
     * nothing raises. There is deliberately no "the positions barely moved"
     * fast path: that precondition can only be checked upstream of here.
     */
    void evaluate( const point_view& sources, const vector_view& charges,
                   const ZModelParams& zparams, Contraction contraction,
                   Real coefficient, const vector_view& vector_out,
                   const scalar_view& scalar_out )
    {
        // The one-integer tag is only meaningful because sources, targets and
        // output rows are the same rows. `ZModelParams::source_quadrature`
        // defaults to `Face`, whose `generate` still throws, so today that
        // equality is protected only by an unimplemented stub -- the guard
        // belongs here rather than being inherited from it.
        if ( zparams.source_quadrature != SourceQuadrature::Vertex )
            throw std::runtime_error(
                std::string( "FarFieldSolver: the far field requires "
                             "--source-quadrature vertex, but '" ) +
                toString( zparams.source_quadrature ) +
                "' is in force. Source, target and output row are one index "
                "on this path, which holds only for the vertex rule." );

        const int ns = static_cast<int>( sources.extent( 0 ) );

        if ( !_impl->haveSolver() )
        {
            _impl->createSolver( _comm, buildConfig( zparams ) );
            _softening = std::sqrt( zparams.blob() );
            _diagnostics.softening = _softening;
        }
        else if ( std::sqrt( zparams.blob() ) != _softening )
        {
            // Canopy fixes the softening at construction and it reaches the
            // M2L operator tables, so a changed blob would silently be a
            // different kernel in the far field than in the near field.
            throw std::runtime_error(
                "FarFieldSolver: the softening length changed between "
                "evaluations, from " +
                std::to_string( _softening ) + " to " +
                std::to_string( std::sqrt( zparams.blob() ) ) +
                ". Canopy fixes it at solver construction." );
        }

        // Step 2: pack the fresh state in mesh order.
        packSources( sources, charges, ns );

        // Step 3: route it into Canopy's ordering, then maintain the tree.
        int reuse = _first_call ? 0 : 1;
        if ( reuse )
        {
            int valid_local = 0;
            auto forward = buildForwardDistributor( ns, valid_local );
            int valid_global = 0;
            MPI_Allreduce( &valid_local, &valid_global, 1, MPI_INT, MPI_LAND,
                           _comm );
            if ( valid_global )
            {
                aosoa_type migrated( "beatnik_far_field_canopy_particles",
                                     forward.totalNumImport() );
                Cabana::migrate( forward, _mesh_particles, migrated );
                _canopy_particles = migrated;
                _diagnostics.maintenance =
                    _impl->autoMaintain( _canopy_particles );
            }
            else
            {
                // The source set changed shape since the last evaluation --
                // refinement, coarsening or a mesh load balance. The tags no
                // longer address it, so the tree is rebuilt from scratch
                // rather than fed a partial array.
                reuse = 0;
            }
        }
        if ( !reuse )
        {
            _canopy_particles.resize( static_cast<std::size_t>( ns ) );
            Cabana::deep_copy( _canopy_particles, _mesh_particles );
            _impl->setup( _canopy_particles, ns );
            _diagnostics.maintenance = FarFieldDiagnostics::Maintenance::Setup;
            _first_call = false;
        }

        // Step 4: one traversal, gradient on.
        _impl->solve( _canopy_particles );
        const int num_local = _impl->numLocalParticles();

        // Step 5: contract the 3x3 tensor in place, on Canopy's ordering.
        {
            auto gradient = _impl->gradient();
            auto out =
                Cabana::slice<FarFieldMember::Output>( _canopy_particles );
            const Real c = coefficient;
            if ( contraction == Contraction::Curl )
            {
                Kokkos::parallel_for(
                    "beatnik_far_field_curl",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
                    KOKKOS_LAMBDA( const int p ) {
                        // u_i = -eps_ijk T_kj with T_cj = gradient(p, c, j).
                        // Swapping either index of a pair negates the field.
                        out( p, 0 ) =
                            c * ( gradient( p, 1, 2 ) - gradient( p, 2, 1 ) );
                        out( p, 1 ) =
                            c * ( gradient( p, 2, 0 ) - gradient( p, 0, 2 ) );
                        out( p, 2 ) =
                            c * ( gradient( p, 0, 1 ) - gradient( p, 1, 0 ) );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "beatnik_far_field_trace",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
                    KOKKOS_LAMBDA( const int p ) {
                        // Psi = -tr T, into component 0. Components 1 and 2
                        // are unused on this path: one tuple layout serves
                        // both contractions so the tag and the round trip have
                        // one definition.
                        out( p, 0 ) =
                            -c * ( gradient( p, 0, 0 ) + gradient( p, 1, 1 ) +
                                   gradient( p, 2, 2 ) );
                        out( p, 1 ) = Real( 0 );
                        out( p, 2 ) = Real( 0 );
                    } );
            }
            Kokkos::fence();
        }

        // Step 6: reverse-distribute on tag.origin_rank and scatter by
        // tag.owned_index.
        Cabana::Distributor<MemorySpace> reverse = [&] {
            Kokkos::View<int*, MemorySpace> origin_ranks(
                Kokkos::ViewAllocateWithoutInitializing(
                    "beatnik_far_field_origin_ranks" ),
                num_local );
            auto tag =
                Cabana::slice<FarFieldMember::Tag>( _canopy_particles );
            Kokkos::parallel_for(
                "beatnik_far_field_origin_ranks",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
                KOKKOS_LAMBDA( const int p ) {
                    origin_ranks( p ) = tag( p, FarFieldMember::OriginRank );
                } );
            Kokkos::fence();
            return Cabana::Distributor<MemorySpace>( _comm, origin_ranks );
        }();

        aosoa_type returned( "beatnik_far_field_returned",
                             reverse.totalNumImport() );
        Cabana::migrate( reverse, _canopy_particles, returned );

        const int num_returned = static_cast<int>( returned.size() );
        int scatter_bad = 0;
        {
            auto tag = Cabana::slice<FarFieldMember::Tag>( returned );
            auto out = Cabana::slice<FarFieldMember::Output>( returned );
            if ( contraction == Contraction::Curl )
            {
                auto u = vector_out;
                Kokkos::parallel_reduce(
                    "beatnik_far_field_scatter_vector",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, num_returned ),
                    KOKKOS_LAMBDA( const int p, int& bad ) {
                        const int i = tag( p, FarFieldMember::OwnedIndex );
                        if ( i < 0 || i >= ns )
                        {
                            ++bad;
                            return;
                        }
                        for ( int d = 0; d < 3; ++d )
                            u( i, d ) = out( p, d );
                    },
                    scatter_bad );
            }
            else
            {
                auto psi = scalar_out;
                Kokkos::parallel_reduce(
                    "beatnik_far_field_scatter_scalar",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, num_returned ),
                    KOKKOS_LAMBDA( const int p, int& bad ) {
                        const int i = tag( p, FarFieldMember::OwnedIndex );
                        if ( i < 0 || i >= ns )
                        {
                            ++bad;
                            return;
                        }
                        psi( i ) = out( p, 0 );
                    },
                    scatter_bad );
            }
            Kokkos::fence();
        }

        // One reduction for every global diagnostic and for the round-trip
        // check, rather than one apiece: they are all sums over the same
        // communicator at the same point in the sequence.
        _impl->readDiagnostics( _diagnostics );
        long long local[5] = {
            static_cast<long long>( num_local ),
            p2pPairCountLocal(),
            _diagnostics.global_m2l_pair_count,
            _diagnostics.global_m2l_fallback_pair_count,
            static_cast<long long>( num_returned != ns ) +
                static_cast<long long>( scatter_bad ) };
        long long global[5] = { 0, 0, 0, 0, 0 };
        MPI_Allreduce( local, global, 5, MPI_LONG_LONG, MPI_SUM, _comm );

        if ( global[4] != 0 )
            throw std::runtime_error(
                "FarFieldSolver: the far-field round trip did not return "
                "exactly one result per owned source on every rank (" +
                std::to_string( global[4] ) +
                " mismatched or out-of-range results globally). A dropped or "
                "duplicated source is a wrong velocity, not a slow one." );

        _diagnostics.global_particle_count = global[0];
        _diagnostics.global_p2p_pair_count = global[1];
        _diagnostics.global_m2l_pair_count = global[2];
        _diagnostics.global_m2l_fallback_pair_count = global[3];
        const double n = static_cast<double>( global[0] );
        _diagnostics.p2p_pair_fraction =
            ( n > 0.0 ) ? static_cast<double>( global[1] ) / ( n * n ) : 0.0;
    }
#else
    /// The `~canopy` face of the two evaluations. A configuration error, not
    /// an unwritten routine, so `runtime_error` and not `logic_error` — the
    /// same distinction `Beatnik_CreateBRSolver.hpp:45-49` draws.
    [[noreturn]] static void throwNoCanopy( const char* method )
    {
        throw std::runtime_error(
            std::string( "FarFieldSolver::" ) + method +
            " requires Beatnik to be built with Canopy support "
            "(Beatnik_ENABLE_CANOPY=ON, spack '+canopy'). Use "
            "--br-approximation direct instead." );
    }
#endif

    MPI_Comm _comm;
    int _rank = 0;
    FmmParams _params;
    FarFieldDiagnostics _diagnostics;

#ifdef BEATNIK_ENABLE_CANOPY
    /// The dispatch arm in force. Constructed once, in the constructor, and
    /// persistent thereafter — the tree, the partition and the communication
    /// plan it holds are the thing being reused across evaluations.
    std::unique_ptr<Impl> _impl;

    /// One tuple per owned source in the caller's order. Repacked from the
    /// caller's views on every evaluation.
    aosoa_type _mesh_particles{ "beatnik_far_field_mesh_particles", 0 };

    /// The same tuples in Canopy's ordering and on Canopy's ranks. Permuted
    /// and migrated by Canopy on every maintenance path; the tag is what
    /// survives that and lets the pair above be routed into it.
    aosoa_type _canopy_particles{ "beatnik_far_field_canopy_particles", 0 };

    /// Set until the first successful `setup()`. After that the reuse branch
    /// applies, subject to the forward distributor's own validation.
    bool _first_call = true;

    /// The softening **length** handed to Canopy, held so a later evaluation
    /// presenting a different blob is an error rather than a silent mismatch
    /// between the near and far fields.
    Real _softening = 0.0;
#endif
};

} // namespace Beatnik

#endif // BEATNIK_FARFIELDINTERFACE_HPP
