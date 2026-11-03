# Milestone 1 — progress log

Session record for milestone1. Companion to [`milestone1.md`](milestone1.md),
which holds the design, the task sequence and the risks; this file holds what
actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `milestone1.md` can cite it by
ID. No dates: the order of the sections is the chronology. If a session covers
more than one task, name them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

Two things this milestone specifically wants in the log, because its design
says a session must decide them rather than inherit them: for every pass that
moves a vertex or changes connectivity, **whether it re-bases the AMR reference
state and why**; and for every measurement, the **17-digit literals**, not a
characterization of them.

The T4d1–T4d6 design moved here from [`framework.md`](framework.md); entries
about T4d written before that move are in
[`framework-progress-log.md`](framework-progress-log.md) — chiefly the T4a, T4b
and T4c entries, which carry the measured shape statistics and the reconciliation
against Tessera that T4d's sub-tasks build on.

## M1-U1

Tessera `279287e` on `mesh-quality`: `DefaultCollapsePolicy` and
`DefaultFlipPolicy` each gain a `quality()` hook, and the hook is what
`minQuality` thresholds. One new test case in each of `test_collapse_edges.cpp`
and `test_flip_edges.cpp`. `README.md` and `docs/design.md` describe the hook. No
Beatnik source changed.

**Decisions, made before the session and recorded here as given:**

- **Tessera build profile.** `manual` mode, build dir `tessera/build-tuolumne`,
  dependencies from `~/spack_envs/tuolumne_trilinos`. These are
  `scripts/tuolumne/profile.defaults.sh`'s values, written to the gitignored
  `scripts/tuolumne/profile.local.sh` so the checkout records the choice.
- **Build only the four test executables** the exit criterion runs, never a bare
  `make`. Tessera is header-only (`add_library(Tessera INTERFACE)`,
  `CMakeLists.txt:74`), so `tessera_test_{collapse_edges,flip_edges}_{SERIAL,HIP}`
  compile every changed line.
- **Targeted run, not Tessera's full gate.** The changed symbols — the two
  default policies, `detail::flipRadiusRatio` and `detail::flipTriNormal` — are
  used nowhere in `src/` outside `Tessera_EdgeCollapse.hpp` and
  `Tessera_EdgeFlip.hpp`. Only `test_collapse_edges.cpp` and
  `test_flip_edges.cpp` call `collapseEdges`/`flipEdges`, and Tessera's
  `docs/testing.md` ("When to run the full gate") makes a targeted run the
  default for a change this narrow.
- **The hook's signature and default** are those of the task's **Do** step 1.
  Default behavior is bit-identical: the default `quality` calls the same
  function the tests called before, on the same arguments.

**Decisions made in the session:**

- **Each new case runs one direction, and the pair covers both.** Collapse goes
  `accepted` → `rejectedQuality`, and flip goes `rejectedGeometric` →
  `accepted`. The direction follows the shape. On a needle-like face the radius
  ratio exceeds the mean ratio (about `2ε` against `√3·ε`); on an obtuse cap it
  is far below (about `2h²` against `1.15·h`). The collapse fixture's worst faces
  are needle-like, and the flip fixture's worst face is a 120° cap.
- **Collapse fixture: case 7's `spikeBipyramidSoup(10)`, unchanged**, with ring
  edge `(2,3)` marked. Case 7's comment calls the ring edges benign, and the fold
  guard does pass with margin: the worst normal turn is `0.19900807528636058`
  rad, under `maxNormalRotation = 0.5`. The link condition holds, since
  `link(2) ∩ link(3) = {0,1}`. Depth 2, as every collapse case uses.
- **Flip fixture: a new `flatBipyramidSoup`**, a unit hexagonal ring with the top
  apex at `z = 0.1` and the bottom apex at `z = -1` (V 8, E 18, F 12). Spoke
  `(0,2)` is marked. The new faces are `(0,7,3)` and the ring triangle `(2,3,7)`,
  and `(3,7)` is not an existing edge. The normal deviations are `0.0977` and
  `0.0997` rad, under `maxNormalDeviation = 0.35`, so only the quality test
  decides. Depth 1, as every other flip case uses.
- **Thresholds:** `0.47` for collapse and `0.55` for flip, each set between the
  two metrics' worst values. Each case recomputes both metrics from the soup
  coordinates and asserts the separation before it trusts the verdicts, so a
  drifted fixture fails as a fixture rather than as the hook.
- **The tests define their own `meanRatio`** (`4√3·A/Σℓ²`) and a
  `MeanRatio{Collapse,Flip}Policy` that derives from the default and shadows
  only `quality`. This is the shape of T4d2's policy. It is test code only: the
  reference metric's production policy is T4d2's work, in Beatnik.

**Signatures changed.** Each of the two structs gains one member:
`double quality( const double* p0, const double* p1, const double* p2 ) const`.
It is host-side, and `p0..p2` are 3-vectors. The default returns
`detail::flipRadiusRatio( p0, p1, p2 )`, in `[0,1]` with 1 for an equilateral
triangle. The three comparisons now call `policy.quality( … )`, with their
directions unchanged: collapse rejects `quality < minQuality`
(`Tessera_EdgeCollapse.hpp:1005-1007`), and flip requires `>= minQuality` on
both new faces (`Tessera_EdgeFlip.hpp:676-678`). The `collapseEdges` and
`flipEdges` signatures are unchanged.

**Where `flipRadiusRatio` ended up.** It moved, together with `flipTriNormal`,
which it calls, into a new `detail` block directly above `DefaultFlipPolicy`
(`Tessera_EdgeFlip.hpp:260-302`). They were moved, not duplicated.
`flipAngleBetween` stays where it was, since no policy member calls it.
`Tessera_EdgeCollapse.hpp` sees both through its include of
`Tessera_EdgeFlip.hpp`. These edits shift line numbers, so the references in
`milestone1.md`'s Read this first #6 and in its M1-U2 and T4d1 entries were
refreshed to `279287e`.

**Fixture values (17-digit literals, identical in all launches):**

| Case | Worst radius ratio | Worst mean ratio | `minQuality` | Default | Mean-ratio policy |
| --- | --- | --- | --- | --- | --- |
| collapse case 15 | `0.48306530887389387` | `0.46367649590853621` | `0.46999999999999997` (0.47) | `accepted=1` | `rejectedQuality=1` |
| flip case 13 | `0.46410161513775444` | `0.59999999999999987` | `0.55000000000000004` (0.55) | `rejectedGeometric=1` | `accepted=1` |

The worst collapse faces are the spike pair `(0,3,4)` and `(0,11,2)`, with the
merged vertex at the midpoint of `(2,3)`. The bottom pair scores
`0.86543920025308263` (radius ratio) and `0.867016717774858` (mean ratio). The
worst flip face is the ring triangle `(2,3,7)`, whose mean ratio is analytically
`3/5`. The shallow top face `(0,7,3)` scores `0.47661290802558659` and
`0.60944456337762387`. A Python transcription of the library's formulas, run
before the build, gave the same values bit for bit as the C++ printout.

**Bugs only running revealed.** Two, neither in the library.

- **`detail` is ambiguous in a test TU on the HIP build.** The first build
  failed with `reference to 'detail' is ambiguous`: under
  `using namespace Tessera`, an unqualified `detail::flipRadiusRatio` in test code
  also finds ROCm's `rocprim` `::detail`
  (`/opt/rocm-6.4.2/include/rocprim/device/../common.hpp:24`). The fix is to
  write `Tessera::detail::` in the tests. Inside the library headers the
  unqualified `detail::` is unambiguous, because lookup from within namespace
  `Tessera` stops at `Tessera::detail`. **T4d2:** any Beatnik code that calls
  `Tessera::detail::*` must qualify it fully.
- **`flux batch --flags=waitable` is refused** on the tuolumne system instance
  (`only the instance owner can submit with FLUX_JOB_WAITABLE`). The job was
  submitted without the flag and waited on with `flux job status`, which
  returned rc 0.

**Provenance.**

- Tessera commit tested: `279287e6a0591146f5c0f5ed1fa208de70e3cdc8`
  (`mesh-quality`), committed before the submit.
- `spack env status`: `==> In environment /g/g20/stewartj/spack_envs/tuolumne_trilinos`.
- Configure: `spack env activate ~/spack_envs/tuolumne_trilinos/`, then
  `mkdir build-tuolumne && cd build-tuolumne && bash ../run_cmake_toulumne.sh`.
- Build, in `build-tuolumne`:
  `make -j tessera_test_collapse_edges_SERIAL tessera_test_collapse_edges_HIP tessera_test_flip_edges_SERIAL tessera_test_flip_edges_HIP`.
  rc 0, with no warning from the changed files; the only warnings are the
  `-Wnontrivial-memcall` ones from `Tessera_HaloRebuild.hpp:99`.
- Submit, from the Tessera root with `TESSERA_REPO=$(pwd)` exported:
  `flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`.
  Job `f3hmiTasm3Ef`, `pdebug`, COMPLETED rc 0, 182 s.
- Result: `100% tests passed, 0 tests failed out of 20`, which is ctest tests
  #251–260 (`flip_edges`) and #271–280 (`collapse_edges`), SERIAL and HIP, np1–5.
  The output is `tessera-unit.f3hmiTasm3Ef.out` in the Tessera root. The case
  lines come from `build-tuolumne/Testing/Temporary/LastTest.log`, because the
  runner's `--output-on-failure` prints nothing for a passing test.

**Re-basing:** not applicable. M1-U1 touches no Beatnik pass.

**Affects:**

- **T4d2:** build the reference-metric collapse policy on this hook. Derive from
  `Tessera::DefaultCollapsePolicy`, shadow
  `double quality( const double*, const double*, const double* ) const` with
  `4√3·A/Σℓ²`, and set `minQuality`, `maxNormalRotation` and `t` explicitly, as
  its **Do** step 4 says. The tests' `MeanRatioCollapsePolicy` is a working
  template. Qualify any `Tessera::detail::` use fully.
- **M1-U2:** edits the same `DefaultFlipPolicy`, which now has a `quality()`
  member. Rebase onto `279287e`; the entry's line references are refreshed to
  it. `test_flip_edges.cpp` case 13 constructs a `DefaultFlipPolicy` and a
  derived `MeanRatioFlipPolicy`, and both must keep the `SharedFace` path
  bit-identical.
- **T4d3, T4d5:** the permissive flip policy `{maxNormalDeviation = π,
  minQuality = 0}` needs no `quality` override. With `minQuality = 0`, the
  radius ratio cannot fail it.

## M1-U2

Tessera `05b5942` on `mesh-quality`: `DefaultFlipPolicy` gains
`FlipConflict conflict = FlipConflict::SharedFace`, and `flipEdges`' step 3 runs
a vertex-coordinator round under `FlipConflict::EndpointInQuad`.
`detail::vertexCoordRank` moves into `Tessera_EdgeFlip.hpp`. One new case,
`test_flip_edges` case 14. `README.md` and `docs/design.md` state both relations.
No Beatnik source changed.

**Decisions, made before the session and recorded here as given:**

- **Tessera build profile.** As M1-U1 recorded it in the gitignored
  `scripts/tuolumne/profile.local.sh`: `manual`, `build-tuolumne`,
  `~/spack_envs/tuolumne_trilinos`. The existing tree was rebuilt incrementally
  and not reconfigured.
- **Build only the four test executables**,
  `tessera_test_{collapse_edges,flip_edges}_{SERIAL,HIP}`, never a bare `make`.
  Tessera is header-only (`CMakeLists.txt:74`), so these compile every changed
  line. `collapse_edges` is in the set because `vertexCoordRank` left its header.
- **Targeted run, not Tessera's full gate.** The changed symbols are
  `DefaultFlipPolicy`, the step-3 round and `detail::vertexCoordRank`.
  `vertexCoordRank`'s only callers are in `Tessera_EdgeFlip.hpp` and
  `Tessera_EdgeCollapse.hpp`, and only `test_collapse_edges.cpp` and
  `test_flip_edges.cpp` call `collapseEdges`/`flipEdges`. Tessera's
  `docs/testing.md` ("When to run the full gate") makes a targeted run the
  default for a change whose blast radius reading bounds.
- **`conflict` is a plain member of `DefaultFlipPolicy`**, read as
  `policy.conflict`. A derived policy inherits it, and `flipEdges`' signature is
  unchanged.
- **`vertexCoordRank` is moved, not duplicated**, into `Tessera_EdgeFlip.hpp`'s
  `detail` namespace. `Tessera_EdgeCollapse.hpp` includes that header (`:18`).

**Decisions made in the session:**

- **The endorsement rule is the task's Do step 2, verbatim.** Each surviving
  candidate sends a `FlipCornerOffer {v, key, len2, endpoint}` to
  `vertexCoordRank(v)` for each of its four corners `lo, hi, w, x`. The
  coordinator finds the best `B` by `(len2 DESCENDING, EdgeKey ASCENDING)`. If
  `B.endpoint` is set, it endorses `B` alone; otherwise it endorses every offer
  with `endpoint == 0`. Each endorsement is a `FlipCornerVote {key}` sent to
  `edgeCoordRank(key)`. A candidate survives iff `votes == 4`, and every loser
  counts into `rejectedConflict`. Under `EndpointInQuad` the per-face round does
  not run. The header states the proof that no conflicting pair is accepted.
- **The `SharedFace` path is the old block, unchanged.** It sits behind
  `if ( policy.conflict == FlipConflict::EndpointInQuad ) { … } else`, and its
  body was not reindented. `git diff 279287e 05b5942 -- src/Tessera_EdgeFlip.hpp`
  removes only comment lines.
- **`vertexCoordRank`'s doc comment now names the flip round too.** It says "the
  flip and collapse rounds" where it said "the collapse rounds". The function
  body is unchanged.
- **The case uses the permissive policy** `{maxNormalDeviation = π,
  minQuality = 0}`, the one T4d5 will pass. So the independent-set round alone
  decides on the fixtures, and every other counter must be 0.
- **The failure direction is covered by two pairs, not one.** There is one pair
  per branch of the endorsement rule: an endpoint-role winner, and an
  opposite-role winner that shuts out an endpoint-role loser. The task asked for
  one fixture with one pair. Each pair still marks exactly two flips.
- **Rank invariance is asserted, not only observed.** The all-marked icosphere
  runs under `EndpointInQuad` on `MPI_COMM_WORLD` and on `MPI_COMM_SELF`, and
  the two gathered flip sets must be equal as sets of `(old key, new key)`.
  Vertex gids equal the soup index in `buildFromTriangleSoup`, so keys compare
  across communicators. For the bipyramid pairs, the exact accepted key set is
  asserted at every rank count.
- **A Python model was run before the build.** It transcribed the fixture and
  both relations, and gave the accepted sets the case asserts.

**Signatures changed.**

- New `enum class FlipConflict { SharedFace, EndpointInQuad }` in namespace
  `Tessera`, directly above `DefaultFlipPolicy` (`Tessera_EdgeFlip.hpp:337-344`).
- New member `FlipConflict conflict = FlipConflict::SharedFace;` on
  `DefaultFlipPolicy` (`:352-371`).
- `detail::vertexCoordRank( GlobalId, int )` is at `Tessera_EdgeFlip.hpp:463-466`
  and no longer in `Tessera_EdgeCollapse.hpp`.
- New message types `detail::FlipCornerOffer` and `detail::FlipCornerVote`
  (`:471-486`).
- Step 3 is `:764-872`. The `EndpointInQuad` branch is `:775-827`, and the
  `SharedFace` branch is the `else` at `:828`.
- `flipEdges` and `FlipResult` are unchanged, apart from the doc comments on
  `rejectedConflict` and on `flipEdges`, which now name both relations.

**Fixtures (identical in all twenty `flip_edges` launches, Serial and Default
spaces):**

The two-ring bipyramid has `T = 0` at `z = 1.5` and `B = 1` at `z = -0.6`. The
ring `R_i = 2+i` has radius 1 at `z = 0.5`, and the ring `S_i = 10+i` has radius
0.6 at `z = -0.5`, for `i = 0..7`. The band diagonal is `(R_{i+1}, S_i)`. V 18,
E 48, F 32. The marked edges, as endpoints | opposite corners, with squared
length:

- `(2,3)`: `| 0, 10`, length² `2 - √2 = 0.586`.
- `(6,7)`: `| 0, 14`, length² `0.586`.
- `(0,6)`: `| 5, 7`, length² `2`.
- `(1,10)`: `| 11, 17`, length² `0.37`.

No pair shares a face, and no flip creates an existing edge.

| Fixture | `SharedFace` | violations in its set | `EndpointInQuad` | violations |
| --- | --- | --- | --- | --- |
| shared opposite `(2,3)+(6,7)` | accepted 2 | 0 | accepted 2 `{(2,3) (6,7)}` | 0 |
| endpoint wins `(2,3)+(0,6)` | accepted 2 | 1 | accepted 1 `{(0,6)}`, `rejectedConflict=1` | 0 |
| opposite wins `(2,3)+(1,10)` | accepted 2 | 1 | accepted 1 `{(2,3)}`, `rejectedConflict=1` | 0 |
| icosphere, all 480 marked | accepted 80, `rejectedConflict=400` | 160 | accepted 28, `rejectedConflict=452`; SELF accepted 28, set identical | 0 |

A violation is an ordered pair `(E1, E2)` of distinct accepted flips with an
endpoint of `E1` among `E2`'s corners, computed from the gathered
`FlipResult::flipped`. On the icosphere, `EndpointInQuad` accepts 28 of the 80
flips `SharedFace` takes in one round. T4d5's multi-round passes are what make
up the difference.

**Bugs only running revealed.** None. The build succeeded first time
(`make` rc 0, 9 min 47 s). The run passed first time.

**Findings from reading and running, not fixed here:**

- **Two flips that create the same new diagonal are both accepted, under
  either relation.** Take accepted candidates `(a,b | w,x)` and `(c,d | w,x)`
  with `{a,b} ∩ {c,d} = ∅`. They share no face, and neither has an endpoint
  among the other's corners. The duplicate-edge round (step 2b) checks only the
  pre-flip edge set, so both pass, and the mesh ends with two edges keyed
  `(w,x)`. M1-U1's `flatBipyramidSoup` has exactly this shape: marking ring
  edges `(2,3)` and `(5,6)` under the permissive policy would do it, since both
  have opposite corners `0, 1`. Some vertex pairs `w, x` are not
  adjacent but have four common neighbours that form two such quads. The flaw
  predates M1-U2, since `SharedFace` has it too. It was found by reading
  `Tessera_EdgeFlip.hpp` and has **not** been reproduced in a run. It is not in
  Tessera's README Known Issues.
- **`collapse_edges` case 14's printout depends on the rank count, and that
  predates this task.** It prints `final F=1244, 1640 total edits` at np1, 3
  and 5, and `final F=1236, 1664 total edits` at np2 and 4, on both backends.
  Its flip pass uses the default `SharedFace` relation behind a caller valence
  mask that skips an owned edge whose second face is not resident. The
  `test_flip_edges` header documents this as the caller's decomposition-dependent
  choice. The case asserts a face-count band and a quality floor (0.0333 at
  every rank count), not exact counts, and it passed at every rank count.

**Line references refreshed.** `milestone1.md`'s Read this first #6 now states
that its Tessera references are against `05b5942`:

- `flipRadiusRatio`: `Tessera_EdgeFlip.hpp:274-300` → `:305-331`.
- `DefaultFlipPolicy`: `:308-325` → `:352-371`.
- The geometric test: `:649-686` → `:725-762`.
- Step 3 and Decision 2: `:154-177` → `:154-206`.
- The collapse fold guard: `Tessera_EdgeCollapse.hpp:1000-1004` → `:991-995`.
- `Tessera_EdgeFlip.hpp:142-150` and `Tessera_EdgeCollapse.hpp:230-236` are
  unchanged.

T4d1's `Tessera_EdgeCollapse.hpp:337-341` is unchanged, and the entry now names
`05b5942`. T4d3 and T4d5 cite no `Tessera_EdgeFlip.hpp` or
`Tessera_EdgeCollapse.hpp` line. M1-U2's own stale references were corrected:
`test_flip_edges.cpp:1061` → `:1067`, and `CollapseNbr`/`CollapseWin`
`:550-568` → `:552-567`, both against `279287e`. The "Current state" bullet that
called the endpoint relation absent upstream now says it is present and unused.

**Provenance.**

- Tessera commit tested: `05b5942de1297075bd4afbd26341daace914beba`
  (`mesh-quality`), committed before the submit.
- `spack env status`: `==> In environment /g/g20/stewartj/spack_envs/tuolumne_trilinos`.
- Build: `spack env activate ~/spack_envs/tuolumne_trilinos/`, then, in
  `build-tuolumne`,
  `make -j tessera_test_collapse_edges_SERIAL tessera_test_collapse_edges_HIP tessera_test_flip_edges_SERIAL tessera_test_flip_edges_HIP`.
  rc 0. The only warnings are the `-Wnontrivial-memcall` ones from
  `Tessera_HaloRebuild.hpp:99`.
- Submit, from the Tessera root with `TESSERA_REPO=$(pwd)` exported:
  `flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`.
  Job `f3hn6UjNexxw`, `pdebug`, COMPLETED rc 0, 193 s. `flux job status` returned
  rc 0.
- Result: `100% tests passed, 0 tests failed out of 20`. These are ctest tests
  #251–260 (`flip_edges`) and #271–280 (`collapse_edges`), SERIAL and HIP,
  np1–5. The output is `tessera-unit.f3hn6UjNexxw.out` in the Tessera root. The
  case lines come from `build-tuolumne/Testing/Temporary/LastTest.log`.

**Re-basing:** not applicable. M1-U2 touches no Beatnik pass.

**Affects:**

- **T4d5:** call `flipEdges` with a policy deriving from or equal to
  `Tessera::DefaultFlipPolicy`, with `maxNormalDeviation = π`,
  `minQuality = 0` and `conflict = Tessera::FlipConflict::EndpointInQuad`. Gather
  `FlipResult::flipped` for the touched set: `first.id[0..1]` are the
  endpoints and `second.id[0..1]` the opposite corners. One round accepts far
  fewer flips than `SharedFace` (28 against 80 on the all-marked icosphere), so
  the multi-round pass is required. T4d5's **Do** step 1 says not to test
  whether `(c,d)` exists. Within one round, though, two accepted flips can
  create the same diagonal (finding above). Either Tessera closes that gap first
  or T4d5's mask must not mark two edges whose quads share both opposite
  corners. Decide before T4d5 lands.
- **M1-T2:** a milestone trajectory whose valence pass hits the
  same-diagonal case would fail conformity mid-run rather than diverge. Read a
  conformity failure in the cleanup flips with that finding in mind.
- **T4d3:** none. The quality flips keep the default `SharedFace` relation.

## M1-U3 (design only)

The same-new-diagonal finding in `## M1-U2` is now fixed in Tessera rather than
in Beatnik's masks. A mask cannot see a second edge with the same opposite
corners on another rank, and the defect corrupts the mesh for every caller of
`flipEdges`, under both relations. `milestone1.md` gains task **M1-U3**:

- the `(w,x)` coordinator in step 2b resolves the competing claims, using the
  existing priority, among the candidates that pass geometry;
- the losers count into `rejectedConflict`;
- no new collective is added;
- on any input without such a pair, every verdict and counter is unchanged.

No code changed.

**Affects:**

- **T4d3, T4d5, M1-T2:** each now depends on M1-U3.
- **T4d5:** the open question in `## M1-U2`'s **Affects:** line is settled, and
  the mask need not avoid pairs of edges whose quads share both opposite corners.

## M1-U3

**Commits:** Tessera `33dc6b9` (`mesh-quality-auto`); Beatnik `a02631f`
(`mesh-quality-auto`, `milestone1.md` and this section).

`flipEdges` accepts at most one of several flips in one call that would create
the same new diagonal, under both `FlipConflict` relations. The `(w,x)`
coordinator settles the claims inside step 2b's duplicate-edge round. One new
case, `test_flip_edges` case 15. The header, `docs/design.md` ("The duplicate-edge
round") and `README.md`'s `rejected*` comments describe the rule. No Beatnik
source changed.

Both repositories were checked out on `mesh-quality-auto`, not `mesh-quality` as
the session prompt said; the two branches were at the same SHA in each
repository. The commits are on `mesh-quality-auto`. Nothing was pushed.

**Decisions, made before the session and recorded here as given:**

- **The fix is in Tessera, not in any Beatnik mask.** A mask cannot see a second
  edge with the same opposite corners on another rank, and the defect corrupts
  the mesh for every caller.
- **The claims are resolved at the `(w,x)` coordinator inside step 2b's existing
  round**, with no new collective. The reply is *exists*, *free* or *claimed*
  (the task's **Do** step 2), and a *claimed* candidate counts into
  `rejectedConflict`.
- **Verdict precedence stays boundary, then duplicate, then geometric, then
  conflict.** Geometry is evaluated before 2b only so its pass/fail can travel
  in the `DupQuery`. Only geometric survivors compete for a diagonal.
- **The fix applies under both `SharedFace` and `EndpointInQuad`.**
  `flipEdges`' signature and `FlipResult`'s members do not change.
- **Tessera build profile** as recorded in the gitignored
  `scripts/tuolumne/profile.local.sh` (`manual`, `build-tuolumne`,
  `~/spack_envs/tuolumne_trilinos`). The tree was rebuilt incrementally, not
  reconfigured.
- **Build only the four test executables**,
  `tessera_test_{collapse_edges,flip_edges}_{SERIAL,HIP}`, never a bare `make`.
  Tessera is header-only (`CMakeLists.txt:74`). `collapse_edges` is in the set
  because its case 14 calls `flipEdges`.
- **Targeted run, not Tessera's full gate.** The change is confined to
  `flipEdgesImpl` and its private message types, and only
  `test_flip_edges.cpp` and `test_collapse_edges.cpp` call `flipEdges`.
  Tessera's `docs/testing.md` ("When to run the full gate") makes a targeted run
  the default for a change this narrow.

**Decisions made in the session:**

- **Step 2 has three labelled parts.** 2a evaluates the geometric test for every
  boundary survivor and stores it in `Cand::geomOk`, erasing nothing. 2b is the
  duplicate and claim round; it erases the *exists* candidates into
  `rejectedDuplicateEdge` and records the *claimed* ones. 2c attributes the rest
  in order: a geometric failure into `rejectedGeometric`, then a *claimed*
  candidate into `rejectedConflict`. The geometric code is the old 2c's,
  unchanged apart from writing `c.geomOk` instead of erasing.
- **The coordinator's claim winner is computed over the queries it received**
  (`asked.data`), per target, skipping askers that failed geometry and targets
  that exist. The comparison is the step-3 priority, `len2` greater, or equal
  `len2` and smaller `EdgeKey`, so no gid enters it.
- **Bit-identity off the defect holds by construction.** The geometric verdict
  is the same computation on the same inputs. A candidate is answered *claimed*
  only when another geometric survivor asked for the same non-existing target,
  so on any input without such a pair the replies are *exists* or *free*
  exactly where the old boolean was 1 or 0.
- **The flip doc comment on `flipEdges` now says** that under either relation
  no two accepted flips create the same new edge, beside its independent-set
  sentence.
- **The case's failure fixture is a perturbed copy, `pushedBipyramidSoup()`**,
  which calls `flatBipyramidSoup` and scales vertex 5's `x` and `y` by
  `kPushOut = 2`. `flatBipyramidSoup` itself is untouched, so case 13 is
  unaffected. A new helper, `diagonalRun`, mirrors case 14's `conflictRun` for
  the V 8 fixtures and takes `minQuality`.
- **The success winner is asserted, not only compared.** Besides equality with
  the `MPI_COMM_SELF` flip set, the case predicts the winner from the soup
  coordinates through `edgeLen2Canonical` and the priority, and asserts the
  gathered set is `{(winner, (0,1))}`.

**Signatures and message types changed.** All private to `flipEdgesImpl`:

- `detail::DupQuery` gains `double len2` and `unsigned char geomOk`
  (`Tessera_EdgeFlip.hpp:444-456`).
- New `enum class detail::DupStatus : unsigned char { Free, Exists, Claimed }`
  (`:458-463`).
- `detail::DupReply::exists` (an `unsigned char` boolean) is replaced by
  `DupStatus status` (`:465-469`). No reference to the old member remains in
  `src/` or `tests/`.
- The local `struct Cand` gains `bool geomOk`.
- `flipEdges`, `FlipResult` and `DefaultFlipPolicy` are unchanged. The
  `FlipResult::rejectedConflict` comment names the diagonal claim as a third way
  to lose, and the header's step-2 `DUPLICATE` text (`:142-178`) states the
  three replies, the precedence, and that the claim winner can still lose step 3
  with the shut-out claimant not reconsidered (Decision 3).

**The Python model, run before the build.** It transcribed `flatBipyramidSoup`,
the library's `edgeLen2Canonical` and `flipRadiusRatio` in their operation order,
and each candidate's `w`, `x` and new faces `(lo,x,w)`, `(hi,w,x)`. Its numbers
equal the C++ printout bit for bit:

| Soup | Edge | Squared length | New-face radius ratios |
| --- | --- | --- | --- |
| flat | `(2,3)` | `0.99999999999999978` | `0.87969658092961811`, `0.87969658092961833` |
| flat | `(5,6)` | `0.99999999999999922` | `0.87969658092961811`, `0.87969658092961833` |
| pushed, vertex 5 at radius 2 | `(2,3)` | `0.99999999999999978` | `0.87969658092961811`, `0.87969658092961833` |
| pushed, vertex 5 at radius 2 | `(5,6)` | `2.9999999999999987` | `0.73625936037355844`, `0.87969658092961833` |

Both edges have `w = 0`, `x = 1`, so both create `(0,1)`. Radius 1.5 was tried
in the model first. It separates the two by only `0.829` against `0.880`, so
radius 2 was chosen, with `minQuality = 0.8` (`0.80000000000000004`) between
`0.736` and `0.880`.

**Fixtures (identical in all twenty `flip_edges` launches, Serial and Default
spaces):**

| Fixture | `minQuality` | `SharedFace` | `EndpointInQuad` |
| --- | --- | --- | --- |
| flat, `(2,3)+(5,6)` | 0 | accepted 1 `{(2,3)→(0,1)}`, `rejectedConflict=1`; SELF set identical | accepted 1 `{(2,3)→(0,1)}`, `rejectedConflict=1`; SELF set identical |
| pushed, `(2,3)+(5,6)` | 0.8 | accepted 1 `{(2,3)}`, `rejectedGeometric=1`, `rejectedConflict=0` | accepted 1 `{(2,3)}`, `rejectedGeometric=1`, `rejectedConflict=0` |

`maxNormalDeviation = π` throughout, and `rejectedBoundary` and
`rejectedDuplicateEdge` are 0 in every run. Every run passes `checkAll`,
`checkCountersPartition` and `checkKeyTables`. Case 3 still prints
`requested=1 rejectedDuplicateEdge=1 accepted=0`.

**Existing cases.** Every `flip_edges` and `collapse_edges` case passed at its
unchanged literals. Of the three places that flip an already-edited mesh, the
printout of `collapse_edges` case 14 matches `## M1-U2` exactly: `final F=1244,
1640 total edits` at np1, 3 and 5, and `final F=1236, 1664 total edits` at np2
and 4, on both backends. Case 9's `mask2` call printed `secondAccepted=120` at
every rank count, and case 12 passed its floor. Neither prints a line that
`## M1-U2` recorded, so this run cannot show whether their counts moved, only
that their assertions hold.

**Bugs only running revealed.** None. The build succeeded first time (`make`
rc 0). The run passed first time. The same-diagonal defect itself has still not
been run against the old code; case 15's success fixture would fail
`checkKeyTables` there, by the analysis in `## M1-U2`.

**Line references refreshed.** `milestone1.md`'s Read this first #6 now states
that its Tessera references are against `33dc6b9`:

- `flipRadiusRatio`: `Tessera_EdgeFlip.hpp:305-331` → `:340-366`.
- `DefaultFlipPolicy`: `:352-371` → `:387-406`.
- The geometric test: `:725-762` → `:744-777` (evaluated in 2a; attributed in
  2c at `:840-857`).
- The duplicate-edge check: `:142-150` → `:142-178`.
- Step 3 and Decision 2: `:154-206` → `:187-239`.

The #6 duplicate-edge bullet states the same-diagonal rule as present behavior,
and the "Current state" bullet that listed the defect as wrong upstream now lists
the rule under "Present upstream, unused by Beatnik". The references inside the
M1-U2 and M1-U3 entries stay against their stated commits. M1-U3's **Do** step 4
said "Four calls" and named three places; it now says "Three places".

**Provenance.**

- Tessera commit tested: `33dc6b9143b977ec657e01c59ac804d611856b5d`
  (`mesh-quality-auto`), committed before the submit.
- `spack env status`: `==> In environment /g/g20/stewartj/spack_envs/tuolumne_trilinos`.
- Build: `spack env activate ~/spack_envs/tuolumne_trilinos/`, then, in
  `build-tuolumne`,
  `make -j tessera_test_collapse_edges_SERIAL tessera_test_collapse_edges_HIP tessera_test_flip_edges_SERIAL tessera_test_flip_edges_HIP`.
  rc 0. The only warnings are the 32 `-Wnontrivial-memcall` ones from
  `Tessera_HaloRebuild.hpp:99`.
- Submit, from the Tessera root with `TESSERA_REPO=$(pwd)` exported:
  `flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`.
  Job `f3hxYnUALpnB`, `pdebug`, COMPLETED rc 0, 176 s. `flux job status`
  returned rc 0.
- Result: `100% tests passed, 0 tests failed out of 20`. These are ctest tests
  #251–260 (`flip_edges`) and #271–280 (`collapse_edges`), SERIAL and HIP,
  np1–5. The output is `tessera-unit.f3hxYnUALpnB.out` in the Tessera root. The
  case lines come from `build-tuolumne/Testing/Temporary/LastTest.log`.

**Re-basing:** not applicable. M1-U3 touches no Beatnik pass.

**Affects:**

- **T4d3:** the quality flips may pass a mask with two edges whose quads share
  both opposite corners. `flipEdges` accepts the higher-priority geometric
  survivor and counts the other in `rejectedConflict`, so the mask need not test
  the new diagonal.
- **T4d5:** the same holds for the valence pass under `EndpointInQuad`; the open
  question in `## M1-U2`'s **Affects:** line is closed. A shut-out claimant is
  not reconsidered within the call, so the multi-round pass is what picks it up.
- **M1-T2:** a conformity failure in the cleanup flips can no longer come from a
  same-diagonal pair.

## M1-G1

**Commits:** Beatnik `26d3959` (gold data, scripts, README, `milestone1.md`); the
commit that adds this section.

**Jobs:** `f3hyB1FBHAym`, `scripts/tuolumne/m1g1_gold.flux` (`pdebug`,
`-t 40m`), log `beatnik_m1g1_gold.f3hyB1FBHAym.log` in the Beatnik repo root,
output `/p/lustre5/stewartj/beatnik/M1-G1/`. COMPLETED rc 0, 48 s job runtime;
`flux job status` returned rc 0.

**Decisions** (made before the task started; recorded here, not reopened):

- **A session generates these sets, not a person**, by running the reference
  Python through flux batch jobs on `pdebug`. The design doc's "human step"
  heading and the Conventions row now read "reference run" and
  "reference-run gold-file generation".
- **Output goes to Lustre, then is copied into the repo.** The run wrote to
  `/p/lustre5/stewartj/beatnik/M1-G1/` via `--checkpoint-dir`, not to the
  reference's `results/`, which holds the T2a 10-step set. All 61 numbered files
  plus `checkpoint_latest.npz` (62) were copied into
  `tests/regression_tests/direct-solve-600-steps/gold/` with `cp -p` and verified
  by `cmp` (0 mismatches) and `sha256sum -c` (all OK). The Lustre copy stays.
- **Walltime `-t 40m`.** Precedent from the reference's `results3/` and
  `results4/` timestamps: frozen level 3 (642 vertices) 0.20 s/step, frozen
  level 4 (2562 vertices) 2.26 s/step. Level 2 was expected under 2 minutes.
- **The command ran exactly as the task entry gives it, changing only
  `--checkpoint-dir`.** No `--A … --eps` flags and no `--icosphere-subdivisions`;
  the level-2 default is the point. Launched from the reference root as
  `flux run -N 1 -n 1 /usr/tce/bin/python3 …`, as the reference's
  `scripts/run_bubble_L3.flux` launches.
- **The reference working tree is dirty, and that is accepted.** HEAD
  `ec7d7bf4aaf9e84c6ac999dce3193ca1650677f1`; `git diff` is one line in
  `examples/run_adaptive_mesh_bubble.py:221`, the `--source-quadrature` default
  `face` → `vertex`. The command passes `--source-quadrature vertex`, so the
  change is inert. `zmodel3d/` was clean (the script refuses otherwise). Nothing
  in the reference repo was edited, stashed or committed, and the run wrote no
  file there.
- **One script per task**, running generation and the exit-criterion checks in
  the same job; walltime and queue in its `# flux:` directives. The script also
  refuses to run if its output directory already holds checkpoints, so a
  resubmit cannot mix two runs.
- **The README sits at directory level**
  (`direct-solve-600-steps/README.md`, data in `gold/`), as
  `direct-solve-10-steps/README.md` does.
- **No CMake or install change.** `tests/CMakeLists.txt` lists each gold
  directory explicitly, so this one is inert until M1-D1 / M1-T1 wire it.

**Provenance.** Beatnik `8c89323` with the two scripts untracked; dev env
(`==> In environment /g/g20/stewartj/spack_envs/tuolumne_beatnik`), no build.
`/usr/tce/bin/python3` = Python 3.13.2, numpy 2.1.2, scipy 1.15.3. Host
`tuolumne1040`.

**Result.** All in-job checks passed (`ALL CHECKS PASS`, `checks rc=0`):

- 61 numbered checkpoints, steps 0, 10, …, 600; `checkpoint_latest.npz`
  byte-identical to the step-600 file by `cmp`.
- Step 0 `np.array_equal` to `initial_conditions/gold.npz` on all nine keys;
  `compare_output.py <step0> initial_conditions/gold.npz --rtol 1e-12 --atol 1e-14`
  rc 0, `162/162 unambiguous`.
- Self-compare of `checkpoint_t00001p314132_step0000600.npz` at the same
  tolerances: rc 0, `matching (eps=1e-09): 162/162 unambiguous, ambiguous cpp=0
  gold=0`.
- `vertices (162, 3)` / `faces (320, 3)`, the nine keys, the stored `step`, and
  finite `vertices`/`potential`/`remesh_material_position` in all 61 files.
- No early stop: last progress line `step=  600`, `saved final finite
  checkpoint` logged, no `stopping at step=`. `refine_events=0 flips=0
  remesh_events=0` throughout. Final `time = 1.3141321331131375`; at step 600
  the reference reports `minQ=0.619`, `volRel=+3.75e-09`.

**Measured wall:** generation **25 s**, 0.042 s/step, for 600 steps on the
162-vertex mesh. The job's 48 s runtime includes provenance and checks.

**Only running revealed:**

- `flux run -N 1 -n 1 nproc` reports **24**: the task gets the cores `flux run`
  exposes, but the driver is a single process.
- The driver prints `skipped video rendering for
  results/adaptive-mesh-bubble-rhs.mp4` under `--no-video`; it names `results/`
  but writes nothing there.
- `checkpoint_latest.npz` is byte-identical to the last numbered file only
  because the final save writes both in the same second: `np.savez_compressed`
  stamps each zip entry with the current time. A content comparison is the robust
  check if a future `cmp` disagrees.

**Affects:**

- **M1-D1:** its gold set is in place at `direct-solve-600-steps/gold/`, 61
  numbered files at steps 0–600 every 10. The step-600 self-compare is fully
  unambiguous at `--match-eps 1e-9`, so any ambiguity M1-D1 sees comes from the
  Beatnik side or from drift, not from the mesh itself. Python takes 25 s for
  this run, so re-generating it costs nothing.

## M1-G2

**Commits:** Beatnik `26d3959` (gold data, scripts, README, `milestone1.md`); the
commit that adds this section.

**Jobs:** `f3hyB1NchZz7`, `scripts/tuolumne/m1g2_gold.flux` (`pdebug`,
`-t 60m`), log `beatnik_m1g2_gold.f3hyB1NchZz7.log` in the Beatnik repo root,
output `/p/lustre5/stewartj/beatnik/M1-G2/`. COMPLETED rc 0, 729 s job runtime;
`flux job status` returned rc 0. Submitted in parallel with M1-G1.

**Decisions:** as `## M1-G1`, with these differences:

- Output `/p/lustre5/stewartj/beatnik/M1-G2/`, copied (62 files) into
  `tests/regression_tests/milestone1-bubble/gold/` and verified by `cmp`
  (0 mismatches) and `sha256sum -c` (all OK). The Lustre copy stays.
- **Walltime `-t 60m`, the `pdebug` maximum.** The mesh grows under
  `--dynamic-remesh` with nothing bounding it but `--remesh-h-min 0.0015`, and no
  Python dynamic-remesh run had a recorded timing.
- The command is the task entry's (`--A 0.3 --g 1.0 --mu 0.002 --eps 0.025
  --viscosity-mode laplace-beltrami --br-approximation direct --adaptive-dt
  --dynamic-remesh --isotropic-cleanup --source-quadrature vertex --steps 600
  --checkpoint-every-steps 10 --no-video`), changing only `--checkpoint-dir`.
- The README (`milestone1-bubble/README.md`) carries the 61-row step / `time` /
  V / F table read from the `.npz` files, and names the job log for the full
  per-step stdout. It has no quality or volume columns, which are outside this
  task's scope.
- An early stop would have been allowed, and recorded as the compare-depth
  ceiling. None happened.

**Provenance:** as M1-G1; host `tuolumne1039`.

**Result.** All in-job checks passed (`ALL CHECKS PASS`, `checks rc=0`):

- 61 numbered checkpoints, steps 0, 10, …, 600 (`run reached step 600: no
  early stop`); `checkpoint_latest.npz` byte-identical to the step-600 file.
- Step 0 `np.array_equal` to `initial_conditions/gold.npz` on all nine keys;
  the step-0 `compare_output.py` at `--rtol 1e-12 --atol 1e-14` rc 0,
  `162/162 unambiguous`.
- Self-compare of `checkpoint_t00000p557934_step0000600.npz` at the same
  tolerances: rc 0, `matching (eps=1e-09): 538/538 unambiguous, ambiguous cpp=0
  gold=0`. **Ambiguity count: 0.**
- The nine keys, the stored `step`, valid face indices, and finite
  `vertices`/`potential`/`remesh_material_position` in all 61 files.
- No early stop: last progress line `step=  600`, `saved final finite
  checkpoint` logged, no `stopping at step=`. Final
  `time = 0.55793412422180344`.

**Measured wall:** generation **707 s**, **1.18 s/step**. This is the first
measurement of Python dynamic-remesh cost. It is 28x M1-G1's per-step cost for a
mesh 3.35x larger in faces. That is above the ~11x that direct-sum F² scaling
alone predicts, so part of the cost is per-step remesh work and the adaptive
time step. It is still well inside the 60-minute cap.

**Face-count trajectory:**

- **Start:** `320` faces / `162` vertices at step 0 (the level-2 icosphere).
- **Step 1:** the progress line already reads `V=462 F=920`, `remesh_events=390`.
- **Peak:** `1090` faces / `547` vertices at **step 10**, the first
  checkpoint after step 0.
- It then coarsens to `1080` (steps 30–50) and `1074` by step 60. From step 60
  to step 600 it stays in `1072`–`1076`.
- **Final:** `1072` faces / `538` vertices at step 600.

The cumulative `remesh_events` reads 631 at step 60 and 639 at step 600, so
nearly all mesh editing happens in the first 60 steps. `refine_events=0` and
`flips=0` throughout. `V = F/2 + 2` holds in all 61 rows. Reference `minQ` is
0.754 at step 1, 0.852–0.855 from step 60 on, and 0.852 at step 600.

**Only running revealed:**

- The mesh does **not** grow over the run. It jumps to about 3.4x the IC face
  count within the first 10 steps, then holds a near-constant ~1074 faces. The
  `--remesh-h-min` bound never comes into play by step 600 (`minEdge` 0.0208 at
  step 600).
- The run covers only `t ≈ 0.558`, against M1-G1's `t ≈ 1.314` over the same 600
  steps. Both runs use the adaptive time step (`--adaptive-dt` is the default),
  but on the refined mesh `dt` falls from `3e-3` at step 1 to about `9e-4`,
  against M1-G1's `3e-3`→`1.25e-3` on the level-2 mesh. A compare over 600 steps of this set therefore
  covers less than half the physical time of M1-G1's.
- Almost all of the divergence-relevant connectivity changes (R7/R13) happen
  before step 60. After that the counts move by at most ±2 vertices per
  checkpoint.

**Affects:**

- **M1-D1:** the mesh M1-G2 settles on is about 1074 faces, so a dynamic-remesh
  divergence measurement in Beatnik costs about 11x per direct-sum evaluation
  compared with M1-G1's 320 (F² scaling). Most connectivity divergence is
  expected in steps 0–60, where nearly all `remesh_events` occur. Any Beatnik
  topology mismatch shows up first at the step-10 checkpoint, which holds the
  1090-face peak.
- **M1-T2:** asserts against this README's table: peak `1090`/`547` at step 10,
  final `1072`/`538` at step 600, and `1072`–`1076` from step 60 to 600. The
  compare-depth ceiling is the full 600 steps. The Beatnik-side milestone run
  sizes its walltime from ~1074 faces held for 600 steps.
- **M1-A1:** the Python baseline is 707 s for 600 steps (1.18 s/step) at
  ~1074 faces. Raising `--steps` toward the 1000–5000 range Read this first #2
  discusses scales roughly linearly, at about 20 minutes per 1000 steps, unless
  the mesh starts growing later. The `pdebug` 1-hour cap holds about 3000 steps
  of this configuration. M1-A1 must also account for the run reaching only
  `t ≈ 0.56` in 600 steps.

## M1-G1 review

**Verdict:** Passed
**Reviewed:** `26d3959^..26d3959` (`0106752`, which adds the log section, touches
only `tasks/`)
**Evidence:** the exit criterion is a check on committed files, so it was re-run
directly against `tests/regression_tests/direct-solve-600-steps/gold/` at
`0106752`, with no job. The implementing session's job `f3hyB1FBHAym`
(`scripts/tuolumne/m1g1_gold.flux`, log `beatnik_m1g1_gold.f3hyB1FBHAym.log`)
was read in full: COMPLETED rc 0, 0 s queue wait, 48 s runtime, Beatnik
`8c89323` with the script untracked. The log ends `ALL CHECKS PASS`,
`checks rc=0`. The re-run gave:

- 62 tracked files, 61 numbered at steps 0, 10, …, 600. All 62 are
  byte-identical to the Lustre originals in `/p/lustre5/stewartj/beatnik/M1-G1/`,
  so the committed data is the job's output. `checkpoint_latest.npz` is
  byte-identical to the step-600 file.
- Step 0 `np.array_equal` to `initial_conditions/gold.npz` on all nine keys.
  `compare_output.py` step 0 against the IC gold at `--rtol 1e-12 --atol 1e-14`
  exits 0, `162/162 unambiguous`.
- The step-600 self-compare at the same tolerances exits 0,
  `matching (eps=1e-09): 162/162 unambiguous, ambiguous cpp=0 gold=0`.
- Every file has the nine keys, a stored `step` equal to its suffix, finite
  float fields and `V = F/2 + 2`. Every undirected edge is used by exactly two
  faces. `time` rises strictly to `1.3141321331131375`.
- Failure direction: the step-600 file compared against the step-590 file exits
  **1**, so the comparator rejects a genuinely different checkpoint and the
  rc-0 results above are not vacuous.

The mechanical pass over the range found no candidates. The two scripts are
outside **Fill in**, but they follow from it: CLAUDE.md requires a batch script
under `scripts/<system>/`, and `## M1-G1` records the one-script-per-task
decision. Both carry the BSD-3-Clause header and source the resolver before
any work. The prompt file is deleted.

1. [nit] `tests/regression_tests/direct-solve-600-steps/README.md:16-17` names
   `beatnik_m1g1_gold.f3hyB1FBHAym.log` "in the Beatnik repository root" as the
   source of the per-step stdout. `*.log` is gitignored (`.gitignore:6`), so the
   file exists only in the checkout that ran the job, and a clone has a dangling
   reference. The README itself records everything the exit criterion and
   M1-D1 need, so no fact is lost. The fix: say the log is not committed, or
   commit it beside `gold/` with `git add -f`.
2. [follow-up] Pre-existing in the design, not caused by this task: the M1-G2
   entry says `--source-quadrature vertex` is "already" the `parse_args`
   default. At reference `ec7d7bf` the committed default is `face`. Only the
   reference's uncommitted working-tree change makes it `vertex`, as this
   task's provenance capture shows. Both gold commands pass the option
   explicitly, so neither set is affected. The sentence is wrong for anyone
   who runs the reference from a clean checkout, and it should say so.

**Affects:** none

## M1-G2 review

**Verdict:** Passed
**Reviewed:** `26d3959^..26d3959` (`0106752` touches only `tasks/`)
**Evidence:** as `## M1-G1 review`, re-run directly against
`tests/regression_tests/milestone1-bubble/gold/` at `0106752`. The implementing
session's job `f3hyB1NchZz7` (`scripts/tuolumne/m1g2_gold.flux`, log
`beatnik_m1g2_gold.f3hyB1NchZz7.log`) was read in full: COMPLETED rc 0, 4 s
queue wait, 12.14 min runtime, Beatnik `8c89323` with the script untracked. The
log ends `ALL CHECKS PASS`, `checks rc=0`. Its echoed command is the task
entry's, except `--checkpoint-dir`. The re-run gave:

- 62 tracked files, 61 numbered at steps 0, 10, …, 600, so there was no early
  stop and the compare-depth ceiling is 600. All 62 are byte-identical to
  `/p/lustre5/stewartj/beatnik/M1-G2/`. `checkpoint_latest.npz` is
  byte-identical to the step-600 file.
- Step 0 `np.array_equal` to the IC gold on all nine keys. The
  `compare_output.py` step-0 check exits 0, `162/162 unambiguous`.
- The step-600 self-compare at `--rtol 1e-12 --atol 1e-14` exits 0,
  `matching (eps=1e-09): 538/538 unambiguous, ambiguous cpp=0 gold=0`.
- Every file has the nine keys, a matching stored `step`, finite float fields,
  valid face indices with no unreferenced vertex, `V = F/2 + 2`, and every edge
  shared by exactly two faces. Every remeshed checkpoint is therefore a closed
  2-manifold. `time` rises strictly to `0.55793412422180344`.
- The README's 61-row step / `time` / V / F table is identical to the table the
  job printed from the `.npz` files. Peak `547`/`1090` at step 10, final
  `538`/`1072`.
- Failure direction: step 600 compared against step 590 exits **1**.

The mechanical pass found no candidates. The script follows from **Fill in** as
M1-G1's does.

1. [nit] `tests/regression_tests/milestone1-bubble/README.md:19-22` cites the
   gitignored job log, as M1-G1 review finding 1 describes. Here it matters
   slightly more: the README quotes facts from that log's progress lines that
   no `.npz` holds — `remesh_events` 631/639, `refine_events=0`, `flips=0`,
   step 1 `V=462 F=920`. The quoted values stand, but they cannot be re-checked
   from a clone. The same fix applies.

**Affects:** none
