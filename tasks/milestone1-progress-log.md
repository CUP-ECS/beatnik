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

## M1-U1 review

**Verdict:** Passed
**Reviewed:** Tessera `279287e^..279287e` (`mesh-quality`; one commit). The
`## M1-U1` section has no `**Commits:**` line; the range is the single commit
its **Met.** paragraph and **Provenance** name, and that commit touches exactly
the six **Fill in** files (`src/Tessera_EdgeCollapse.hpp`,
`src/Tessera_EdgeFlip.hpp`, `tests/test_collapse_edges.cpp`,
`tests/test_flip_edges.cpp`, `README.md`, `docs/design.md`).
**Evidence:** implementing session's job `f3hmiTasm3Ef`, output read —
`scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`,
output `../tessera/tessera-unit.f3hmiTasm3Ef.out`. Flux: COMPLETED, rc 0,
submitted 13:40:49, started 13:41:41 (queue wait 52 s), run 3.04 min. The
output ends at `=== regression tests complete (rc=0)` with
`100% tests passed, 0 tests failed out of 20`: `flip_edges` and
`collapse_edges`, SERIAL and HIP, np1–np5, the full registration the criterion
names. Checks:
- **Provenance:** the runner echoes no commit SHA. Established instead by
  timing: `279287e` was committed at 13:40:39, ten seconds before the submit,
  and the next Tessera commit (`05b5942`) is 14:29. Corroborated by two later
  runs of the same 20 tests, `f3hn6UjNexxw` (at `05b5942`) and `f3hxYnUALpnB`
  (at `33dc6b9`), both 20/20 rc 0; between `279287e` and `33dc6b9` neither
  `quality()` body, neither call site, nor `test_collapse_edges.cpp` changed.
- **Completion:** normal completion, rc 0, output ends where a finished run ends.
- **Both directions:** the new cases assert, not just print, each direction.
  Collapse case 15 increments its failure count unless the default policy gives
  `accepted=1, rejectedQuality=0` *and* the mean-ratio policy gives
  `accepted=0, rejectedQuality=1`; flip case 13 the mirror
  (`rejectedGeometric=1` → `accepted=1`). Each also fails if the fixture stops
  separating the two metrics. The passing run therefore shows the hook live in
  both directions at every launch. Default-path bit-identity: every pre-existing
  case passes at its unchanged literals, and by inspection the default
  `quality()` calls `detail::flipRadiusRatio` on the same arguments the three
  call sites passed before.
- **Agreement:** matches the **Met.** paragraph (20/20, SERIAL and HIP,
  ranks 1–5).

Code: the three calls (`Tessera_EdgeCollapse.hpp:1005`,
`Tessera_EdgeFlip.hpp:677-678`) are in host coordinator loops, not in a Kokkos
functor, so a host-only `quality()` is correct. Comparison directions unchanged.
`flipTriNormal`/`flipRadiusRatio` moved, not duplicated; no other `src/` user.
Every doc comment and README/design passage that named the radius ratio as the
thresholded quantity now names `quality()`. The mechanical pass found no
candidates; no prompt file `tasks/milestone1_M1-U1.md` remains. The test-only
`MeanRatio*Policy` and `meanRatio` are within the task's "one new case each".

1. [nit] `tasks/milestone1-progress-log.md` `## M1-U1` — no `**Commits:**` or
   `**Jobs:**` line; the commit and job are only in prose. Accepted; the review
   records both above.
2. [follow-up] Tessera's `scripts/tuolumne/run_unit_tests.flux` does not echo
   the commit SHA it ran, so a reviewer must establish provenance by timing.
   Echoing `git rev-parse HEAD` at job start would close that; it is Tessera
   infrastructure, not this task's.

**Affects:** none

## M1-U2 review

**Verdict:** Passed
**Reviewed:** Tessera `05b5942^..05b5942` (`mesh-quality`; one commit). The
`## M1-U2` section has no `**Commits:**` line; the range is the single commit
its **Met.** paragraph and **Provenance** name. It touches
`src/Tessera_EdgeFlip.hpp`, `src/Tessera_EdgeCollapse.hpp` (the
`vertexCoordRank` removal only), `tests/test_flip_edges.cpp`, `README.md` and
`docs/design.md`, all named in **Fill in**.
**Evidence:** implementing session's job `f3hn6UjNexxw`, output read —
`scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`,
output `../tessera/tessera-unit.f3hn6UjNexxw.out`. Flux: COMPLETED, rc 0,
submitted 14:29:07, started 14:34:39 (queue wait 5.5 min), run 3.21 min. The
output runs from its header to `=== regression tests complete (rc=0)` with
`100% tests passed, 0 tests failed out of 20`: `flip_edges` and
`collapse_edges`, SERIAL and HIP, np1–np5, the registration the criterion
names. Checks:
- **Provenance:** the runner echoes no commit SHA. Established by timing:
  `05b5942` was committed at 14:29:02, five seconds before the submit, and the
  next Tessera commit (`33dc6b9`) is the following day. Corroborated by
  `f3hxYnUALpnB` at `33dc6b9`, 20/20 rc 0, which keeps case 14 and the step-3
  round.
- **Completion:** normal completion, rc 0, output ends where a finished run ends.
- **Both directions:** case 14 asserts, not just prints, each one. For every
  bipyramid pair it fails unless `SharedFace` accepts both marked edges and
  `EndpointInQuad` accepts exactly the expected key set with
  `rejectedConflict = 2 - accepted`, zero violations in the `EndpointInQuad`
  set, and a `SharedFace` violation present exactly when `EndpointInQuad`
  rejects one. So "shared opposite" shows the mode is not vertex disjointness,
  and "endpoint wins" and "opposite wins" show the failure direction through
  each branch of the endorsement rule. On the all-marked icosphere it fails on
  any violation, on a vacuous `accepted = 0`, or if the `MPI_COMM_WORLD` flip
  set or any counter differs from `MPI_COMM_SELF`'s. `SharedFace` bit-identity:
  the `else` block is unchanged in the diff, and every pre-existing case passes
  at its unchanged literals.
- **Agreement:** matches the **Met.** paragraph (20/20, SERIAL and HIP,
  ranks 1–5). The case-14 printout itself could not be re-read, because
  `LastTest.log` was overwritten by the `33dc6b9` run; the verdict rests on
  the assertions above, not on the printed counts.

Code: the `EndpointInQuad` branch is Do step 2 as written. The best offer per
vertex uses the step-3 priority with no gid, a vertex endorses `B` alone when
`B.endpoint` is set and otherwise every opposite-role offer, and a candidate
survives iff `votes == 4`. Losers go to `locConflict`. The branch runs on host,
outside any Kokkos functor. Both `allToAllV` rounds run on every rank, and the
branch depends only on `policy.conflict`, which a caller passes uniformly.
`best` holds pointers into `offers.data`, which is not modified while they are
live. `vertexCoordRank` was moved, not duplicated, and its six collapse call
sites resolve through the include. The header proof, the
`rejectedConflict` and `flipEdges` doc comments, `README.md` and
`docs/design.md` all state both relations. Mechanical pass: the S4 hits on
`vertexCoordRank` are the moved definition and its callers (a move, not a
rename), and the one S2 hit is a bulleted comment. Both dismissed. No prompt
file `tasks/milestone1_M1-U2.md` remains.

1. [nit] `tasks/milestone1-progress-log.md` `## M1-U2` has no `**Commits:**` or
   `**Jobs:**` line; the commit and job appear only in prose. Accepted; this
   review records both above.
2. [follow-up] Tessera's `scripts/tuolumne/run_unit_tests.flux` echoes no commit
   SHA (as `## M1-U1 review` noted), and the passing run leaves the case
   printouts only in `LastTest.log`, which the next run overwrites. Copying
   `LastTest.log` beside the job output would keep the measured literals
   reviewable. This is Tessera infrastructure, not this task's.

**Affects:** none

## M1-U3 review

**Verdict:** Passed
**Reviewed:** Tessera `33dc6b9^..33dc6b9` (`mesh-quality-auto`; one commit), per
the `**Commits:**` line of `## M1-U3`. It touches `src/Tessera_EdgeFlip.hpp`,
`tests/test_flip_edges.cpp`, `README.md` and `docs/design.md`, all named in
**Fill in**. Beatnik `a02631f` touches only `tasks/`.
**Evidence:** implementing session's job `f3hxYnUALpnB`, output read —
`scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`,
output `../tessera/tessera-unit.f3hxYnUALpnB.out`, case lines from
`../tessera/build-tuolumne/Testing/Temporary/LastTest.log` (written 12:41, the
job's end; not overwritten since). Flux: COMPLETED, rc 0, submitted 12:39:00,
started 12:39:03 (queue wait 3 s), run 2.93 min. Checks:
- **Provenance:** the runner echoes no commit SHA. `33dc6b9` was committed at
  12:38:56, four seconds before the submit; it is Tessera's HEAD and the only
  untracked path is `thirdparty/`, so no later change exists.
- **Completion:** normal completion, rc 0; the output runs from its header to
  `=== regression tests complete (rc=0)` with
  `100% tests passed, 0 tests failed out of 20`: `flip_edges` and
  `collapse_edges`, SERIAL and HIP, np1–np5.
- **Both directions:** case 15 asserts, not just prints, each one. The flat
  pair fails unless `accepted=1`, `rejectedConflict=1`, `rejectedGeometric=0`,
  the gathered set equals the `MPI_COMM_SELF` set and the priority's predicted
  `{(winner,(0,1))}`, and `checkKeyTables` passes. The pushed pair fails unless
  `accepted=1`, `rejectedGeometric=1`, `rejectedConflict=0` with set `{(2,3)}`,
  and the case fails if its own premise (`(5,6)` longer, threshold between the
  two radius ratios) does not hold. `LastTest.log` holds the premise line 20
  times, the pair line 40 times (both relations × 20 launches), all identical,
  and case 3's `requested=1 rejectedDuplicateEdge=1 accepted=0` 20 times.
- **Agreement:** matches the **Met.** paragraph.

Code: 2a evaluates the unchanged geometric test into `Cand::geomOk` without
erasing; 2b's coordinator picks, per non-existing target, the best `geomOk`
asker by `(len2 DESC, EdgeKey ASC)` with no gid, and answers Exists / Free /
Claimed as **Do** step 2 states; 2c attributes geometric failures before
claims, so precedence is boundary, duplicate, geometric, conflict. `claim`
holds pointers into `asked.data`, which is not modified while they are live,
and `claim.at()` is only reached for a `geomOk` asker of a non-existing target,
which is always present. Both `allToAllV` rounds stay unconditional on every
rank; no collective was added. All on host, outside any Kokkos functor. Off the
defect, replies are Exists/Free exactly where the old boolean was 1/0, so the
default path is unchanged. `DupReply::exists` has no remaining reference. The
header, `FlipResult::rejectedConflict` comment, `flipEdges` doc comment,
`docs/design.md` and `README.md` state the rule in the present tense.
Mechanical pass: the S4 hits on `exists` are the English word and the
unchanged `exists` edge set, and the one S2 hit is the case-15 header comment;
all dismissed. No prompt file `tasks/milestone1_M1-U3.md` remains.

1. [follow-up] As `## M1-U1 review` and `## M1-U2 review` noted, Tessera's
   `run_unit_tests.flux` echoes no commit SHA, so provenance again rests on
   timing. Tessera infrastructure, not this task's.

**Affects:** none

## M1-D1

**Commits:** the commit that adds this section (the script
`scripts/tuolumne/milestone1_divergence.flux`, `milestone1.md`, this log).

**Jobs:** `f3hypK87UYCo`, `scripts/tuolumne/milestone1_divergence.flux`
(`pdebug`, `-t 20m`, `--bank=unm`), log `beatnik_m1d1.f3hypK87UYCo.log` in the
Beatnik repo root. Run output is under
`/p/lustre5/stewartj/beatnik/M1-D1/scratch/sub2_<BACKEND>_np<N>_steps600/sub2_<space>_np<N>/`
and analysis artifacts under `/p/lustre5/stewartj/beatnik/M1-D1/analysis/`.
COMPLETED rc 0, 62.8 s job runtime, 40 s of launches, and `flux job status` rc 0.
It showed the **passing direction** of the exit criterion: step 0 matches at
`1e-12` in all four runs, so the measurement is valid and no regression was
found.

**Decisions** (given to the task and not reopened):

- **The driver is `Beatnik_Test_Milestone0Run_MPI_{SERIAL,HIP} 2 600 10`**,
  resolved through `beatnik_exe`, with no source change. At those positionals its
  `makeParams()` is the M1-G1 command field for field.
  `examples/02_adaptive_mesh_bubble` is HIP-only (`adaptive_mesh_bubble.cpp:212`),
  so it could not cover SERIAL.
- **Every comparison goes through `tests/regression_tests/milestone0_ladder.py`,
  unchanged**, at its five `RUNGS`. It lacked nothing the exit criterion needed.
- **The dev spack env.** The script refuses `BEATNIK_USE_PROD=1` and sources
  the resolver before any work, and the resolver printed `spack_env = … (dev)`.
- The script is a trimmed copy of `milestone0_divergence.flux`. It keeps the
  same preamble, provenance block, binding (`--gpus-per-task=1
  --cores-per-task=24 --setopt=mpibind=verbose:1`, with `/usr/bin/time -v`
  inside `flux run`), FD-3 row loop, and fresh scratch directory per run. It also
  keeps the per-row deadline guard. It adds `# flux: --bank=unm`, because the
  loop's wrapper requires an account directive.

**Departure — submission.** The job was submitted with plain
`flux batch scripts/tuolumne/milestone1_divergence.flux` from the repo root, not
through `$TASKLOOP_SUBMIT`. The wrapper
(`~/.claude/skills/run-tasks/scripts/hpc_submit.py:192`) hardcodes
`flux batch --flags=waitable`, and this system instance refuses that with
`flux-batch: ERROR: only the instance owner can submit with FLUX_JOB_WAITABLE`.
That is the same refusal `## M1-U1` records. The task loop's own attempt
(`.taskloop/milestone1/steps/005-implement-M1-D1.json`) stopped on it. The user
approved a plain `flux batch` from an interactive session, and the waiting used
`flux job status`, which works on a non-waitable job. The task was finished by a
second session from the handoff `tasks/complete_M1-D1.md`.

**Provenance.** Dev env, beatnik **`ckhvawf`**
(`beatnik@develop%cce@20.0.0`), canopy `hdvkqdz`, tessera `aikc55n`, installed
by `spack -e ~/spack_envs/tuolumne_beatnik install` with `HIPCC_*_APPEND`
cleared, from HEAD **`70fda7c`**. The install log is at
`/p/lustre5/stewartj/beatnik/m1d1_spack_install.log`. The job echoed commit
`70fda7c1158e2216c1ccebe3ee68686ee7a1b258`, with only untracked files and
`.claude/settings.json` dirty, so nothing compiled was dirty. It also echoed Cray
clang 20.0.0 and host `tuolumne1041`. No commit landed between the install and
the tabulation.

**In-job checks**, all four runs: `[PASS] Beatnik_Test_Milestone0Run (6/6
checks)` on every rank, 61 checkpoints per run, a `LAUNCH_WALL` line per run, and
the closing `launched=4 skipped=0`. The log has no `FAIL`, `STOPPED EARLY` or
`ENTITY COUNTS CHANGED`. Every progress line reads `V=162 F=320 refine_events=0
flips=0 remesh_events=0`.

### Step-0 gate — **passes, all four runs**

`compare_output.py <run>/…_step0000000.h5 direct-solve-600-steps/gold/…_step0000000.npz --rtol 1e-12 --atol 1e-14`:
rc 0, `RESULT: match` for HIP np1, HIP np4, SERIAL np1 and SERIAL np4. All four
show `162/162 unambiguous, ambiguous cpp=0 gold=0`, `faces: 320/320 triangles
identical`, `vertices` `max|e|` = `5.55111512312578270e-17` (`max|e|/|g|`
`2.916908e-15`) and `potential` `max|e|` = `0`. These are the level-2 numbers
M0-D1 step 1 recorded.

### The tolerance ladder against Python — derived / confirmed

First failing checkpointed step, from `pair --run <run> --ref
direct-solve-600-steps/gold`. Artifacts are `analysis/<tag>_vs_py.{txt,json}`.
`None` means the comparator's bounds never permitted a failure at that rung
(derived `None`), so there was nothing to confirm (0 invocations).

| backend, ranks | `1e-12`/`1e-14` | `1e-10`/`1e-12` | `1e-8`/`1e-10` | `1e-6`/`1e-8` | `1e-4`/`1e-6` |
| --- | --- | --- | --- | --- | --- |
| SERIAL, 1 | None / **None** | None / **None** | None / None | None / None | None / None |
| SERIAL, 4 | None / **None** | None / **None** | None / None | None / None | None / None |
| HIP, 1 | None / **None** | None / **None** | None / None | None / None | None / None |
| HIP, 4 | None / **None** | None / **None** | None / None | None / None | None / None |

**No (backend, rank) combination fails at any rung through step 600, not even
the tightest.** The binding constraint is `vertices` against the `1e-14` `atol`.
Its `max|e|/|g|` reaches `3.307733e+03` at step 150 because some coordinates are
near zero, so `rtol` never rescues it, and only `max|e| <= atol` keeps the rung
from being permitted. Peak `vertices` `max|e|` is `9.88098491916389321e-15`
(HIP np4, step 580), about 1.2 % under `1e-14`. The level-2 frozen-mesh `1e-12`
horizon therefore lies **just beyond step 600**, not comfortably beyond it. The
growth series below rises about `1.5e-17` per step on average, so the
`1e-14` `atol` would be crossed within a few tens of steps of 600 if the trend
held. That is an extrapolation, not a measurement, and no step past 600 was
run. For comparison, M0-D1 measured level 3 at
derived 450–475 and confirmed 1325–1350.

The `potential` `max|e|/|g|` spikes (`9.200553e-12` at step 370 and
`7.065909e-12` at step 420 for SERIAL np1) exceed `rtol 1e-12`, but `max|e|`
stays under `atol`, so those steps cannot fail either.

### The attribution — rank 1 against rank 4

`pair --run <np1> --ref <np4>` and `growth` likewise. Artifacts are
`analysis/{serial,hip}_np1_vs_np4{,_growth}.{txt,json}`. Two Beatnik `.h5` sides
share `sheet_vector`, which the gold `.npz` lacks, so the full-field horizon is
measured over a larger set than the Python one.

| backend | full set `1e-12`, derived / **confirmed** | shared subset (`vertices`, `potential`) `1e-12`, derived | other rungs | field that binds |
| --- | --- | --- | --- | --- |
| SERIAL | 530 / **None** (8 invocations) | None / None | None / None | `sheet_vector` (530) |
| HIP | 420 / **None** (19 invocations) | None / None | None / None | `sheet_vector` (420) |

- **Confirmed, full field set:** no checkpointed step fails at any rung on
  either backend. The `1e-12` derivation is permitted only by `sheet_vector`,
  and the confirming invocations at that rung found no failing step.
- **Derived, shared subset:** `vertices` and `potential` never permit a failure
  at any rung. That is the number to set beside the Python horizon, which is also
  `None` at every rung. The two horizons are therefore equal ("beyond 600"), and
  `max|e|` is what separates them.

**Cross-rank disagreement stays below the Python disagreement at every
checkpointed step** (steps 10–600, both shared fields, both backends, against
both the np1 and np4 Python comparisons). The smallest Python/cross-rank ratio at
any single step is 2.128x (SERIAL `vertices`, np4 side) and 1.500x (SERIAL
`potential`, np4 side). For HIP the minima are 4.000x and 2.167x. So Do step 3's
"first step at which two Beatnik runs at different rank counts disagree by more
than the Python comparison does" is **none through 600**. R2's cross-rank noise
does not set the horizon, and what grows is Python-versus-Beatnik drift. Peak
ratios, as M0-D1 step 4 reported them:

| backend | field | peak vs Python (np1) | peak rank1-vs-rank4 | ratio |
| --- | --- | --- | --- | --- |
| SERIAL | `vertices` | `9.07607322631065472e-15` | `4.10782519111307920e-15` | **2.21x** |
| SERIAL | `potential` | `4.38538094726936833e-15` | `1.37390099297363122e-15` | **3.19x** |
| HIP | `vertices` | `9.71445146547011973e-15` | `2.05391259555653960e-15` | **4.73x** |
| HIP | `potential` | `4.21884749357559485e-15` | `9.43689570931383059e-16` | **4.47x** |

At level 3 M0-D1 measured 10x–22x. The gap is narrower at level 2, but it is the
same sign everywhere.

Cross-rank `max|e|` at selected steps (`growth`, 17 digits):

| step | SERIAL `vertices` | SERIAL `potential` | SERIAL `sheet_vector` | HIP `vertices` | HIP `potential` | HIP `sheet_vector` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `0` | `0` | `0` | `0` | `0` | `0` |
| 10 | `1.78039804133946247e-21` | `1.73472347597680709e-18` | `1.30104260698260532e-17` | `3.33643077449107971e-21` | `8.67361737988403547e-19` | `1.21430643318376497e-17` |
| 100 | `1.38777878078144568e-16` | `4.16333634234433703e-17` | `3.05311331771918049e-16` | `8.32667268468867405e-17` | `3.46944695195361419e-17` | `4.23272528138340931e-16` |
| 200 | `2.22044604925031308e-16` | `9.71445146547011973e-17` | `6.24500451351650554e-16` | `1.66533453693773481e-16` | `8.32667268468867405e-17` | `7.49400541621980665e-16` |
| 300 | `1.11022302462515654e-15` | `1.94289029309402395e-16` | `2.47284831500493851e-15` | `5.55111512312578270e-16` | `2.49800180540660222e-16` | `2.10942374678779743e-15` |
| 400 | `1.49880108324396133e-15` | `3.88578058618804789e-16` | `4.99600361081320443e-15` | `1.22124532708767219e-15` | `4.99600361081320443e-16` | `9.21485110438879929e-15` |
| 420 | `1.55431223447521916e-15` | `3.33066907387546962e-16` | `4.55191440096314182e-15` | `1.24900090270330111e-15` | `4.44089209850062616e-16` | `1.09912079437890498e-14` |
| 500 | `2.83106871279414918e-15` | `7.70217223333702350e-16` | `8.54871728961370536e-15` | `1.33226762955018785e-15` | `5.82867087928207184e-16` | `1.12548859121375244e-14` |
| 530 | `3.27515792264421179e-15` | `9.99200722162640886e-16` | `1.07691633388640184e-14` | `1.38777878078144568e-15` | `7.49400541621980665e-16` | `1.24900090270330111e-14` |
| 600 | `4.10782519111307920e-15` | `1.37390099297363122e-15` | `1.56541446472147072e-14` | `2.05391259555653960e-15` | `7.77156117237609578e-16` | `1.34284944275364637e-14` |

### The growth series against Python (`growth`, 17 digits)

`max|e|` through the comparator's own vertex pairing. Artifacts are
`analysis/<tag>_vs_py_growth.{txt,json}`. Those files hold all 61 steps, and
every 50th is reproduced here.

`vertices`:

| step | SERIAL 1 | SERIAL 4 | HIP 1 | HIP 4 |
| --- | --- | --- | --- | --- |
| 0 | `5.55111512312578270e-17` | `5.55111512312578270e-17` | `5.55111512312578270e-17` | `5.55111512312578270e-17` |
| 10 | `3.33066907387546962e-16` | `3.33066907387546962e-16` | `3.33066907387546962e-16` | `3.33066907387546962e-16` |
| 50 | `7.77156117237609578e-16` | `7.77156117237609578e-16` | `7.77156117237609578e-16` | `7.77156117237609578e-16` |
| 100 | `1.44328993201270350e-15` | `1.44328993201270350e-15` | `1.44328993201270350e-15` | `1.44328993201270350e-15` |
| 150 | `1.94289029309402395e-15` | `1.94289029309402395e-15` | `1.94289029309402395e-15` | `1.94289029309402395e-15` |
| 200 | `2.77555756156289135e-15` | `2.77555756156289135e-15` | `2.77555756156289135e-15` | `2.77555756156289135e-15` |
| 250 | `2.77555756156289135e-15` | `2.77555756156289135e-15` | `2.77555756156289135e-15` | `2.77555756156289135e-15` |
| 300 | `3.27515792264421179e-15` | `3.27515792264421179e-15` | `3.27515792264421179e-15` | `3.27515792264421179e-15` |
| 350 | `4.21884749357559485e-15` | `4.21884749357559485e-15` | `4.21884749357559485e-15` | `4.21884749357559485e-15` |
| 400 | `4.88498130835068878e-15` | `4.88498130835068878e-15` | `4.88498130835068878e-15` | `4.88498130835068878e-15` |
| 450 | `5.93969318174458749e-15` | `5.93969318174458749e-15` | `5.93969318174458749e-15` | `5.93969318174458749e-15` |
| 500 | `7.32747196252603317e-15` | `7.32747196252603317e-15` | `7.32747196252603317e-15` | `7.32747196252603317e-15` |
| 550 | `8.60422844084496319e-15` | `8.65973959207622102e-15` | `8.65973959207622102e-15` | `8.65973959207622102e-15` |
| 600 | `9.07607322631065472e-15` | `8.74300631892310776e-15` | `8.99280649946376798e-15` | `9.49240686054508842e-15` |

Peaks: SERIAL 1 `9.07607322631065472e-15` (step 600), SERIAL 4
`9.38138455808257277e-15` (580), HIP 1 `9.71445146547011973e-15` (580), HIP 4
`9.88098491916389321e-15` (580).

`potential`:

| step | SERIAL 1 | SERIAL 4 | HIP 1 | HIP 4 |
| --- | --- | --- | --- | --- |
| 0 | `0` | `0` | `0` | `0` |
| 10 | `4.33680868994201774e-18` | `4.33680868994201774e-18` | `4.33680868994201774e-18` | `4.33680868994201774e-18` |
| 50 | `4.51028103753969845e-17` | `4.85722573273505986e-17` | `4.33680868994201774e-17` | `5.20417042793042128e-17` |
| 100 | `1.31838984174237339e-16` | `1.31838984174237339e-16` | `1.35308431126190953e-16` | `1.45716771982051796e-16` |
| 150 | `3.05311331771918049e-16` | `2.98372437868010820e-16` | `3.12250225675825277e-16` | `3.53883589099268647e-16` |
| 200 | `4.99600361081320443e-16` | `4.85722573273505986e-16` | `4.71844785465691530e-16` | `4.85722573273505986e-16` |
| 250 | `6.80011602582908381e-16` | `7.35522753814166208e-16` | `6.80011602582908381e-16` | `7.91033905045424035e-16` |
| 300 | `1.08246744900952763e-15` | `1.16573417585641437e-15` | `9.99200722162640886e-16` | `1.24900090270330111e-15` |
| 350 | `1.49880108324396133e-15` | `1.47104550762833242e-15` | `1.33226762955018785e-15` | `1.60982338570647698e-15` |
| 400 | `2.13717932240342634e-15` | `2.02615701994091069e-15` | `2.02615701994091069e-15` | `2.19269047363468417e-15` |
| 450 | `2.66453525910037570e-15` | `2.55351295663786004e-15` | `2.63677968348474678e-15` | `2.83106871279414918e-15` |
| 500 | `3.16413562018169614e-15` | `2.80331313717852026e-15` | `2.91433543964103592e-15` | `3.16413562018169614e-15` |
| 550 | `3.77475828372553224e-15` | `3.44169137633798528e-15` | `3.55271367880050093e-15` | `3.83026943495679006e-15` |
| 600 | `4.38538094726936833e-15` | `3.80251385934116115e-15` | `4.21884749357559485e-15` | `4.24660306919122377e-15` |

The step-600 value is the peak for all four runs.

**The growth is roughly linear, not exponential.** `vertices` climbs from one
ulp at step 0 to about 9e-15 at step 600, so about 1.5e-17 per step, with no
acceleration as `minQ` falls from 0.977 to 0.619. The `vertices` series is
bit-identical across all four (backend, rank) combinations at most steps. The
first disagreement is at step 160, then sporadic at 440, 490 and 520, and
continuous from 540. After that the four series spread by up to about 9 % (step
580: `9.0483e-15` to `9.8810e-15`). `time |e|` against Python peaks at
`9.10382880192628363e-15` (HIP np1). Beatnik's final times are
`1.3141321331131461` (HIP 1), `…459` (HIP 4), `…448` (SERIAL 4) and `…437`
(SERIAL 1), against Python's `1.3141321331131375`.

### Ambiguity (M1-R8) — **none**

At `--match-eps 1e-9`, **no comparison reported `ambiguous cpp>0` or `gold>0` at
any step**. That covers the 366 `pair` scan invocations (6 comparisons × 61
steps, each with `matched = n_paired = 162`, status 0, and empty
`pairing_problems`), the 366 `growth` rows (`amb` = 0 throughout) and the 4
step-0 gates. `--match-eps` was not raised. With the worst position disagreement
at about 1e-14, five decades below the cell size, the pairing has ample margin on
this mesh at 600 steps.

### Cost (M1-R6)

| backend, ranks | solve `MPI_Wtime` (s) | s/step | `LAUNCH_WALL` (s) | `/usr/bin/time` elapsed | peak RSS per rank (kB) |
| --- | --- | --- | --- | --- | --- |
| HIP, 1 | `2.56339` | `0.00427232` | 10 | 0:09.72 | 862360 |
| HIP, 4 | `3.34126` | `0.00556876` | 10 | 0:09.58–0:09.59 | 1002328 / 1002332 / 1002492 / 1003816 |
| SERIAL, 4 | `6.35231` | `0.0105872` | 13 | 0:12.15 | 706480 / 706936 / 707316 / 707368 |
| SERIAL, 1 | `3.06415` | `0.00510691` | 7 | 0:06.07 | 704104 |

**Face-count trajectory: constant 320 faces / 162 vertices at every step** of all
four runs. The driver asserts it every step, and `pair`'s count table reads
`(162, 320): 61 step(s)` for every comparison. On this path cost is flat in the
step index, so M1-R6 cannot bite here. It is a property of the dynamic-remesh
path (M1-T2). Python took 25 s for the same run (`## M1-G1`), so Beatnik is
4–10x faster at 162 vertices.

### Only running revealed

- **SERIAL np4 is twice as slow as SERIAL np1** (6.35 s against 3.06 s). At 162
  vertices the direct sum is too small to amortize the halo and all-gather, and
  `/usr/bin/time` shows about 985 000 involuntary context switches per rank at np4
  against 142 000 at np1. HIP np4 is likewise slower than HIP np1. Neither is a
  correctness concern, and both are expected to flip at larger N.
- **The SERIAL np1 `initial_volume` differs from gold by one ulp**
  (`6.32350731246694997e-02` against `6.32350731246695136e-02`, `|e|`
  `1.387779e-17`). The other three runs match the gold bit-for-bit. It is a
  reduction-order effect inside the step-0 tolerance. The driver's own echo shows
  it too (`initial_volume 0.0632350731246695`).
- **The tightest rung is decided by `vertices`' `atol`, not `rtol`.** Near-zero
  coordinates make `max|e|/|g|` reach up to `3.3e3`. Any compare depth past about
  600 at `1e-12` therefore rests on `atol 1e-14` alone, and `potential` stays at
  about half of `vertices`' `max|e|` throughout.
- The deadline guard's estimates were padded 6–17x. All four launches together
  took 40 s of the 1080 s budget, so `-t 20m` could be cut to `-t 5m` for a rerun.

**Affects:**

- **M1-A1:** the frozen-mesh level-2 horizon is **beyond step 600 at every rung**
  on both backends and both rank counts. The connectivity-frozen comparison
  supports the milestone's 600-step premise even at `1e-12`/`1e-14`, but with
  only about 1.2 % of `atol` headroom on `vertices` at step 580. Any divergence
  M1-T2 sees before step 600 at `1e-10` or looser is therefore **not**
  floating-point drift on a fixed mesh. It is edit-set divergence (M1-R2) or
  something the remesh path adds. M1-R1 did not fire. Cross-rank noise is
  2–5x below the Python drift, so a single tolerance can serve both rank counts.
  The cost of a frozen 600-step run is negligible (at most 6.4 s solve, at most
  1.0 GB per rank).
- **M1-T2:** the `vertices` `max|e|` series above is the floor its comparison
  sits on. At any step where M1-T2's `max|e|` is far above that floor with equal
  face counts, the cause is the remesh path, not R2. The pairing is unambiguous
  at `--match-eps 1e-9` throughout on the frozen mesh, so any ambiguity M1-T2
  reports comes from remeshing (M1-R8), not drift. A test asserting `1e-12` must
  use the `atol` (`1e-14`) with the measured `max|e|` beside it, because `rtol`
  never governs `vertices`.

## M1-D1 review

**Verdict:** Passed
**Reviewed:** `cff5193^..cff5193` (the only commit outside `tasks/` is
`scripts/tuolumne/milestone1_divergence.flux`, the task's **Fill in**)
**Evidence:** implementing session's job `f3hypK87UYCo`, output read —
`scripts/tuolumne/milestone1_divergence.flux`, log
`beatnik_m1d1.f3hypK87UYCo.log`, Flux `COMPLETED` rc 0, submitted
15:20:17, started 15:29:06 (queue wait about 9 min), runtime 1.047 min. The job
echoed commit `70fda7c`; the one commit after it, `cff5193`, adds only the batch
script and `tasks/` files, so the installed binary it ran is the reviewed code.
The log reads `[PASS] Beatnik_Test_Milestone0Run (6/6 checks)` on every rank of
all four launches, `checkpoints written: 61` each, `launched=4 skipped=0`, and
no `FAIL`, `STOPPED EARLY` or `ENTITY COUNTS CHANGED`. Re-checked on the login
node with `compare_output.py` against the run's lustre output (read-only):
HIP np4 step 0 at `--rtol 1e-12 --atol 1e-14` gives `RESULT: match`, `vertices`
`max|e|` `5.551115e-17`, `potential` `0`, `162/162 unambiguous` — the passing
direction of the step-0 gate. HIP np4 step 580 at the same tolerances matches
with `vertices` `max|e|` `9.880985e-15`, the logged peak. The failure direction:
the same step-580 pair at `--rtol 1e-16 --atol 1e-16` exits 1 with
`vertices: 449 of 486 entries outside` and `potential: 157 of 162`, so the
comparator does detect the drift the ladder reports as within tolerance. The
`ladder` arrays of all six `analysis/*_vs_{py,np4}.json` files agree with the
log's tables (all `null` against Python; cross-rank `1e-12` derived 420 HIP /
530 SERIAL, confirmed `null`, 19 / 8 invocations), with `structural_ok` true
and empty `pairing_problems` throughout.

Mechanical pass (`review_checks.py`): no candidates. The script follows
`milestone0_divergence.flux`'s preamble, provenance block, binding, FD-3 loop
and per-run scratch on lustre, carries the BSD-3-Clause header, refuses
`BEATNIK_USE_PROD=1`, and resolves the driver through `beatnik_exe`. The
submission departure (plain `flux batch`, user-approved) is recorded in
`## M1-D1`. No prompt file remains under `tasks/`.

1. [follow-up] `tasks/milestone1-progress-log.md` `## M1-D1`, "The tolerance
   ladder against Python" — the reading that the level-2 `1e-12` horizon lies
   "just beyond step 600" and that the `1e-14` `atol` "would be crossed within a
   few tens of steps" describes when the ladder's *derived* bound first permits
   a failure, not when a comparison fails. The comparator's test is elementwise
   `|e| <= atol + rtol·|g|`, and step 580 HIP np4 still matches at
   `--rtol 1e-12 --atol 1e-15`, an `atol` an order below the peak `max|e|`. So
   "`rtol` never governs `vertices`" holds for the derived bound only; the
   confirmed horizon is likely well past the derived one, as M0-D1 found at
   level 3 (derived 450–475, confirmed 1325–1350). M1-A1 should read the
   headroom in the derived sense when it sets the ladder. No change to M1-D1's
   measurements.

**Affects:** none

## T4d1

**Commits:** `14764a5` (the adapters, `MeshEditReport`, the test, its CMake
registration and `scripts/tuolumne/t4d1_mesh_edits.flux`), and the commit that
adds this section (`milestone1.md`, this log).

**Jobs:** `f3iArVXa12CF`, `scripts/tuolumne/t4d1_mesh_edits.flux` (`pdebug`,
two nodes, `-t 15m`, `--bank=unm`), log
`beatnik_t4d1_mesh_edits.f3iArVXa12CF.log` in the Beatnik repo root. COMPLETED
rc 0, 68 s, `flux job status` rc 0. It echoed commit `14764a5` and the dev-env
binary `.spack-env/view/share/Beatnik/tests/Beatnik_Test_MeshEdits` (mtime
14:42:42, from this task's `spack install`). It showed **both directions** of
the criterion: the passing one (a)–(c), and the failure one (d), at every rank
count: `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank counts)`, and all 21
ranks printed `[PASS] Beatnik_Test_MeshEdits (86/86 checks)` on HIP.

**Decision** (given to the task and not reopened): the rank sweep runs through
the new `scripts/tuolumne/t4d1_mesh_edits.flux`, not
`scripts/tuolumne/unit_tests.flux`, which allocates one node while each launch
asks for `ceil(n/4)` and carries no `--bank`. The new script copies
`t4_fmm_vs_direct.flux`'s shape and binding, adds `--bank=unm`, defaults to
ranks `1 2 3 4 5 6` in one job, echoes `spack env status`, the compiler, the
commit and `git status --short`, the binary and each launch command, and exits
non-zero if any rank count fails. `unit_tests.flux` is untouched.

**Final signatures** (`src/Beatnik_MeshInterface.hpp`):

```
template <class CollapsePolicy = collapse_policy_type>
MeshEditReport collapseEdges( const std::vector<char>& edgeMask,
                              const CollapsePolicy& policy = CollapsePolicy{} );
template <class FlipPolicy = flip_policy_type>
MeshEditReport flipEdges( const std::vector<char>& edgeMask,
                          const FlipPolicy& policy = FlipPolicy{} );
MeshEditReport compact();
```

`SurfaceMesh` gains three aliases, `collapse_policy_type`
(`Tessera::DefaultCollapsePolicy`), `flip_policy_type`
(`Tessera::DefaultFlipPolicy`) and `flip_conflict_type`
(`Tessera::FlipConflict`), so a caller can name and derive from the policies
without spelling a Tessera type, per the file's adapter contract. The policy is
forwarded verbatim. `MeshEditReport` gains `requested`, `accepted`,
`rejected_{boundary, link_condition, normal_flip, quality, duplicate_edge,
geometric, conflict}`, `{vertices,edges,faces}_removed` and
`gid_space_{before,after}`, all Tessera's global counts, not re-reduced; the
split fields are unchanged. Its doc comment lists which edit fills which field.
`compact()` wraps `Tessera::compact` only; `compactAndRenumberGids` stays
unwrapped for T4d6.

**Halo-exchange placement: inside the adapter.** `collapseEdges` calls
`haloExchange()` before `Tessera::collapseEdges`. Tessera's header states the
precondition (vertex positions and user fields halo-consistent on entry,
`Tessera_EdgeCollapse.hpp` "ENTRY PRECONDITION ON VERTEX DATA" and the
`collapseEdges` doc comment) but does not make it the adapter's caller's
obligation in a way the adapter cannot meet, so the adapter meets it itself:
one collective, and the call is correct after any position update. The doc
comment says so and contrasts it with `splitEdges()`. `flipEdges` does not
exchange: it reads the four quad corners' positions and changes no vertex
data; its comment states that positions must be halo-consistent on entry, as
for every kernel that reads ghosts.

**The V/E/F check** throws `std::runtime_error` naming accepted and the
removed counts. The counters are global, so every rank throws or none does;
the code comment says so. It never fired.

**Test fixtures** (`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`). Gids are
not on the adapter's surface, so edges are named geometrically: the owned edge
whose midpoint direction has the largest dot product with a fixed generic
vector, chosen by one global `MPI_MAX`. Each selection asserts exactly one edge
was marked globally. Case 1 uses six collapse targets near ±x, ±y, ±z and eight
flip targets near the cube diagonals. **The link-condition fixture (d)** is
built by editing, as the prompt suggested: take the valence-5 vertex `v`
furthest along `(0.21, 0.37, 0.9)`, flip twice the spoke whose far end is
furthest along `(0.53, -0.29, 0.11)` (one per call, each `accepted=1`), so `v`
reaches valence 3 with ring `{x, y, z}`; mark the ring edge closest to
`(-0.17, 0.44, 0.61)`. The test gathers the one-rings and asserts the premise:
`valence(v)=3` and `|link(x) ∩ link(y)| = 3` (v, z and the far opposite
corner). The log prints `valence(x)=6 valence(y)=7` on 15 ranks and `7`/`6` on
6. That is only the order in which `v`'s gathered ring lists `x` and `y`, which
depends on the partition; the edge and the result are the same.

**Measured counts** (identical at every rank of np 1–6):

| Step | requested | accepted | rejected | removed V/E/F | mesh after |
| --- | --- | --- | --- | --- | --- |
| start | | | | | V 162 E 480 F 320, χ 2 |
| collapse 6, `minQuality = 1.1` | 6 | 0 | `rejectedQuality=6` | 0/0/0 | unchanged |
| collapse 6, permissive (`maxNormalRotation = π`, `minQuality = 0`) | 6 | 6 | 0 | 6/18/12 | V 156 E 462 F 308, χ 2 |
| flip 8, permissive (`maxNormalDeviation = π`, `minQuality = 0`) | 8 | 8 | 0 | — | unchanged V/E/F, face gids preserved |
| `compact()` | | | | 0/0/0 | gid space 962 → 962 (live V+E+F 926), face gids preserved |
| case 2: bad edge alone, default policy | 1 | 0 | `rejectedLinkCondition=1` | 0/0/0 | V 162 E 480 F 320, face gids unchanged |
| case 2: bad + antipodal, permissive | 2 | 1 | `rejectedLinkCondition=1` | 1/3/2 | V 161 E 477 F 318, χ 2 |

The gid-space premise in (c) — 962 against 926 live entities — is what makes a
renumbering `compact()` observable; the test asserts it.

**Nothing only running revealed.** The build and the job passed first time.

**Session notes.**
- The first `spack install` was killed with rc 137 by the session's ten-minute
  command timeout; the rerun, in the background, finished in 11m19s, rc 0. The
  full `spack install` rebuilds every test and example; the user confirmed that
  is the build to keep. `spack env activate` is blocked by the loop's guard, so
  the install ran as `spack -e ~/spack_envs/tuolumne_beatnik install` after
  unsetting `HIPCC_*_FLAGS_APPEND`.
- The build log carries 70 warnings, all `-Wpass-failed` occupancy notes from
  Canopy's FMM kernels; none is in a file this task touched.
- After a resume, `$TASKLOOP_SUBMIT` was unset in the shell. The job was
  submitted through the same wrapper (`run-tasks/scripts/hpc_submit.py`), with
  `TASKLOOP_POLICY`, `TASKLOOP_DIR` and `TASKLOOP_STEP` exported to this loop's
  values. It accepted the script, rc 0.
- Cosmetic: `splitEdges()`'s brief claimed it was "the only topological edit
  Beatnik performs"; it now says it is the only one the solver calls until
  T4d2–T4d5 wire the others in. The file header and the topological-edits
  banner no longer say the three adapters are unwritten.
- `README.md` documents none of these signatures; no change.

**Re-basing:** the adapters do not touch the AMR reference state
(`/faces/u0..u2`); collapse, flip and compact carry face user fields verbatim.
Whether to call `AdaptiveMesh::resetReferenceState` after an edit is the
caller's decision (T4d2, T4d3, T4d5).

**Affects:** T4d2 (derive the collapse policy from
`mesh_type::collapse_policy_type`; `collapseEdges` already exchanges the halo,
so its caller need not), T4d3 and T4d5 (pass `flip_policy_type` with
`conflict = flip_conflict_type::EndpointInQuad` for the valence pass; exchange
positions before flipping if they moved), T4d6 (`compact()` reports the gid
space; `compactAndRenumberGids` still needs an adapter).

## T4d1 review

**Verdict:** Passed
**Reviewed:** `14764a5^..14764a5` (the one code commit; `83e034e` after it
touches only `tasks/`). It touches exactly the **Fill in**:
`src/Beatnik_MeshInterface.hpp`, `tests/unit_tests/Beatnik_Test_MeshEdits.cpp`,
its `BEATNIK_UNIT_TEST_SOURCES` entry, and `scripts/tuolumne/t4d1_mesh_edits.flux`.
**Evidence:** implementing session's job `f3iArVXa12CF`, output read.
`scripts/tuolumne/t4d1_mesh_edits.flux`, log
`beatnik_t4d1_mesh_edits.f3iArVXa12CF.log`, Flux `COMPLETED` rc 0. It was
submitted and started at 14:44:28 (no queue wait) and ran for 1.134 min. The job
echoed commit `14764a5`. The binary's mtime predates the commit by 91 s, but the
installed `Beatnik_MeshInterface.hpp` in the dev view is byte-identical to the
file at `14764a5`, and no later commit touches `src/`, `tests/` or `scripts/`.
So the binary is the reviewed code. The log shows `PASS np=1` … `PASS np=6` and
`SUMMARY: PASS (6/6 rank counts)`. It has exactly 21 `[PASS]
Beatnik_Test_MeshEdits (86/86 checks)` lines (1+2+…+6 ranks), all on HIP, and no
`FAIL`, abort or exception. The passing direction: the six permissive collapses
remove `6/18/12` (V 156 E 462 F 308, χ 2), the eight flips are accepted with
V/E/F and face gids unchanged, and `compact()` removes `0/0/0` with gid space
962/962 against 926 live entities. The failure direction (d): the
link-condition edge alone gives `accepted=0 rejectedLinkCondition=1` and the
mesh is unchanged. Marked beside the antipodal edge, it gives `accepted=1
rejectedLinkCondition=1` and removes `1/3/2`. The strict-policy collapse
(`rejectedQuality=6`) shows that the policy is forwarded. Every count is
identical at every rank count, and every one agrees with the **Met.** paragraph.

Mechanical pass (`review_checks.py --old-names EdgeListView`): four K4 and three
M3 candidates, all in the test, all dismissed. The K4 hits are one-off host
snapshots, not a per-step loop. The device copy is fenced before the mirror. The
M3 buffers are host `std::vector`s, not Views. `EdgeListView` is gone from the
tree. Every Tessera counter is mapped to its `MeshEditReport` field: all seven
`CollapseResult` counters plus the three `*Removed`, all six `FlipResult`
counters, and all five `CompactStats` fields. Tessera's `flipEdges` bumps the
generation (`setOwnedCounts`, then `rebuildHalo`), so the adapter's
generation-keyed caches cannot go stale across a flip. The identity check throws
on global counters, so it is rank-uniform, and the test's one early return
(`ring.size() != 3`) is on gathered data, so it is rank-uniform too. The header
is BSD-3-Clause, and the script resolves the binary through `beatnik_exe` after
sourcing the resolver. No prompt file remains under `tasks/`. The log section
records the departures from **Do** (halo exchange inside `collapseEdges`, the
three policy aliases, the new runner).

1. [nit] `src/Beatnik_MeshInterface.hpp`, `haloExchange()`'s `@pre` (about
   `:1384`): it still argues that the halo plans are always live because
   "`splitEdges()` calls `rebuildHalo()` itself". The claim holds, since
   `collapseEdges`, `flipEdges` and `compact` also end in a halo rebuild, but
   the comment names only the split. It can be folded into T4d2's touch of this
   file. No fix task is needed.

**Affects:** none

## T4d2

**Commits:** `848e009` (`collapseShortEdges`, `ReferenceCollapsePolicy`, the
generalized cap search, the pass-loop wiring, the `RemeshDiagnostics` fields,
case 3 of `Beatnik_Test_MeshEdits`, the `haloExchange()` `@pre` nit, README and
`docs/testing.md`), and the commit that adds this section (`milestone1.md`,
this log).

**Jobs:**
- `f3iBGQsreZPm`, the exit criterion: `scripts/tuolumne/t4d1_mesh_edits.flux`
  at its default ranks 1 2 3 4 5 6 (`pdebug`, two nodes, `-t 15m`,
  `--bank=unm`). Log `beatnik_t4d1_mesh_edits.f3iBGQsreZPm.log` in the repo
  root. COMPLETED rc 0, 66 s, `flux job status` rc 0. It echoed commit
  `848e009` with only unrelated files dirty, and the dev-view binary (mtime
  15:20:08, from this task's second `spack install`). It shows both directions:
  the pass removes faces, and factor 0 is a no-op. `PASS np=1` … `PASS np=6`,
  and all 21 ranks printed `[PASS] Beatnik_Test_MeshEdits (122/122 checks)` on
  HIP.
- `f3iB782W4h4B`, a measurement run of the same script before the three count
  literals were known. The test held `-1` sentinels, so it FAILED by design on
  exactly the 7 checks that read them (115/122 passed). It printed the counts
  below, identical on all 21 ranks. Log
  `beatnik_t4d1_mesh_edits.f3iB782W4h4B.log`, kept.

**Decisions given to the task, recorded as given:**
- **The `Solver` collapse rejection stays**
  (`src/Beatnik_Solver.hpp:797-817`), and so does the T4b `checkRejected`
  case. T4d6 deletes both. A `Solver` run therefore reaches
  `collapseShortEdges` only at `collapse_factor <= 0`, where it returns 0 as its
  first statement: no `countMask`, no `collapseEdges`, no halo exchange, no
  compaction. The pass loop computes the second sizing field unconditionally,
  into a fresh local View. That costs the T4b gate path one extra
  `vertexTargetEdgeLength` (its gradation halo exchanges and reductions) per
  remesh, and mutates nothing. The rejection's message still says "not yet
  implemented"; it is deleted whole at T4d6, so it was left untouched.
- **The cap mirrors the split pass's.** `searchRatioThreshold` takes a
  `RatioKeep { Largest, Smallest }`. `Largest` (split) keeps `ratio > t` and
  returns the smallest fitting `t`, bit-identical to the old code path.
  `Smallest` (collapse) keeps `ratio < t` and returns the largest fitting `t`,
  with bracket `[0, max·(1+1e-12)+1e-300]` and the same fixed 60 probes. A new
  helper, `applyRatioThreshold`, clears the mask in either direction and
  replaces the split pass's inline kernel. Ties at the threshold are all
  dropped, so a capped pass may request fewer than the cap. On the perturbed
  test mesh it requested exactly 25 of 25.
- **One `collapseEdges` call per pass**, no `while (accepted > 0)`: the
  reference walks its list once.
- **Policy values:** `ReferenceCollapsePolicy` derives from
  `mesh_type::collapse_policy_type` and shadows `quality()` with `4√3·A/Σℓ²` on
  host 3-vectors. It returns 0 for a zero edge-length sum or a non-finite value,
  matching the reference's `np.isfinite` rejection. `t = 0.5`,
  `minQuality = _params.min_quality` and `maxNormalRotation = std::acos( 0.1 )`
  are each set explicitly. No `Tessera::detail::` name is used.

**Decisions made in the session:**
- `collapseShortEdges` gains a trailing `RemeshDiagnostics&`, as
  `splitLongEdges` has. It had no caller before this task.
- New `RemeshDiagnostics` fields, accumulated over passes as the split ones
  are: `collapse_candidates` (before the cap), `collapse_requested` (Tessera's
  `requested`), `collapse_rejected_conflict`, `collapse_capped` and
  `collapse_ratio_threshold`. With `collapses` (Tessera's `accepted`) and the
  cap parameter, they give M1-R3's requested, accepted and cap per pass;
  `passes` is 1 by default.
- No halo exchange after the collapse. Tessera's `collapseEdges` ends in
  `compact()` and returns a fully haloed mesh (`Tessera_EdgeCollapse.hpp`,
  doc comment near `:1490`), and the adapter exchanges before.
- **Test fixture:** each sub-case uses a fresh subdivision-2 icosphere whose
  owned vertices are scaled radially by `1 + 0.02·sin(3x+5y+7z)`, then
  halo-exchanged. This breaks the icosphere's length ties so that the cap's
  answer is unique. The target is `h_max = 0.05` everywhere on this sphere, so
  `collapse_factor = 6` makes an edge a candidate iff it is shorter than 0.3.

**Re-basing:** `collapseShortEdges` does **not** call
`AdaptiveMesh::resetReferenceState`. The `Solver` remesh branch already calls it
after every remesh that ran (`src/Beatnik_Solver.hpp:521-541`), followed by
`centerPotential`. That is the reference's own place for it:
`dynamic_remesh_state_with_material` rebuilds the state after the whole cycle,
not after each pass. A per-pass re-base would also drag the AMR header into
`DynamicRemesh`.

**Measured counts** (integers; no real-valued quantity is asserted), identical
on all 21 ranks at np 1–6:

| Sub-case | candidates | requested | accepted | rejectedConflict | mesh after |
| --- | --- | --- | --- | --- | --- |
| 3a, factor 0 | 0 | 0 | 0 | 0 | V 162 E 480 F 320; gids, positions, generation unchanged |
| 3b, factor 6, cap 0 (unlimited) | 240 | 240 | 4 | 236 | V 158 E 468 F 312, χ 2 |
| 3c, factor 6, cap 25 | 240 | 25 | 4 | 21 | V 158 E 468 F 312, χ 2, capped |
| 3d, `remesh()`, split off | — | — | 4 | — | F 320 → 312, 0 splits |
| 3d, `remesh()`, factor 0 | 0 | — | 0 | — | F 320, face gids unchanged |

**Only running revealed:** **one `collapseEdges` call accepts very few
candidates when they are dense.** Of 240 candidates, all of which passed the
link, fold and quality tests, only 4 were accepted. The other 236 lost
Tessera's one-round independent set, which accepts an edge only if it is the
shortest candidate in both endpoints' one-rings. On a smoothly varying mesh
there are few such local minima. The reference's serial walk would accept every
candidate whose endpoints no earlier collapse removed, which here is plausibly
tens. With the per-pass single call decided above, Beatnik's collapse counts
will therefore lag the reference far more than M1-R3's "diverges when the cap
binds" anticipates. This is not a fault in the pass. It is information for
M1-A1 and for the T4d exit criterion's statistics comparison.

**Session notes:**
- Two `spack install`s, both rc 0: 12m18s, then 2m14s incremental. Each ran
  after unsetting `HIPCC_*_FLAGS_APPEND`. The build carries the same 70
  warnings, all Canopy `-Wpass-failed`; none is in a file this task touched.
- The first resubmit was refused (exit 4) by the loop wrapper. It named the two
  finished jobs `f3iArVXa12CF` and `f3iB782W4h4B` as active. On Flux 0.90.0,
  `flux jobs --filter=active <ids>` warns `Filtering options ignored with
  jobid list` and lists inactive jobs too, and `taskloop_common.py::job_states`
  counts them. After the loop restarted, the submit was accepted.
- Cosmetic corrections made: `DynamicRemesh.hpp`'s header table, the pass-loop
  comment and the collapse doc comment now state the code as it is; the
  collapse citation is `:361-405`. README Known Issues and `docs/testing.md:140`
  say collapse is implemented and still rejected at setup until T4d6.
  `haloExchange()`'s `@pre` names every edit that rebuilds the halo (the T4d1
  review nit). `milestone1.md`'s Current-state line references were refreshed.

**Affects:**
- **T4d6:** it deletes the `Solver` collapse rejection and the T4b
  `checkRejected` collapse case, as planned. After that, the T4b member's
  `collapse_factor` (0) still keeps its trajectory unchanged. Gid space: each
  accepted collapse leaves holes, and T4d1 measured 962 against 926 live
  entities after six collapses. T4d2 adds no renumbering, so the leak is live
  the moment the rejection goes.
- **M1-A1 / the T4d exit criterion:** see "Only running revealed". Compare
  collapse statistics knowing one call per pass accepts a small independent
  subset of the candidates.
- **T4d3, T4d5:** `RatioKeep` / `applyRatioThreshold` exist if a flip mask
  ever needs a cap.

## T4d2 review

**Verdict:** Passed
**Reviewed:** `848e009^..848e009` (the one code commit; `fb9d75f` after it
touches only `tasks/`). It touches `src/Beatnik_DynamicRemesh.hpp` (the
**Fill in**), the test `tests/unit_tests/Beatnik_Test_MeshEdits.cpp`,
`haloExchange()`'s `@pre` in `src/Beatnik_MeshInterface.hpp` (the T4d1 review
nit, folded in as that review proposed), and the README Known Issues and
`docs/testing.md` lines that CLAUDE.md requires to stay in sync.
**Evidence:** implementing session's job `f3iBGQsreZPm`, output read.
`scripts/tuolumne/t4d1_mesh_edits.flux`, log
`beatnik_t4d1_mesh_edits.f3iBGQsreZPm.log`, Flux `COMPLETED` rc 0. It was
submitted at 15:36:56, started at 15:36:57 (1 s queue wait) and ran for
1.097 min. The job echoed commit `848e009` with no source file dirty, and the
dev-view binary's mtime (15:20:08) is after the commit (15:18:08). The log has
`PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank counts)`, and exactly 21
`[PASS] Beatnik_Test_MeshEdits (122/122 checks)` lines, all on HIP, with no
`FAIL`, abort or exception. Each case-3 note appears 21 times with identical
values, so the counts are the same at ranks 1 and 4 (and at 2, 3, 5 and 6). The
passing direction: 3b, at factor 6 with `max_collapses_per_pass = 0`, finds 240
candidates, requests 240 and accepts 4 (236 `rejectedConflict`), leaving
V 158 E 468 F 312 with χ 2. So the pass runs, and the zero cap means unlimited.
3c, at cap 25, requests 25, reports capped, and accepts 4. 3d's `remesh()`
collapses 4 with 0 splits, so F goes 320 → 312. The failure direction: 3a, at
factor 0, reports 0 candidates and 0 collapses, and asserts that counts, face
gids, every local position and the generation are unchanged. 3d at factor 0
leaves 320 faces and the gids unchanged. Every number agrees with the **Met.**
paragraph.

Mechanical pass (`review_checks.py`): no candidates. Read against the reference
(`dynamic_remesh.py:158-165, 361-405, 509-549`):
- The candidate predicate, the 1e-300-floored ratio key and the min-of-endpoints
  target all match.
- `max_collapses_per_pass <= 0` is unlimited, as the driver maps it.
- The `Smallest` bisection brackets `[0, hi]`. `fits(0)` is true because the
  ratios are non-negative, and `fits(hi)` is false whenever the cap binds. It
  returns `lo`, and `applyRatioThreshold` keeps `ratio < lo`. The `Largest` path
  is unchanged.
- `ReferenceCollapsePolicy::quality` is the formula of `triangleQuality`.
  Tessera takes the policy as a template parameter
  (`Tessera_EdgeCollapse.hpp:1501`) and calls `policy.quality` at `:996`, so the
  shadowing member is the one called. No virtual dispatch is needed.
- `minQuality`, `maxNormalRotation = acos(0.1)` and `t = 0.5` are set
  explicitly.
- Every branch before `collapseEdges` is on a parameter or on a global count, so
  every rank reaches the same collectives.

The old `const int collapses = 0` placeholder and the `BEATNIK_NOT_IMPLEMENTED`
call are gone. `searchRatioThreshold` has no caller left on the old signature,
and the split pass's inline cap kernel is replaced by `applyRatioThreshold`.
Nothing is made dead. No prompt file remains under `tasks/`. The Solver
rejection text that still says "not yet implemented" is the recorded decision
(it is deleted whole at T4d6) and is not a finding.

1. [follow-up] `tests/unit_tests/Beatnik_Test_MeshEdits.cpp` case 3: no
   sub-case shows `ReferenceCollapsePolicy` *rejecting* a collapse.
   `rejectedQuality` and `rejectedNormalFlip` are 0 in every case-3 run, and all
   236 losses are `rejectedConflict`. The hook's dispatch is confirmed by
   reading, and T4d1's case 1 shows that a forwarded policy's `minQuality` is
   honored. But the reference metric itself, as against Tessera's radius ratio,
   is never discriminated by a test. The exit criterion does not ask for this.
   A candidate check for T4d6 or M1-T1 is a fixture where the two metrics
   disagree.

**Affects:** none

## T4d3

**Commits:** `5fa0488` (the two entry points, `qualityFlipSweep` and its
helpers, the flip map on `MeshEditReport`, `SurfaceMesh::vertexGids()`, the
pass-loop and refine-branch wiring, the new `RemeshDiagnostics` fields, case 4
of `Beatnik_Test_MeshEdits`, README and `docs/testing.md`), and the commit that
adds this section (`milestone1.md`, this log).

**Jobs:**
- `f3iL4TUfQCCX`, the exit criterion, **both directions**:
  `scripts/tuolumne/t4d1_mesh_edits.flux` at its default ranks 1 2 3 4 5 6
  (`pdebug`, two nodes, `-t 15m`, `--bank=unm`). Log
  `beatnik_t4d1_mesh_edits.f3iL4TUfQCCX.log` in the repo root. COMPLETED rc 0,
  63 s, `flux job status` rc 0. It echoed commit `5fa0488` with only unrelated
  files dirty (`.claude/settings.json`, `tasks/milestone1.md`, untracked
  files), and the dev-view binary, mtime 10:15:14. That is 23 s before the
  commit (10:15:37); the build ran on the exact tree that was then committed,
  with no edit in between. `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank
  counts)`, and all 21 ranks printed `[PASS] Beatnik_Test_MeshEdits (177/177
  checks)` on HIP. The rank-to-GPU binding is the script's, unchanged.
- `f3iL2hYCLPBV`, a measurement run of the same script before the count
  literals were known, on the uncommitted tree. The test held `-1` sentinels,
  so it FAILED by design on exactly the 8 checks that read them (169/177
  passed) on all 21 ranks, and printed the counts below. Log
  `beatnik_t4d1_mesh_edits.f3iL2hYCLPBV.log`, kept.

**Decisions given to the task, recorded as given:**
- **The Solver rejections stay**, message text included: `--flip-passes > 0`
  under `--refine-every` (now `src/Beatnik_Solver.hpp:764-769`, shifted by this
  task's comments and constant) and `--remesh-flip-min-gain < 1e12` (now
  `:848-856`). The regression cases asserting them are untouched. Both passes
  are wired anyway: `diag.flips += flipEdgesForQuality( mesh, diag )` in the
  repair gate, and the refine branch's `flips` from
  `improveConnectivityByFlips( _mesh, _params.filter.flip_passes,
  kRefineFlipMinGain )`.
- **Pass structure.** One sweep is several `flipEdges` rounds sharing a
  sweep-wide touched set; `flipEdgesForQuality` runs one sweep and
  `improveConnectivityByFlips` up to `max_passes` with early stop after a sweep
  that accepts nothing. Rounds never restart with a fresh mask.
- **Predicate, written once.** `QualityFlips::accepts` with a
  `QualityFlipCriterion { has_floor, floor, min_gain }`. The remesh path passes
  `floor = min_quality`, `gain = flip_min_gain`; the refine path no floor and
  `Solver::kRefineFlipMinGain = 1e-3`, citing `mesh_solver.py:1708`.
- **Policy.** `mesh_type::flip_policy_type` with `maxNormalDeviation =
  std::acos( -1.0 )` (π), `minQuality = 0` and `conflict = SharedFace`, each
  set explicitly. No `Tessera::detail::` name is used.

**Decisions made in the session:**
- **Where the machinery lives.** `FlipSweepStats`, `QualityFlipCriterion`,
  namespace `QualityFlips` (`triangleQuality`, `unitNormal`, `accepts`,
  `gidPair`, `gatherNewDiagonals`) and the free function template
  `qualityFlipSweep` are in `src/Beatnik_MeshQuality.hpp`, above the class.
  `DynamicRemesh.hpp` now includes `MeshQuality.hpp`, which depends on neither
  `DynamicRemesh` nor `AdaptiveMesh`, so there is no cycle.
  `ReferenceCollapsePolicy::quality` now calls `QualityFlips::triangleQuality`
  (same formula, same 0 for a non-finite value), so one host copy of the metric
  exists.
- **The touched set is keyed by new-diagonal vertex-gid pairs**, the second
  option the prompt offered. A face is touched iff one of its edges is a
  diagonal created in the sweep, which holds because a flip rewrites exactly the
  two faces of its new diagonal and no touched face is rewritten again in the
  sweep. This needs vertex gids, so `SurfaceMesh` gains `vertexGids()` (host,
  every local vertex) beside `ownedFaceGids()`.
- **The flip map crosses the adapter.** `MeshEditReport` gains
  `std::vector<FlippedEdge> flipped` (rank-local; old and new endpoints as
  vertex-gid pairs), filled by `flipEdges` from `FlipResult::flipped`, so no
  caller spells a Tessera type. `gatherNewDiagonals` does one `MPI_Allgather`
  plus one `MPI_Allgatherv`, deduplicates by old key, and throws if the distinct
  count is not `accepted` (never fired).
- **Residency checks.** Each round calls `faceAdjacency()` unconditionally and
  also counts owned edges whose two resident faces or any corner is missing;
  both counts are reduced and either nonzero throws, on every rank together.
  Never fired. The quad itself comes from `edgeAdjacency().resident_faces`
  (the two faces of the edge), which is what the reference reads.
- **Empty-mask exit.** A round whose mask is empty on every rank (one
  `MPI_Allreduce`) breaks before `flipEdges`, so 1e12 makes no edit call; it
  still costs one `faceAdjacency()` build and one reduction per call.
  `rounds` counts only `flipEdges` calls.
- **New diagnostics** in `RemeshDiagnostics`, accumulated like the collapse
  ones: `flip_sweeps`, `flip_rounds`, `flip_requested` and
  `flip_rejected_{boundary, duplicate_edge, geometric, conflict}`.
  `improveConnectivityByFlips` takes an optional trailing
  `std::vector<FlipSweepStats>*` for the same per-sweep counts; the Solver
  passes none.
- **The normal-sum test** uses the remesh path's `‖n‖ ≤ 1e-300` for both entry
  points; the refine path's reference tests `‖n‖ == 0` or non-finite. A sum of
  two unit vectors is zero or far above 1e-300, so they agree on finite input.
- **Fixture chosen with a Python model** (numpy, transcribing the reference
  sweep and Tessera's one-round `SharedFace` set with the M1-U3 claim rule).
  Radial and ellipsoidal perturbations gave one round only. The field
  `(sin(12y+1), sin(15z+2), sin(18x+3))`, tangential part, amplitude 0.1,
  renormalized, was predicted at 42 + 2 flips in 2 rounds; the C++ run gave
  44 in 2 rounds.
- **Cosmetic corrections:** the `kFlipsDisabledMinGain` comment
  (`Beatnik_Params.hpp`) no longer calls the pass unimplemented and states the
  1e12 claim exactly (unsatisfiable when both old faces have positive quality);
  `splitEdges()`'s brief, `DynamicRemesh.hpp`'s header table, two Solver
  comments, the unit-test CMake comment, and `milestone1.md` Current state
  (stubs moved out of "Throws" into an "Implemented, unreachable from a
  `Solver` run until T4d6" bullet; the inert bullet names only
  `diag.smooth_steps += 0`). Cited lines checked against the reference:
  the repair gate is `dynamic_remesh.py:169-175`, the duplicate skip `:430-431`.

**Signatures changed:**
- `int DynamicRemesh::flipEdgesForQuality( mesh_type&, RemeshDiagnostics& )
  const` (was `( mesh_type& )`; it had no caller).
- `int MeshQuality::improveConnectivityByFlips( mesh_type&, int max_passes,
  Real min_gain, std::vector<FlipSweepStats>* sweeps = nullptr ) const`.
- New: `SurfaceMesh::vertexGids()`, `FlippedEdge`, `MeshEditReport::flipped`,
  `Solver::kRefineFlipMinGain`, and the `MeshQuality.hpp` free symbols above.

**Re-basing:**
- `flipEdgesForQuality` does **not** call `AdaptiveMesh::resetReferenceState`.
  The reference's cycle passes no re-base flag, and the Solver already re-bases
  after every remesh that ran (`src/Beatnik_Solver.hpp`, the remesh branch),
  as T4d2 recorded.
- `improveConnectivityByFlips` does not either; **its call site does**, under
  the reference's exact conditions: `flips > 0 && smooth_iters == 0`
  (`reset_reference=(smooth_iters == 0)` at
  `run_adaptive_mesh_bubble.py:1440-1445`, and `_state_with_faces` is reached
  only when `total_flips > 0`, `mesh_solver.py:1770-1772`). `flips` is global,
  so the collective is entered together. The parked comment in
  `MeshQuality.hpp` and the Solver comment that said "T4d owns the question"
  now state this. Unreachable while `--flip-passes > 0` is rejected.
- `improveConnectivityByFlips` is **unexercised by any milestone run**:
  `--flip-passes` is 0 in the milestone configuration, as `relaxTangential`
  was after T4c. Only case 4d runs it.

**Measured counts** (integers asserted as literals; the two minimum qualities
are printed, compared only as rises-or-holds), identical on all 21 ranks at
np 1–6:

| Sub-case | rounds | requested | accepted | rejDup | rejConflict | min q before → after | below 0.18 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4a, gain 1e12 | 0 | 0 | 0 | 0 | 0 | 0.0775, unchanged | 3; mesh, gids, positions, generation unchanged |
| 4b, one sweep, gain 1e-3, floor 0.18 | 2 | 59 | 44 | 0 | 15 | 0.0775 → 0.2128 | 3 → 0; V/E/F, χ, gids unchanged |
| 4c, `remesh()`, split/collapse off | 2 | — | 44 | — | — | — | 1 sweep |
| 4c, `remesh()`, gain 1e12 | 0 | — | 0 | — | — | unchanged | 1 sweep; no mutation |
| 4d, `max_passes = 0` | — | — | 0 | — | — | unchanged | 0 sweeps; no mutation |
| 4d, sweep 0 (no floor) | 2 | 61 | 44 | 0 | 17 | | |
| 4d, sweep 1 | 2 | 14 | 10 | 4 | 0 | | |
| 4d, sweep 2 | 2 | 10 | 2 | 8 | 0 | 0.0775 → 0.2963 overall | 3 → 0 |

**Only running revealed:**
- **A duplicate-edge rejection is re-requested in every later round of the same
  sweep.** The mask does not test whether `(c,d)` exists (Do step 2), and a
  rejected flip touches no face, so the edge stays eligible. In 4d sweeps 1
  and 2 the second round accepted nothing, so every edge it requested was one
  the first round had also requested; sweep 2's 8 duplicate rejections are
  therefore the same edges counted twice (the per-round split is not printed).
  The sweep still ends, because a
  round that accepts nothing ends it, and the reference skips those edges too.
  It inflates `flip_requested` and costs one more `flipEdges` call per sweep.
- 4d's three sweeps each accepted something (44, 10, 2), so the early stop after
  an empty sweep is implemented but was not exercised by this fixture.
- Nothing else: no throw, and the predicted round structure held.

**Session notes:**
- The session was interrupted mid-build; `/tmp` (node-local) was wiped with it,
  including the Python model scripts. The model is described above; the
  rebuild logged to a persistent path. Two `spack install`s, both rc 0
  (12m23s, then 2m10s incremental), after unsetting `HIPCC_*_FLAGS_APPEND`.
  The build carries the same 70 warnings, all `-Wpass-failed` from
  `Kokkos_HIP_KernelLaunch.hpp`; none is in a file this task touched.
- `tasks/milestone1.md` carried uncommitted edits from outside this session
  when it resumed (a Conventions row "`Solver` rejections during T4d" and
  T4d4/T4d5 text). They are kept and committed with this section. That row's
  Solver line references predate this task's shifts (see above).
- clang-format was not run.

**Affects:**
- **T4d5:** reuse `qualityFlipSweep`'s shape — the gathered `MeshEditReport::
  flipped`, `vertexGids()`, `gatherNewDiagonals` and the empty-mask exit — with
  a touched set of **vertices** (all four quad corners per flip) and
  `conflict = EndpointInQuad`. Note the duplicate-edge re-request above.
- **T4d6:** deletes the two flip rejections (now `Solver.hpp:764-769` and
  `:848-856`) and their regression cases; after that the refine branch's
  re-base under `flips > 0 && smooth_iters == 0` becomes reachable.
- **M1-A1 / the T4d exit criterion:** compare flip statistics knowing one round
  accepts about three quarters of a dense candidate set (44 of 59 here) and the
  sweep needs a second round.

## T4d3 review

**Verdict:** Passed
**Reviewed:** `5fa0488^..5fa0488` (the one code commit; `c60f9e7` after it
touches only `tasks/`). It touches the **Fill in** (`src/Beatnik_DynamicRemesh.hpp`,
`src/Beatnik_MeshQuality.hpp`, `src/Beatnik_Solver.hpp`), what Do step 4
requires to reach `FlipResult::flipped` (`FlippedEdge`,
`MeshEditReport::flipped` and `SurfaceMesh::vertexGids()` in
`src/Beatnik_MeshInterface.hpp`, recorded under "Decisions made in the
session"), the `kFlipsDisabledMinGain` comment in `src/Beatnik_Params.hpp`
(recorded as a cosmetic correction), the test
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp` and its CMake comment, and the
README Known Issues and `docs/testing.md` lines CLAUDE.md requires to stay in
sync. Nothing out of scope.
**Evidence:** implementing session's job `f3iL4TUfQCCX`, output read.
`scripts/tuolumne/t4d1_mesh_edits.flux`, log
`beatnik_t4d1_mesh_edits.f3iL4TUfQCCX.log`, Flux `COMPLETED` rc 0. Submitted
10:15:41, started 10:15:42 (1 s queue wait), ran 1.053 min. It echoed commit
`5fa0488` with no source file dirty (only `.claude/settings.json`,
`tasks/milestone1.md` and untracked files), and the dev-view binary's mtime
(10:15:14) precedes the commit by 23 s with no source edit between. The log
has `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank counts)` and exactly
21 `[PASS] Beatnik_Test_MeshEdits (177/177 checks)` lines, all on HIP, with no
`FAIL`, abort or exception. Every case-4 note appears 21 times with identical
values, so the statistics agree at ranks 1 and 4 (and 2, 3, 5, 6). The passing
direction: 4b's single `flipEdgesForQuality` sweep runs 2 rounds, requests 59,
accepts 44 (15 `rejectedConflict`); minimum quality 0.0775 → 0.2128, faces
below 0.18 3 → 0, V 162 E 480 F 320 χ 2. The test asserts `min_q` after ≥
before and `below` after ≤ before, which is the criterion, and compares only
statistics, never flip sets (R7). 4c's `remesh()` flips the same 44 in 2
rounds through the repair gate; 4d's `improveConnectivityByFlips` at three
passes flips 44 + 10 + 2 = 56, minimum 0.2963. The failure direction: 4a and
4c at gain 1e12 report `flip_rounds = 0`, `flips = 0`, and generation, face
gids, every local position and V/E/F unchanged; 4d at `max_passes = 0` returns
0 with no sweep and the same checks. Every number agrees with the **Met.**
paragraph.

Mechanical pass (`review_checks.py`): candidates only, all dismissed. K4 (host
mirrors in `qualityFlipSweep`, `vertexGids()`) is the designed host mask pass,
once per round, not per step. M3 (`MPI_Allgather`/`MPI_Allgatherv` in
`gatherNewDiagonals`) hands host `std::vector` data, not View data. S2 at
`src/Beatnik_Solver.hpp:452` is prose. Read against the reference and the task:
- `QualityFlips::accepts` is written once, with the floor as
  `QualityFlipCriterion::has_floor`; the remesh path passes `min_quality` and
  `flip_min_gain`, the refine path no floor and `Solver::kRefineFlipMinGain =
  1e-3` citing `mesh_solver.py:1708` (Do step 1). Normal-sum, finiteness,
  floor and strict gain tests in the reference's order. It uses the reference
  metric, and `ReferenceCollapsePolicy::quality` now shares the same host
  function.
- No `(c,d)`-exists test in the mask (Do step 2). The policy is
  `{maxNormalDeviation = π, minQuality = 0, SharedFace}`, set explicitly (Do
  step 3, Conventions "Where a reference acceptance rule lives").
- One sweep is several rounds sharing a touched set, never a fresh mask (Do
  step 4). Keying it by new-diagonal gid pairs is equivalent to touched faces:
  the faces containing a newly created diagonal are exactly the two the flip
  rewrote, since a flip onto an existing edge is rejected as a duplicate.
- Residency: `faceAdjacency().non_resident` and an unresolved-quad count are
  reduced together and throw on every rank or none (Do step 1).
- Collectives: every branch in `qualityFlipSweep` is on a reduced count or on
  Tessera's global `accepted`; `improveConnectivityByFlips` returns before
  anything collective only on the parameter `max_passes`; the Solver's re-base
  is under the global `flips` and a parameter. Every rank reaches the same
  calls.
- Re-basing (Conventions row, Do step 5): recorded in the log with reasons,
  and the code matches it.
- Both `Solver` rejections and their regression cases are untouched
  (Conventions "`Solver` rejections during T4d").

Stale code: the `diag.flips += 0` placeholder, the refine branch's
`const int flips = 0` and both `BEATNIK_NOT_IMPLEMENTED` calls are gone; no
comment in `src/`, `README.md` or `docs/` still calls either entry point
unimplemented. `flipEdgesForQuality`'s old one-argument signature had no
caller. Nothing is made dead. No prompt file remains under `tasks/`.

1. [follow-up] `tests/unit_tests/Beatnik_Test_MeshEdits.cpp` case 4d: the
   early stop after a sweep that flips nothing is not exercised (the fixture's
   three sweeps flip 44, 10 and 2), as the log records. A fourth pass at the
   same fixture, or a `max_passes` large enough to reach an empty sweep, would
   cover it. Candidate for T4d6, which makes the path reachable.
2. [follow-up] The refine branch's `resetReferenceState` under
   `flips > 0 && smooth_iters == 0` (`src/Beatnik_Solver.hpp`) is unreachable
   until T4d6 and covered by no test. T4d6, when it deletes the
   `--flip-passes > 0` rejection, is where a run should show it fires.
3. [follow-up] A duplicate-edge rejection is re-requested in every later round
   of the same sweep, inflating `flip_requested` and costing one extra
   `flipEdges` call per sweep, as the log records under "Only running
   revealed". Correct (the reference skips those edges too), but a diagnostics
   reader should know `flip_requested` counts requests, not distinct edges.

**Affects:** none

## T4d4

**Commits:** `4041aec` (`DynamicRemesh::tangentialSmooth`, its wiring into
the repair gate, the header and Solver comments, case 5 of
`Beatnik_Test_MeshEdits`, README Known Issues and `docs/testing.md`), and the
commit that adds this section (`milestone1.md`, this log).

**Jobs:**
- `f3iLqpsgPez3`, the exit criterion, **both directions of the guard** and the
  no-op: `scripts/tuolumne/t4d1_mesh_edits.flux` at its default ranks
  1 2 3 4 5 6 (`pdebug`, two nodes, `-t 15m`, `--bank=unm`). Log
  `beatnik_t4d1_mesh_edits.f3iLqpsgPez3.log` in the repo root. COMPLETED rc 0,
  1.09 min, `flux job status` rc 0. It echoed commit `4041aec` with only
  unrelated files dirty (`.claude/settings.json`, `tasks/milestone1.md`,
  untracked files), and the dev-view binary, mtime 11:54:40, 29 s before the
  commit (11:55:09); no source file was edited between the build and the
  commit. `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank counts)`, and all
  21 ranks printed `[PASS] Beatnik_Test_MeshEdits (232/232 checks)` on HIP.
  The rank-to-GPU binding is the script's, unchanged. It was the only job: no
  sentinel run was needed (see "Measured").

**Decisions given to the task, recorded as given:**
- **Shape preservation is asserted in T4c's form, not as a volume bound.**
  Per accepted iteration, `max|Δx·n| ≤ 1e-13·max|Δx|` against the normals that
  iteration started from, both maxima global over owned vertices. The relative
  volume change is printed and asserted at least 1e3 times smaller in
  magnitude than one un-projected iteration at the same relaxation, built in
  the test from the same public operators. The reason: the projection cancels
  the volume change to first order only, and the reference itself changes
  volume by `-4.338e-07` relative (one iteration) and `-1.166e-06` (three) at
  relaxation 0.04 on this mesh, so a `1e-12` bound is unmeetable by a correct
  port.
- **Quality is asserted through the guard, not as "the minimum does not
  fall".** The mean must rise, and each accepted iteration must satisfy
  `min_after + 1e-14 ≥ max(min_quality, 0.85·min_before)`. The minimum is
  printed and not required to hold, because the reference lowers it on this
  mesh (0.97727413140883002 → 0.97725307555529162 → … 0.97721315828587241)
  while the mean rises (0.988529 → 0.989143). A port that held the minimum
  would be the altered behaviour M1-R4 describes.
- **The `Solver` rejection stays**, message text included
  (`src/Beatnik_Solver.hpp:838-845`), and so does the `tangentialSmooth`
  `checkRejected` case in `Beatnik_Test_DynamicRemeshSplit.cpp:699-701`. T4d6
  deletes both. The rejection's message still says "not yet implemented"; it
  is deleted whole at T4d6, so it was left as it is, as T4d2 did.

**Decisions made in the session:**
- **How the kernel is shared with `relaxTangential`:** `tangentialSmooth`
  calls the three public operators `relaxTangential` is built from —
  `MeshGeometry::compute` (vertex normals), `SurfaceOperators::
  graphLaplacianVector` over `vertexOneRing()`, and `SurfaceOperators::
  projectTangent` — directly. No new stencil and no new helper;
  `relaxTangential` and T4c's test literals are untouched. The only new
  kernels are the trial write (owned rows, skipping a row whose one-ring has
  fewer than three entries) and the restore.
- **The reference's three differences, as implemented:** the early return is
  `iterations == 0 || relaxation <= 0` (`max(iterations, 0)` first);
  `offsets(i+1) - offsets(i) < 3` leaves the trial row at `x_v`; and the guard
  uses `globalMinQuality` (the reference metric over owned faces, finite
  values only, `MPI_MIN`) before and after the trial. Only accepted
  iterations are counted.
- **Rejection restores in place.** The pre-trial owned rows are copied to a
  View, the trial is written into `positions()`, the halo is exchanged so the
  trial's owned faces see trial ghost corners, and on rejection the owned rows
  are copied back and the halo exchanged again. Ghosts then hold their owners'
  restored values, so every local row is bit-identical to entry (the test
  compares every local position with `==`). `haloExchange()` does not bump
  `generation()`, so a rejected call leaves the generation unchanged too.
- **Halo exchanges:** one before the loop, one after each trial write, and one
  after a restore. After an accepted iteration the mesh is already
  halo-consistent from the post-trial exchange, so no exchange is added at the
  end. At `smoothing_iterations = 0` (or relaxation ≤ 0) the function returns
  before any exchange, reduction or allocation.
- **`old_quality` is recomputed each iteration** rather than carried over
  from the previous iteration's `new_quality`. The two are the same reduction
  over the same positions; recomputing costs one reduction and keeps the code
  a line-for-line transcription.
- **No new `RemeshDiagnostics` field.** A rejection shows as `smooth_steps`
  below `smoothing_iterations · passes`; the signature stays
  `int tangentialSmooth( mesh_type& ) const`.
- **The test is case 5 of `Beatnik_Test_MeshEdits`**, no separate member.
  Every check is a global reduction or a bitwise all-rank comparison. Beyond
  the criterion it compares the one- and three-iteration minimum and mean
  quality with the reference at `1e-12` relative and the three-iteration
  volume change at `1e-8` (T4c's two tolerances, for T4c's reasons). Those
  literals came from calling the reference's own
  `tangential_smooth_vertices` (`min_quality = 0.18`, relaxation 0.04) on
  `mesh.icosphere_mesh( subdivisions=2, radius=1 )`. The model script lived in
  node-local `/tmp` and is not kept; it is the reference function plus a
  signed-volume sum and a guard-free trial loop.
- **Cosmetic corrections:** the stub's doc comment rewritten to describe the
  implementation and the guard (Doxygen math, as its neighbours use); the
  provenance line now cites `:459-490`; `DynamicRemesh.hpp`'s header table and
  "What is implemented" paragraph; the pass-loop comment; two `Solver.hpp`
  comments (`:451-455`, `:787-791`) that called smoothing unimplemented;
  README Known Issues and `docs/testing.md:140-148`. In `milestone1.md`: the
  Status line, Current state ("Throws" now lists only the two `MeshQuality`
  stubs; the inert bullet became an "Implemented, unreachable from a `Solver`
  run until T4d6" bullet for `tangentialSmooth`; collapse and flip line
  references refreshed), and the `Solver` line references in the Conventions
  row and T4d6's rejection table (now `:764-769`, `:777-783`, `:816-836`,
  `:838-845`, `:847-856`, comment block `:785-796`).
- Left for T4d6, as its entry says: the header of
  `tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp` (`:21-40`),
  which says everything but the split third is unimplemented.

**Re-basing:** `tangentialSmooth` does **not** call
`AdaptiveMesh::resetReferenceState`. The reference's cycle passes no re-base
flag to `tangential_smooth_vertices`, and the `Solver` remesh branch already
re-bases after every remesh that ran, as T4d2 and T4d3 recorded.

**The rejecting fixture:** the default icosphere at `min_quality = 0` and
relaxation 5.0, three iterations. With `min_quality = 0` the bound is
`0.85·old`, so the `0.85` branch is the one that binds. A guard-free Python
model of one projected iteration on this mesh gave the minimum-quality ratio
at relaxation 2, 3, 4, 5, 6 as 0.973, 0.884, 0.766, 0.636, 0.505; 5.0 leaves
a wide margin under 0.85. The test re-measures the premise from its own
unguarded iteration (the same operators, on a separate mesh): minimum
0.97727413140883013 → 0.62203242443688667, ratio 0.63650, and asserts
`trial + 1e-14 < 0.85·before` before asserting `smooth_steps == 0`.

**Signatures changed:** none. `int tangentialSmooth( mesh_type& mesh ) const`
keeps the stub's signature; its one caller is the repair gate.

**Measured** (integers and bounds identical on all 21 ranks; the real values
agree to the last digit or two across ranks, the differences being the
atomic normal assembly and the reduction order):

| Quantity | Iteration 1 | Iteration 2 | Iteration 3 |
| --- | --- | --- | --- |
| `smooth_steps` (single-iteration call) | 1 | 1 | 1 |
| `max|Δx|` | 1.06304e-3 | 1.01808e-3 | 9.75202e-4 |
| `max|Δx·n|` | 2.19e-17 | 6.85e-17 | 4.39e-17 |
| tangency ratio | 2.06e-14 | 6.73e-14 | 4.50e-14 |
| min quality after | 0.97725307555529173 | 0.97723275839760848 | 0.97721315828587252 |
| guard bound | 0.830683 | 0.830665 | 0.830648 |
| mean quality after | 0.98914296293991 | 0.98969688301295 | 0.99019655118968 |

- Before: minimum 0.97727413140883013, mean 0.988528666232463.
- One call at three iterations: `smooth_steps = 3`; it lands within
  `4.5e-20` of the three single calls; relative volume change
  `-1.16636709e-06` (reference `-1.16636709257544879e-06`).
- One un-projected iteration at 0.04: `-5.37065006530e-03`; factor `4604.6`
  over the guarded pass (asserted ≥ 1e3).
- Case 5c: `smooth_steps = 0`, every local position bit-identical,
  generation unchanged.
- Case 5a: `smoothing_iterations = 0`, relaxation 0 and relaxation −0.04 each
  return 0, positions bit-identical, generation unchanged.
- Case 5d: `remesh()` on case 4's fixture with flips off and
  `smoothing_iterations = 0`: `flip_sweeps = 1` (the gate fired),
  `smooth_steps = 0`, positions bit-identical. With flips on and one
  smoothing iteration: `flips = 44`, `smooth_steps = 1`, minimum quality
  0.07745 → 0.26479 (the flips alone reach 0.2128, case 4b).

**Only running revealed:** nothing that changed the code. The second
iteration's tangency ratio, `6.73e-14`, sits at two thirds of the `1e-13`
bound — less margin than T4c's sweeps had at relaxation 0.12. The bound is the
criterion's, and the ratio is a rounding residual, but a reviewer reading a
future failure at this check should look at the arithmetic before the
projection. `kRemeshSmoothSteps = 1` was written down before the run, from
the argument that after the flips (minimum 0.2128) the guard's bound is
`max(0.18, 0.181)` and one 0.04 iteration barely moves the minimum. The run
confirmed it, so no sentinel run was needed.

**Session notes:**
- One `spack install`, rc 0, 10m31s, a full rebuild (staging, cmake, build,
  install), after unsetting `HIPCC_*_FLAGS_APPEND`. Logs:
  `~/spack_envs/tuolumne_beatnik/t4d4_spack_install.log` and the build output
  `t4d4_build_out.txt` beside it. 70 warnings, all `-Wpass-failed` (the
  accepted Canopy and Kokkos HIP notes); none is in a file this task touched.
- clang-format was not run.

**Affects:**
- **T4d6:** it deletes the `--remesh-smooth-iters > 0` rejection (now
  `Solver.hpp:838-845`) and the `checkRejected` smoothing case, as planned.
  After that, the default `--remesh-smooth-iters 1` runs this pass in every
  remesh whose repair gate fires. The T4d6 table's line references are
  refreshed as of this task.
- **M1-T2 / M1-A1:** `smooth_steps` is the accepted-iteration count. A run
  whose `smooth_steps` total is below `smoothing_iterations ×` (gate firings)
  had the guard reject a trial; that is the reference's behaviour, not a
  fault.

## T4d4 review

**Verdict:** Passed
**Reviewed:** `4041aec^..4041aec` (the one code commit; `599d909` after it
touches only `tasks/`). It touches the **Fill in** (`tangentialSmooth` and the
`diag.smooth_steps` placeholder in `src/Beatnik_DynamicRemesh.hpp`), two
`src/Beatnik_Solver.hpp` comments that called smoothing unimplemented
(recorded as cosmetic corrections), the test
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`, and the README Known Issues and
`docs/testing.md` lines CLAUDE.md requires to stay in sync. Nothing out of
scope; the `Solver` rejection at `src/Beatnik_Solver.hpp:838-845` and the
`checkRejected` smoothing case in `Beatnik_Test_DynamicRemeshSplit.cpp` are
untouched (Conventions "`Solver` rejections during T4d").
**Evidence:** implementing session's job `f3iLqpsgPez3`, output read.
`scripts/tuolumne/t4d1_mesh_edits.flux`, log
`beatnik_t4d1_mesh_edits.f3iLqpsgPez3.log`, Flux `COMPLETED` rc 0. Submitted
11:55:12, started 11:58:59 (about 3.8 min queue wait), ran 1.087 min. It
echoed commit `4041aec` with no source file dirty (`.claude/settings.json`,
`tasks/milestone1.md` and untracked files only), and the installed binary's
mtime (11:54:40) precedes the commit. The log has `PASS np=1` … `PASS np=6`,
`SUMMARY: PASS (6/6 rank counts)` and exactly 21
`[PASS] Beatnik_Test_MeshEdits (232/232 checks)` lines on HIP, with no `FAIL`,
abort or exception, so ranks 1 and 4 are both covered. Passing direction (5b):
three accepted single-iteration calls with tangency ratios `2.06e-14`,
`6.73e-14`, `4.50e-14` (asserted ≤ `1e-13`), each minimum above its guard
bound (≈0.8307, asserted), mean 0.988529 → 0.990197 (asserted to rise);
one call at three iterations returns 3 and matches the single calls to
`2.7e-20`; volume change `-1.166e-06` against `-5.371e-03` un-projected, factor
4605 (asserted ≥ 1e3); minimum, mean and volume agree with the reference's
`tangential_smooth_vertices` literals. Failure direction (5c): the test first
shows its own unguarded trial at relaxation 5 takes the minimum to ratio
0.6365 < 0.85, then `tangentialSmooth` at `min_quality = 0` returns 0 with
every local position bit-identical and the generation unchanged. The no-op
(5a, 5d): `smoothing_iterations = 0` and relaxation ≤ 0 return 0 with
positions bit-identical, directly and through `remesh()` with the repair gate
firing. Every number agrees with the **Met.** paragraph.

Mechanical pass (`review_checks.py`): candidates only, all dismissed. K4 (two
host mirrors) is in test code. S2 at `Beatnik_Test_MeshEdits.cpp:225` is
prose. Read against `dynamic_remesh.py:459-490` and the task:
- Line-for-line transcription: `max(iterations, 0)`, the `relaxation <= 0`
  early return before any collective, `old_quality` over the current
  positions, normals and one-ring per iteration, Jacobi trial (`saved` copy,
  displacement computed before the write), `< 3` neighbours keeps `x_v`, the
  guard `new + 1e-14 < max(min_quality, 0.85·old)` with `break`, and only
  accepted iterations counted (Do steps 2, 3).
- Do step 1: the stencil is `relaxTangential`'s three public operators,
  called directly; no new umbrella operator. Do step 4: `projectTangent` is
  kept, and the test's un-projected comparison is the evidence.
- Collectives: after the parameter-only early return, every branch is on
  `globalMinQuality` (an `MPI_MIN`) or the loop count, so every rank reaches
  the same halo exchanges and reductions. Only owned rows are written, the halo
  is exchanged after each trial write and after a restore, both kernels are
  fenced before the exchange.
- Re-basing (Conventions row): recorded in the log with its reason; the code
  calls no `resetReferenceState`, as recorded.

Stale code: the `BEATNIK_NOT_IMPLEMENTED` stub, the `diag.smooth_steps += 0`
placeholder and its `//   tangentialSmooth      T4d4` marker are gone; no
comment in `src/`, `README.md` or `docs/` still calls smoothing unimplemented.
The header of `Beatnik_Test_DynamicRemeshSplit.cpp:21-40` is left for T4d6,
as the log and T4d6's entry say. Nothing is made dead. No prompt file remains
under `tasks/`.

1. [nit] `src/Beatnik_DynamicRemesh.hpp` `tangentialSmooth`: the
   `displacement` and `saved` Views are allocated inside the iteration loop.
   Harmless at `--remesh-smooth-iters 1`, but hoisting them above the loop
   (sizes are fixed within a call) is the usual Kokkos practice.
2. [follow-up] No test exercises a rejection *after* an accepted iteration
   (accept, then reject and restore to the accepted state, returning 1). 5c
   rejects on the first trial. The code path is the same restore, but a
   fixture where the guard binds on iteration 2 or later would cover the
   `accepted > 0` return with restored positions.
3. [follow-up] The second iteration's tangency ratio, `6.73e-14`, is at two
   thirds of the `1e-13` bound, as the log records under "Only running
   revealed". A future failure of that check should be read as rounding
   before as a projection defect.

**Affects:** none

## T4d5

**Commits:** `8120fa1` (`valenceEqualizingFlips`, `isotropicCleanup`, the
shared flip helpers, `Solver::rebuildState` and both call sites, case 6 of
`Beatnik_Test_MeshEdits`, README Known Issues, `docs/testing.md` and the
`DynamicRemesh.hpp` header table), and the commit that adds this section
(`milestone1.md`, this log).

**Jobs:**
- `f3iWgRGrp42K`, the exit criterion, **both directions of the switch**,
  the touched rule, the valence fall, the volume and tangency clauses and the
  early stop: `scripts/tuolumne/t4d1_mesh_edits.flux` at its default ranks
  1 2 3 4 5 6 (`pdebug`, two nodes, `-t 15m`, `--bank=unm`), unchanged. Log
  `beatnik_t4d1_mesh_edits.f3iWgRGrp42K.log` in the repo root. COMPLETED,
  66 s, `flux job status` rc 0. `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6
  rank counts)`, all 21 ranks `[PASS] Beatnik_Test_MeshEdits (282/282
  checks)` on HIP. It echoed commit `2f2dd73` with the task's files dirty (the
  run preceded the code commit `8120fa1`) and the dev-view binary, mtime
  08:46:12. Every source edit in `8120fa1` was in the build except a
  comment-only edit to the `DynamicRemesh.hpp` header table, made after the
  `spack install` had started; README, `docs/testing.md` and the design doc
  are not compiled. It was the only job: the Python model's literals held on
  the first run, so no sentinel run was needed.
- Submitted through the loop's `hpc_submit.py` with the loop's environment set
  by hand: the session was resumed outside the loop and `TASKLOOP_SUBMIT` was
  unset.

**The user's ruling, made in this session (2026-11-06):** the valence flips'
normal test uses the **new faces as Tessera winds them**, not the reference's
literal `(c,d,a)`, `(d,c,b)`. Found while modelling the reference
(`mesh_quality.py:76-79`): which face is `f0` depends on face index alone
(`edges_from_faces` order), and on every edge where `f0` runs `a → b`
(`a < b` by index) both literal children are inverted. On the default
icosphere that is 206 of 480 edges, worst dot `-0.983`; the other 274 pass,
best `0.955`. So the reference's normal test mostly rejects by array index.
Beatnik's `resident_faces` order is rank-local, so a literal copy would reject
a rank-count-dependent set. Three options went to the user: consistent
winding (1), a gid-based imitation of the artifact (2), and a literal copy (3).
The user chose 1. On a scrambled icosphere the model gave 7, 1, 0 flips per
pass for the reference (7 of 15 valence candidates rejected by the normal
test, 25 irregular after) against 7, 2, 0 (3 of 12, 26 after). Recorded under
Deliberate deviations in `milestone1.md`, in Do step 1 and in README Known
Issues.

**Decisions given to the task, recorded as given:**
- **The volume clause is T4d4's form, not a `1e-12` bound.** The valence
  flips' relative volume change is printed and not bounded; each relaxation
  pass is asserted tangent, `max|Δx·n| ≤ 1e-13·max|Δx|`, both maxima global;
  the relaxation passes' volume change is asserted at least `1e3` times
  smaller than one un-projected pass at weight 0.4, built in the test from the
  same public operators (`unguardedIteration` with `project = false`).
- **The refine branch follows the reference's rebuild.** After
  `isotropicCleanup`, under `_params.cleanup.enabled`, `rebuildState()`
  (`resetReferenceState` + `centerPotential`), before the volume projection.
  The comment that said nothing re-bases the reference state after the
  tangential pass now says that under `--isotropic-cleanup` the rebuild after
  the cleanup does.
- **The `Solver` rejection stays**, message text included (now
  `src/Beatnik_Solver.hpp:791-797`), and so does the `checkRejected` case in
  `Beatnik_Test_DynamicRemeshSplit.cpp:705-707`. The other four T4d
  rejections are untouched (now `:778-783`, `:830-850`, `:852-859`,
  `:861-870`).
- **The mask holds only the reference's tests:** the valence-deviation test,
  the two `≥ 0.2` normal dots (each child against its own old face), and
  `min(q_new) ≥ 0.05` in `QualityFlips::triangleQuality`. `c == d` is
  skipped; whether `(c,d)` exists is not tested (Tessera's
  `rejected_duplicate_edge`). Policy `flip_policy_type` with
  `maxNormalDeviation = π`, `minQuality = 0`, `conflict = EndpointInQuad`,
  each set explicitly. The two thresholds are named constants
  `kMinValenceFlipNormalDot` and `kMinValenceFlipQuality`, citing
  `mesh_quality.py:78` and `:80`.
- **Pass structure:** rounds until one accepts nothing, never a fresh mask
  within a pass; a pass-wide touched set of vertex gids holding all four
  corners of every accepted flip, from the gathered `flipped` lists; passes
  stop after one that flips nothing.

**Decisions made in the session:**
- **How the pass-start valences are obtained, including at ghost endpoints.**
  Each round takes the *current* valence (owned rows from `vertexOneRing()`
  row sizes, which on a closed manifold equal the face-incidence count; ghost
  rows by `haloExchangeVertexView` on a `Real` view) and adds a per-gid
  `shift`: `+1` for each of this pass's flips with the vertex as an old
  endpoint, `-1` for each with it as a new endpoint. The shift comes from the
  gathered flips, the same on every rank, so the result is exactly the
  pass-start valence, keyed by gid, including for a vertex that entered a
  rank's halo mid-pass. That is one extra vertex halo exchange per round.
- **Pass-start face normals are not stored.** A face rewritten in the pass
  has all three corners touched, so no later candidate edge lies on it, and
  positions do not move during the flips, so the faces the mask reads are
  still at their pass-start state.
- **Child winding:** if `f0` runs `a → b` the children are `(c,a,d)`
  (replaces `f0`) and `(d,b,c)` (replaces `f1`), otherwise the reference's
  `(c,d,a)` and `(d,c,b)`: the quad boundary is `a,d,b,c` in the first case.
- **The switch is shown through `MeshQuality`, not `advanceOneStep`.**
  `isotropicCleanup` returns before anything collective when
  `CleanupParams::enabled` is false; the `Solver` constructs `MeshQuality`
  from `_params.cleanup` and also gates both call sites (and their rebuilds)
  on `_params.cleanup.enabled`. A Solver-level test needs a full setup; the
  gate tested is the same flag in the same struct.
- **Shared helpers:** `QualityFlips::gatherFlips` (the gathered, deduplicated
  `FlippedEdge` list; `gatherNewDiagonals` now reduces it to new diagonals,
  same behaviour, the error message now begins "Beatnik flips:"),
  `QualityFlips::hostPositions` and `QualityFlips::oppositeCorner`.
  `qualityFlipSweep` uses the last two instead of its inline copies, so
  `MeshQuality.hpp` has one copy of each.
- **`valenceEqualizingFlips` exchanges the halo on entry** (T4d1 Affects: the
  mask reads ghost corners). `isotropicCleanup` is then halo-consistent on
  return, through `relaxTangential`'s closing exchange.
- **`Solver::rebuildState()`** is a new private method, the port of
  `MeshPotentialZModelState.__post_init__` (`mesh_solver.py:138-163`), used
  three times; the remesh branch's inline rebuild block now calls it.
- **Fixture chosen with a Python model** (persistent, outside the repo:
  `~/spack_envs/tuolumne_beatnik/t4d5_model/fixture_model.py`, plus
  `orient_check.py` for the winding finding). It transcribes the reference
  pass, Tessera's `SharedFace` and `EndpointInQuad` rounds with the M1-U3
  claim rule, and the Beatnik pass. A 30° cap gave one round per pass;
  45°, one scramble round, gave pass 0 three rounds; 60° with two scramble
  rounds gave a second pass with no flip (the early-stop sub-case). Every
  predicted literal held in C++.
- **Cosmetic corrections** (the prompt's list): `milestone1.md` Status line,
  Problem table and its `requireSupportedConfiguration` range (`:759-892`),
  Conventions row, T4d6 table and comment-block reference (`:799-810`),
  T4d5 Fill in and rejection references, Current state ("Throws" bullet
  removed; "Missing entirely" bullet replaced by an "Implemented, unreachable
  from a `Solver` run until T4d6" bullet; the `improveConnectivityByFlips`,
  `qualityFlipSweep`, collapse and `EndpointInQuad` references refreshed);
  `tangentialRelaxation`'s and `valenceEqualizingFlips`' doc comments;
  `Solver.hpp`'s "3. isotropicCleanup is task T4d5" and remesh-branch
  comments; the `DynamicRemesh.hpp` header table's cleanup row (now cites
  `:1491`); README Known Issues and `docs/testing.md:147`.

**Re-basing (Conventions row):**
- `valenceEqualizingFlips` and `isotropicCleanup` do **not** call
  `AdaptiveMesh::resetReferenceState`; `MeshQuality` does not depend on the
  AMR header. The reference's callers rebuild the state, so the call sites
  do.
- **Remesh branch** (Do step 4): `remesh` → `rebuildState()` →
  under `enabled`, `isotropicCleanup` → `rebuildState()` → `projectToVolume`.
  The cleanup is gated on the remesh having run (inside `if ( remeshed )`),
  `:1491`'s `remesh_diag is not None`.
- **Refine branch** (the decision above): after the tangential pass, under
  `enabled`, `isotropicCleanup` → `rebuildState()`, then the projection under
  the unchanged `repaired` gate. The `flips > 0 && smooth_iters == 0` re-base
  before it is unchanged.
- Under `--no-isotropic-cleanup` neither the cleanup nor its rebuild runs, so
  every `regression` trajectory is unchanged (the remesh branch's first
  rebuild is the same code as before, moved into the method).

**Signatures changed:**
- `int MeshQuality::valenceEqualizingFlips( mesh_type&, int passes,
  std::vector<ValenceFlipPass>* record = nullptr ) const` (the stub's
  signature plus a defaulted trailing record, as T4d3 gave
  `improveConnectivityByFlips`; its only caller is `isotropicCleanup`).
- `void MeshQuality::isotropicCleanup( mesh_type& ) const`: unchanged.
- New: `ValenceFlipPass`, `QualityFlips::{gatherFlips, hostPositions,
  oppositeCorner}`, `kMinValenceFlipNormalDot`, `kMinValenceFlipQuality`,
  `Solver::rebuildState()` (private).

**Measured** (identical on all 21 ranks except the last digits of the real
values, which come from the atomic normal assembly):

| Sub-case | Result |
| --- | --- |
| 6a, `enabled = false`, and `passes = 0` | positions, face gids, generation bit-identical; returns 0 |
| fixture | 14 scramble flips; valence ≠ 6: 35, maximum 10 |
| 6b pass 0 | 3 rounds, requested 18, accepted 6 (rounds 4, 1, 1 in the model), rejectedConflict 12 |
| 6b pass 1 | 1 round, requested 7, accepted 3, rejectedConflict 4 |
| 6b pass 2 | 1 round, requested 2, accepted 1, rejectedConflict 1 |
| 6b after | 10 flips; valence ≠ 6: 35 → 20; maximum 10 → 7; touched-rule violations 0; ΔV/V `+1.2006751809e-03` |
| 6c relaxation pass 1 | `max|Δx|` 4.4513e-02, `max|Δx·n|` 9.46e-17, ratio 2.12e-15 |
| 6c relaxation pass 2 | `max|Δx|` 2.7246e-02, `max|Δx·n|` 6.89e-17, ratio 2.53e-15 |
| 6c volume | relaxation ΔV/V `3.1015090185e-05`; one un-projected pass `-5.3809043581e-02`; factor 1734.9 |
| 6d, `enabled = true` | valence ≠ 6: 20; within 1.1e-16 of the separate calls; positions and generation changed |
| 6e, early stop | 2 of 3 passes run: 1 flip in 1 round, then 0 |

No duplicate-edge or geometric rejection occurred on this fixture.

**Only running revealed:** nothing that changed the code; the model's
counts, including the per-pass `requested`, matched exactly. The relaxation's
tangency ratios here (~2.5e-15) sit far below T4d4's 6.7e-14 because the
displacements are 40 times larger at weight 0.4.

**Session notes:**
- One `spack install`, rc 0, 12m19s, staging, cmake and a full build, after
  unsetting `HIPCC_*_FLAGS_APPEND`. Logs:
  `~/spack_envs/tuolumne_beatnik/t4d5_spack_install.log` and the build output
  `t4d5_build_out.txt` beside it. 70 warnings, all `-Wpass-failed`; none in a
  file this task touched.
- clang-tidy does not apply: spack mode, no `compile_commands.json` and no
  `.clang-tidy`. clang-format was not run.
- The session was paused once, mid-implementation, for the user's ruling
  above, and resumed outside the loop.

**Affects:**
- **T4d6:** it deletes the `--isotropic-cleanup` rejection (now
  `Solver.hpp:791-797`) and its `checkRejected` case; after that the default
  configuration runs the cleanup and the extra `rebuildState()` on every
  remesh and every marking refine. The T4d6 table's references are refreshed
  as of this task.
- **M1-T2 / M1-A1:** Beatnik's valence flips accept flips the reference never
  makes (the winding deviation), so its valence histogram after a cleanup
  should be at least as regular as the reference's, and the flip counts are
  not comparable one for one. Compare the valence histograms, not flip
  totals.

## Long-job split (design only)

**Commits:** the commit that adds this section (`milestone1.md`, this log).

The remaining tasks whose exit criteria wait on a long job were split into
submit/post-job pairs, under the new Conventions row "Long jobs".

- **T4d6 → T4d6j + T4d6.** The 600-step cadence run at ranks 1 and 4 is
  estimated at about 12 min per launch, from M1-G2's Python wall of 707 s
  (`## M1-G2`); no Beatnik dynamic-remesh run has been timed. The standalone
  "T4d exit criterion" section had no owning task. Its 50-step umbrella run now
  goes in T4d6j's job and is judged in T4d6, the first point at which it can
  run, because T4d6j deletes the rejections. T4d6j's job also gains a control
  launch with renumbering off, so the bound has a failure direction. That needs
  a cadence value meaning "never".
- **M1-T1 deleted, merged into M1-T2j.** M1-T1's exit criterion listed the
  M1-T2 member, whose source only M1-T2 wrote, and M1-T2 depended on M1-T1. Its
  bare `run_milestone.flux` would also have run the whole tier, including the
  4.2 h FMM jobs. Registration, the gold `install()` rule and a dedicated
  staged runner, `m1_milestone_run.flux`, now live in M1-T2j. The task loop
  protects `run_milestone.flux` and `submit_milestone_split.sh`.
- **M1-T2 / M1-A1 cycle broken.** M1-T2 ran at M1-A1's depth, while M1-A1
  depended on M1-T2's first failing step. The order is now: M1-T2j submits a
  600-step measurement whose test continues past failures and prints per-step
  `max|e|`; M1-T2 reads it; M1-A1j decides with the user, encodes the decision
  and submits the confirming run; M1-A1 judges it green.
- The gate (8 and about 11 min in the last two runs `framework-progress-log.md`
  records) stays inside the submit tasks and is waited on in the session.

**Affects:** T4d6j, T4d6, M1-T2j, M1-T2, M1-A1j and M1-A1: new or rewritten
entries. Every reference to M1-T1 in earlier sections of this log now means
M1-T2j.

## M1-T2j protected-file edits, made ahead (design only)

**Commits:** the commit that adds this section (`CLAUDE.md`, `docs/testing.md`,
`scripts/tuolumne/submit_milestone_split.sh`, `milestone1.md`, this log).

The task loop protects CLAUDE.md and `submit_milestone_split.sh`. So both were
changed ahead of M1-T2j, in forms that stay true before and after the member
lands:

- **CLAUDE.md** keeps the `milestone` tier's rules: not the gate, one `pbatch`
  job per (member, backend), the FMM ruling and the parallel-filesystem scratch
  rule. It defers the member list, the counts and the measured walls to
  `docs/testing.md`, which already carried them. Adding a member no longer
  touches CLAUDE.md.
- **`submit_milestone_split.sh`** accepts an optional fourth row field,
  `pending=<task ID>`. A pending row is submitted only when its
  `<member>_MPI_<backend>` line is in the installed
  `beatnik_milestone_manifest.txt`. Otherwise it prints `PENDING …` to stderr
  and is skipped. Unmarked rows are unchanged, so a missing established member
  still fails in the runner. The `Beatnik_Test_Milestone1Bubble` SERIAL and HIP
  rows are in, `pending=M1-T2j`, at `-t 40m`, an estimate of about 1.5x the
  M1-G2 Python bound of 2 x 707 s. The script was checked with a stub `flux` on
  the login node. Today both milestone-1 rows print PENDING and the eight
  milestone-0 rows submit. A copy that marked `Milestone0Frozen HIP` pending
  still submitted it, because that member is installed.
- `docs/testing.md` is not protected. Its member list stays M1-T2j's. Only the
  hardcoded "eight separate `pbatch` jobs" became "one `pbatch` job per
  (member, backend)".

**Affects:** M1-T2j: no edit to CLAUDE.md or the split script; it updates the
`docs/testing.md` entry. M1-A1: its log asks the user to re-time the two rows
and drop `pending=M1-T2j`.

## T4d5 review

**Verdict:** Passed
**Reviewed:** `8120fa1^..8120fa1` (the one code commit; `d126add`, `38a612b`
and `a5dba69` after it touch no file under `src/` or `tests/`). It touches the
**Fill in** (`valenceEqualizingFlips` and `isotropicCleanup` in
`src/Beatnik_MeshQuality.hpp`, both call sites in `Solver::advanceOneStep`),
the shared helpers and `Solver::rebuildState` the log records, the test
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`, the README Known Issues and
`docs/testing.md` lines CLAUDE.md requires to stay in sync, and one
comment-only row of the `src/Beatnik_DynamicRemesh.hpp` header table (recorded
as a cosmetic correction). Nothing out of scope. The `--isotropic-cleanup`
rejection at `src/Beatnik_Solver.hpp:791-797` and its `checkRejected` case in
`Beatnik_Test_DynamicRemeshSplit.cpp` are untouched (Conventions "`Solver`
rejections during T4d").
**Evidence:** implementing session's job `f3iWgRGrp42K`, output read.
`scripts/tuolumne/t4d1_mesh_edits.flux`, log
`beatnik_t4d1_mesh_edits.f3iWgRGrp42K.log`, Flux `COMPLETED` rc 0, submitted
and started 08:46:46 (no queue wait), ran 1.10 min. Provenance needed more
than the echo: the job echoed `2f2dd73` with the task's files dirty, because
it ran before the code commit `8120fa1`. The dirty source set it printed is
exactly `8120fa1`'s file list. The installed `Beatnik_MeshQuality.hpp`,
`Beatnik_Solver.hpp` and `Beatnik_DynamicRemesh.hpp` in the dev view's
`include/` are byte-identical to `8120fa1`'s. Every string literal of 12+
characters that `8120fa1` adds to the test is present in the binary
(`share/Beatnik/tests/Beatnik_Test_MeshEdits`, mtime 08:46:12); the only two
misses were regex fragments spanning string concatenations. So the binary is
`8120fa1`'s code, and the evidence stands without a verification job. The log
has `PASS np=1` … `PASS np=6`, `SUMMARY: PASS (6/6 rank counts)` and 21
`[PASS] Beatnik_Test_MeshEdits (282/282 checks)` lines on HIP, with no
`FAIL`, abort or exception, so ranks 1 and 4 are both covered. Passing
direction: three valence passes accept 6, 3, 1 over 3, 1, 1 rounds; 0
touched-rule violations in every pass; valence ≠ 6 falls 35 → 20 (maximum
10 → 7); V/E/F and χ unchanged; flips' ΔV/V `+1.2007e-3`, printed; relaxation
tangency ratios `2.12e-15`, `2.53e-15` (bound `1e-13`); relaxation ΔV/V
`3.1015e-05` against `-5.3809e-02` un-projected, factor 1735 (asserted ≥ 1e3);
early stop runs 2 of 3 passes. Both directions of the switch: 6a
(`enabled = false`, and `passes = 0`) leaves positions, face gids and
generation bit-identical; 6d (`enabled = true`) changes positions and
generation and lands within `2.8e-17` at most (asserted ≤ `1e-13`) of the
separate calls. Every number agrees with the **Met.** paragraph. The checked
tolerances sit orders of magnitude from their bounds, so the atomic-assembly
nondeterminism the log notes is no reason for a second run.

Mechanical pass (`review_checks.py`): candidates only, all dismissed. K4 (nine
host mirrors in `valenceEqualizingFlips` and `hostPositions`) is the host-mask
design the Conventions row "Where a reference acceptance rule lives" chose,
the same pattern as `qualityFlipSweep`. S4 `BEATNIK_NOT_IMPLEMENTED` hits are
other tasks' stubs; none remains in `Beatnik_MeshQuality.hpp`. Read against
`mesh_quality.py:44-87`, `:146-167` and `run_adaptive_mesh_bubble.py:1438-1504`:
- Do step 1: dev0/dev1 transcribed exactly; `c == d` skipped; no `(c,d)`
  existence test; both `≥ 0.2` dots and `min(q_new) ≥ 0.05` as named
  constants citing `:78` and `:80`; policy set explicitly to
  `{π, 0, EndpointInQuad}`. The winding derivation is correct: when `f0`
  runs `a → b` the quad boundary is `a→d→b→c`, so `(c,a,d)` and `(d,b,c)`
  are the consistently wound children, each tested against the face it
  replaces (Deliberate deviations).
- Do step 2: pass-wide touched set of vertex gids, edges with a touched
  endpoint left out of the mask, all four corners added from
  `gatherFlips`, round loop ends on an empty global mask or zero accepts,
  passes stop after one with no flip. The per-gid `shift` (+1 old
  endpoint, −1 new endpoint) reconstructs pass-start valence exactly.
  Pass-start normals are read from untouched faces only, as argued.
- Do step 3: `isotropicCleanup` = valence passes then `relaxTangential` at
  the cleanup params; `enabled` checked before anything collective.
- Do steps 4–6: remesh branch `remesh` → `rebuildState` → (enabled)
  `isotropicCleanup` → `rebuildState` → projection, inside `if ( remeshed )`;
  refine branch adds the cleanup and its rebuild inside
  `marked_faces > 0`, with the `repaired` gate untouched; the comment at
  `src/Beatnik_Solver.hpp:491-497` rewritten as asked. Matches the reference
  call sites line for line.
- Collectives: every branch after the parameter-only early return is on a
  global count (`MPI_Allreduce` of the three counters, `report.accepted`) or
  the gathered flips; `faceAdjacency()`, `haloExchangeVertexView` and
  `flipEdges` are reached by every rank. The one device kernel
  (`hostPositions`) captures Views by value, no `this`, and is fenced.
- Conventions: provenance comments with line ranges on all three ports; loud
  failure on a non-resident quad; re-basing recorded in the log with reasons.
- Tests: case 6 lives in the registered `unit` member; each clause of the exit
  criterion is an assertion with a literal, including both switch directions.

Stale code: both `BEATNIK_NOT_IMPLEMENTED` stubs are gone; the remesh
branch's "It is T4d and unimplemented" comment and its inline rebuild block
are gone (the block is now `rebuildState`); `qualityFlipSweep`'s inline copies
of `hostPositions` and `oppositeCorner` are gone, and `gatherNewDiagonals` is
still called. Nothing is made dead. No prompt file remains under `tasks/`.

1. [nit] `tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp:305-311`
   still says "Flips and isotropic cleanup are unimplemented (T4d)". Both are
   now implemented (flips by T4d3, the cleanup by this task); what remains
   true is that the `Solver` rejects them. Outside this task's **Fill in**,
   and T4d6j edits this file when it deletes `checkFailureDirection`, so it
   is the natural place to rewrite the comment (CLAUDE.md "Describe the code
   as it is now").
2. [follow-up] `isotropicCleanup` returns `void`, so the cleanup's flip count
   reaches no run diagnostic. The reference discards it too, and the
   Conventions row "Diagnostics that must be numbers" lists the remesh pass's
   counters, not the cleanup's. But M1-T2/M1-A1 are told to compare valence
   statistics after cleanup (`## T4d5` Affects); a per-step cleanup flip
   count or valence-≠-6 count in the run output would give them a number to
   compare without a checkpoint post-process.

**Affects:** T4d6j (nit 1: rewrite the `RefineSplitEdges.cpp:305-311` comment
when it edits that file).

## T4d6j open questions settled (design only)

**Commits:** the commit that adds this section (`milestone1.md`, this log).

T4d6j's "Additional information needed" is resolved and removed. Both answers
now stand in the task as design:

- **`RefineSplitEdges` check 6.** It asserted that `--flip-passes 2` is
  rejected, and T4d6j deletes that rejection. No refine-branch rejection is
  left, the only remaining general one is `--field-filter-every` (T5c), and the
  refiner's own throws (`AdaptiveMesh.hpp:930`, `:1754`) cannot be reached from
  `SolverParams`. Retargeting to T5c was rejected because it is unrelated to
  refinement and would be deleted again at T5c. Dropping the check was rejected
  because it would leave flips unguarded. The replacement is a flips liveness
  check: a second 20-step `--flip-passes 2` solver must reach cumulative
  `flips > 0`, with Euler and conformity after every pass.
- **The driver.** It is the new `Beatnik_Test_Milestone1Run`, in
  `BEATNIK_DRIVER_SOURCES`, with three positionals (steps, checkpoint cadence,
  renumber cadence). `adaptive_mesh_bubble` is HIP-only (`:209`), and
  `Milestone0Run` asserts a constant mesh and forbids new options. Running the
  umbrella on HIP only was rejected because it narrows the criterion.
  `makeParams()` goes in `Beatnik_Milestone1Params.hpp` so M1-T2j's test
  shares it.

**Affects:** T4d6j: no longer needs these questions answered, so the loop does
not pause on it. M1-T2j: includes `Beatnik_Milestone1Params.hpp` rather than
writing its own `makeParams()`.

## T4d6j

**Commits:** `092e2e5` (the code, the tests, the driver and its header, the
CMake entry, README, `docs/testing.md`, `scripts/tuolumne/unit_tests.flux` and
the new `scripts/tuolumne/t4d6_gid_cadence.flux`), and the commit that adds
this section (`milestone1.md`, this log).

**Jobs:**
- `f3iXQgeQwhKD`, `scripts/tuolumne/unit_tests.flux` at `BEATNIK_UNIT_RANKS=1`,
  log `beatnik_unit_tests.f3iXQgeQwhKD.log`. COMPLETED rc 0,
  `[unit] SUMMARY: PASS (8/8 tests)`. Passing direction.
- `f3iXQgvioo7D`, the same at `BEATNIK_UNIT_RANKS=4`, log
  `beatnik_unit_tests.f3iXQgvioo7D.log`. Exit 1, `SUMMARY: FAIL (7/8 tests)`.
  The one failure is `Beatnik_Test_T2bOperators` (`check 1: expected 1, got
  4`): its deliberate `comm_size == 1` guard, pre-existing and recorded in
  `framework.md` T4c and the `grouped-io` log as making the tier red at four
  ranks by design. Every other member passed, `Beatnik_Test_MeshEdits` with
  `282/282` on all four ranks. Read per member, the passing direction.
- `f3iXR6G8cAF9`, `scripts/tuolumne/run_regression_minset.flux` with
  `BEATNIK_TEST_SCRATCH=/p/lustre5/stewartj/beatnik/gate_scratch`, log
  `beatnik_regression_minset.f3iXR6G8cAF9.log`. COMPLETED, 7.5 min,
  `[gate] PASS (label=regression)` over 60 launches, zero `[FAIL]` lines.
  Passing direction: the gate members' trajectories did not move.
- `f3iXs3sWtvv3`, `scripts/tuolumne/t4d6_gid_cadence.flux` (`pdebug`, 1 node
  `--exclusive`, `-t 60m`, `--bank=unm`), staged directory
  `/p/lustre5/stewartj/beatnik/T4d6j/stage/` (`COMMIT` = `092e2e5`, porcelain
  count 4: the user's `run_regression_minset.flux`, `commands.sh`,
  `milestone1.md` and the prompt file, none compiled), output
  `/p/lustre5/stewartj/beatnik/T4d6j/out/<group>_<BACKEND>_np<N>/` and
  `beatnik_t4d6.f3iXs3sWtvv3.log`. **Pending: read by `T4d6`.** Accepted and
  queued (`S`) at 11:17.

**Decisions** (given to the task and not reopened):
- **`--bank` directives (the user's ruling).** The user added
  `# flux: --bank=unm` to `scripts/tuolumne/run_regression_minset.flux`; that
  edit is theirs and is neither staged nor committed here. This task added the
  same single line to `scripts/tuolumne/unit_tests.flux`'s directive block and
  changed nothing else there. All four submissions went through
  `$TASKLOOP_SUBMIT`; the wrapper's two-job limit put the gate in after the
  first unit job finished.
- **The driver**, the replacement for `RefineSplitEdges` check 6 and the three
  job groups are as the entry states them (`## T4d6j open questions settled`).

**The cadence: `RemeshParams::renumber_gids_every = 50`** remesh calls, `0` (or
negative) = never, the `SolverParams` default and therefore what a milestone
run uses. It counts `DynamicRemesh::remesh` calls (a member counter,
`remeshCalls()`), identical on every rank. Why 50: one renumbering costs two
halo rebuilds plus a whole-AoSoA exchange, while every remesh call at the
milestone defaults already does a split, a collapse with its own compaction,
several flip rounds and a smoothing step, so one renumbering per 50 calls is
well under 1 % of remesh cost; much longer stops bounding the space within a
milestone run, and at `--remesh-every 1` a 600-step run renumbers 12 times,
with at least three renumberings in any 200-step window (T4d6's steps
400-600). Tessera's header suggests "every few hundred steps"; 50 is the
conservative end while the leak rate is unmeasured. The reasoning is in the
field's comment. The gate's dynamic-remesh member runs 20 remesh calls, so it
never renumbers.

**Re-basing:** renumbering does **not** call `resetReferenceState`. It moves no
vertex and changes no connectivity; it changes gids and the local storage
order. The AMR reference area and curvature are face user fields and travel
with their faces through the compaction, and no Beatnik state caches a local
index or a gid across a step (the valence flips' touched set lives inside one
cleanup pass, which runs after `remesh()` returns). The Solver's existing
`rebuildState()` after every remesh still runs.

**Signatures added:**

```
// src/Beatnik_MeshInterface.hpp, SurfaceMesh
MeshEditReport compactAndRenumberGids();   // Tessera::compactAndRenumberGids
long long gidSpace();                      // Tessera::detail::gidSpace, collective
// src/Beatnik_Params.hpp, RemeshParams
int renumber_gids_every = 50;              // 0 = never
// src/Beatnik_DynamicRemesh.hpp
RemeshDiagnostics::gid_space_high_water;   // GlobalIndex, before renumbering
RemeshDiagnostics::gids_renumbered;        // bool
RemeshDiagnostics::gid_space_after;        // GlobalIndex
long long DynamicRemesh::remeshCalls() const;
// src/Beatnik_Solver.hpp
long long Solver::edgeFlips() const;       // cumulative refine-branch flips
```

`gidSpace()` wraps `Tessera::detail::gidSpace`, the function `Tessera::compact`
reports with, so the high-water mark and `CompactStats` cannot disagree; it
copies the three AoSoAs to the host (O(local entities), once per remesh). The
renumbering runs as step 5 of `remesh()`, after every measurement. The
`edgeFlips()` accessor is additive and exists for check 6 (an implementation
choice inside **Fill in**; no other caller).

**The driver,** `Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP,OPENMP}`, three
positionals: `argv[1]` steps, `argv[2]` checkpoint cadence (`0` = none, not even
step 0), `argv[3]` renumber cadence (`0` = never). Every rank prints after every
step one line of `key=value` fields:

```
[m1run] step= time= rank= V= F= gid_space= gid_space_after= renumbered=
  gid_ratio= volume_drift= min_quality= splits= collapses= flips=
  smooth_steps= rss_kb= rss_peak_kb= elapsed=
```

(one line). `gid_space` is the step's high-water mark, `gid_ratio` is it over
`V`, `min_quality` is measured after the whole step, the four counters are
cumulative sums of the remesh pass's diagnostics (the cleanup's valence flips
are not in `flips`), RSS is this rank's `VmRSS`/`VmHWM`. A closing
`[m1run] FINAL rank= … wall= s_per_step= …` line per rank. It asserts only the
step budget. `Beatnik_Milestone1Params.hpp` holds
`Beatnik::Test::Milestone1::makeParams( checkpoint_dir )` (600 steps, every 10)
and `kSteps`, `kCheckpointEvery`, with a reference-line provenance comment on
every default (`run_adaptive_mesh_bubble.py` at the gold README's version).
The job's checkpoint cadence is 10 in every row, as a milestone run writes.

**Removes, confirmed gone:** the five rejections and their comments
(`grep` for `improveConnectivityByFlips`, `isotropicCleanup`,
`collapseShortEdges`, `tangentialSmooth`, `flipEdgesForQuality` in a `throw`
message in `src/` finds none); the four `checkRejected` cases (the T4e
proximity case stays); `checkFailureDirection` and its header description
(`grep checkFailureDirection` finds nothing).

**Only running revealed:**
- Check 6's cumulative flips are rank-count dependent: 58 at 16 launches and
  60 at 5 (passes 3 and 4 are `--max-faces`-capped, which the file already
  records as not rank-count invariant). The check asserts only `> 0`. The
  refine branch's re-base under `flips > 0 && smooth_iters == 0` (T4d3 review
  item 2) is now exercised by every gate launch of that check.
- The four-rank unit tier is red by `T2bOperators`'s design, as noted under
  Jobs; this is the first milestone-1 task to submit `unit_tests.flux` at four
  ranks.

**Cosmetic corrections** (prompt's list, all made): Status line; the Fill in
line references (`remesh()` `:1367-1448`, `RemeshDiagnostics` `:149-259`) and
the two files it omitted (the adapter and the `RemeshParams` field), added to
the entry; `compact()`'s doc comment now says the cadence calls
`compactAndRenumberGids()` and `compact()` is called only by tests, with the
matching Deliberate-deviations bullet; every "rejected until T4d6" comment in
`src/`, `tests/`, README Known Issues and `docs/testing.md` rewritten in the
present tense; T4d5 review nit 1 (`RefineSplitEdges.cpp` makeParams comment).
Current state's four "unreachable until T4d6j" bullets now say reachable.

**Session notes:** one `spack install`, rc 0, 12m28s, after unsetting
`HIPCC_*_FLAGS_APPEND`; log `/p/lustre5/stewartj/beatnik/T4d6j/
t4d6j_spack_install.log`, build output `t4d6j_build_out.txt` beside it (the
guard blocks writes under `~/spack_envs`). clang-tidy does not apply (spack
mode); clang-format was not run. The session was interrupted once while
polling the gate; the gate had completed and was read on resume.

**Affects:**
- **T4d6:** read `beatnik_t4d6.f3iXs3sWtvv3.log` and each `out/<tag>/launch.log`;
  the line format above is the contract. The cadence runs use 50, the control
  0. The 600-step estimate (900 s per launch) is untimed.
- **M1-T2j:** include `Beatnik_Milestone1Params.hpp` and call
  `Beatnik::Test::Milestone1::makeParams()`; it leaves
  `renumber_gids_every` at its default. T4d6's measured 600-step wall sets the
  walltime.
- **Follow-up:** `Beatnik_Test_T2bOperators` makes the unit tier red at four
  ranks; the "ranks 1 and 4" convention would read cleaner if its guard became
  a reduced skip (another task's test, not changed here).

## T4d6j review

**Verdict:** Passed
**Reviewed:** `092e2e5^..092e2e5` (the one code commit; `be10161` after it
touches only `tasks/`). It touches every file in the **Fill in** (the remesh
cycle and `RemeshDiagnostics` in `src/Beatnik_DynamicRemesh.hpp`, the adapter
and query in `src/Beatnik_MeshInterface.hpp`, `RemeshParams` in
`src/Beatnik_Params.hpp`, the five rejections in `src/Beatnik_Solver.hpp`, the
driver, its header and its `BEATNIK_DRIVER_SOURCES` entry, the job script,
`RefineSplitEdges` and `DynamicRemeshSplit`), plus the README and
`docs/testing.md` lines CLAUDE.md requires to stay in sync, the user-ruled
`--bank=unm` line in `scripts/tuolumne/unit_tests.flux`, the additive
`Solver::edgeFlips()` accessor check 6 needs, and comment-only rewrites in
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp` (the log's cosmetic corrections).
Nothing out of scope.
**Evidence:** a submit task (**Submits:** field), so verified without a
verification job. Job `f3iXs3sWtvv3` (`scripts/tuolumne/t4d6_gid_cadence.flux`)
is known to the scheduler, state `SCHED`, submitted 11:17:18; its result is
T4d6's. The stage `/p/lustre5/stewartj/beatnik/T4d6j/stage/` exists and its
`COMMIT` names `092e2e580a91…`, porcelain count 4 (none compiled, as the log
lists). The script takes its binaries only from the stage (`$1`, never
`beatnik_exe`), prints `stage/COMMIT` and each binary's `sha256sum` before any
launch, writes under `T4d6j/out/<tag>/`, and carries the deadline guard (rows
sum to 3300 s inside a 3420 s budget of the 60 m wall). The rest of the
criterion was read from the implementing session's jobs. They ran at
10:19–10:28, before the commit (11:17), so provenance rests on the installed
tree, not an echo: the dev view's `Beatnik_DynamicRemesh.hpp`,
`Beatnik_MeshInterface.hpp`, `Beatnik_Params.hpp` and `Beatnik_Solver.hpp` are
byte-identical to `092e2e5`'s, the gate log carries check 6's new
`flips liveness, pass N` notes (code that exists only in `092e2e5`), and the
staged `Beatnik_Test_Milestone1Run_MPI_HIP` is byte-identical to the view's
(mtime 10:18:58). Unit tier `f3iXQgeQwhKD` COMPLETED rc 0, no queue wait,
48.6 s; `f3iXQgvioo7D` rc 1, 1.06 min, the one failure the
pre-existing `T2bOperators` `comm_size == 1` guard, as recorded. Gate
`f3iXR6G8cAF9` COMPLETED rc 0, no queue wait, 7.51 min, ends `[gate] PASS
(label=regression)`, zero `[FAIL]` lines; its only remaining failure direction
in `DynamicRemeshSplit`, the T4e proximity rejection, is noted on every
launch, and check 6's flips go 0, 18, … to a positive cumulative count.
The build log (`/p/lustre5/stewartj/beatnik/T4d6j/t4d6j_spack_install.log`)
was not re-read; the **Met.** paragraph's warning count is the session's.

Mechanical pass (`review_checks.py`): S2 at `Beatnik_MeshInterface.hpp:451` is
a doc comment, not code; the S4 hits on `improveConnectivityByFlips` are the
live function (passed in error as an old name). No K/M/B candidates. Read:
- Collectives: `remesh()` step 5 runs `gidSpace()` (one `MPI_Allreduce`) on
  every call and `compactAndRenumberGids()` on `_remesh_calls % every == 0`, a
  counter identical on every rank; the driver's per-step measurements and its
  early-stop `break` follow `advanceOneStep`'s global result.
- State across renumbering: the solution lives in the Tessera vertex user
  pack and the AMR reference fields in face user fields, so they travel with
  the reorder; the `Solver`'s `rebuildState()` after every remesh still runs.
  No `resetReferenceState`, as the log argues.
- The driver's one kernel captures a View by value and is reduced into a host
  scalar; the cadence's "why 50" is documented on the field.
- Check 6 is the liveness check the **Removes** field asks for: four passes,
  Euler and `checkConformity` after each, `edgeFlips() > 0` on a global count.

Stale code: the five rejections, the four T4d `checkRejected` cases and
`checkFailureDirection` are gone (grep); the `refining` flag went with them.
`DynamicRemeshSplit`'s header and `RefineSplitEdges`' makeParams comment
(T4d5 review nit 1) are rewritten. No prompt file remains under `tasks/`.

1. [nit] `examples/02_adaptive_mesh_bubble/adaptive_mesh_bubble.cpp:39-47`
   and `examples/02_adaptive_mesh_bubble/CMakeLists.txt:19-26` still say the
   mesh-quality passes (T4d) are unported and that the DEFAULT configuration
   is rejected because `--isotropic-cleanup` defaults on. This diff deleted
   that rejection, so both now describe a failure the example no longer has
   (CLAUDE.md "Describe the code as it is now"). Rewrite both to name only
   T4e (off by default), the tight set and T5c.
2. [follow-up] `docs/dataflow.html:64-68` says the defaults do not run, that
   `--source-quadrature face` is the default (it is `vertex`,
   `Beatnik_Params.hpp:96`) and gives a production path with the four T4d
   knobs off. Stale before this task in part; now wholly.
3. [follow-up] `scripts/tuolumne/grouped_io_t1.flux:25-32` says the four
   remesh flags are T4d, unimplemented, and throw. Stale since T4d2–T4d5;
   the job's split-only flags are still a valid choice.

**Affects:** none.

## T4d6

**Commits:** the one commit that adds this section (the diag selector in
`scripts/tuolumne/t4d6_gid_cadence.flux`, the README Known Issues entry,
`milestone1.md`, this log). No code under `src/` or `tests/` changed.

**Jobs:**
- `f3iXs3sWtvv3` (submitted by T4d6j), `scripts/tuolumne/t4d6_gid_cadence.flux`
  (default `measure` selector), stage `/p/lustre5/stewartj/beatnik/T4d6j/stage`
  (`COMMIT` `092e2e580a91…`), output `beatnik_t4d6.f3iXs3sWtvv3.log` and
  `/p/lustre5/stewartj/beatnik/T4d6j/out/<group>_<BACKEND>_np<N>/launch.log`.
  COMPLETED rc 0, `launched=7 skipped=0`, total 66 s. Showed the cadence's
  bounded direction (ranks 1 and 4), the control's growing direction, the
  umbrella, the 1-rank RSS clause, and the 4-rank checkpoint-step RSS growth.
- `f3iYLanaSJHd`, the same script with `-- /p/lustre5/stewartj/beatnik/T4d6/stage diag`
  (one `diag` row then: 4 ranks, HIP, 600 steps, checkpoints 0, renumber 50),
  output `beatnik_t4d6.f3iYLanaSJHd.log` and
  `/p/lustre5/stewartj/beatnik/T4d6/diag_HIP_np4.f3iYLanaSJHd/launch.log`
  (moved aside before the next job reused `out/diag_HIP_np4`). COMPLETED rc 0,
  0 checkpoints, wall 17 s. Showed that 4-rank RSS still grows without
  checkpoints, which by the entry's text fails the RSS clause, so the session
  stopped and asked.
- `f3iYMv1J4fSw`, the same arguments after the user's ruling added a `diag0`
  row (renumber 0), output `beatnik_t4d6.f3iYMv1J4fSw.log` and
  `/p/lustre5/stewartj/beatnik/T4d6/out/{diag,diag0}_HIP_np4/launch.log`.
  COMPLETED rc 0, `launched=2 skipped=0`, walls 20 s (`diag`) and 16 s
  (`diag0`). Showed that the 4-rank no-checkpoint growth is the same with and
  without renumbering: not the gid space, not the cadence.

Stage `/p/lustre5/stewartj/beatnik/T4d6/stage/`: both
`Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP}` copied from the T4d6j stage
(sha256 `4556994…` HIP, `d279e9b…` SERIAL, identical to the source), `COMMIT`
naming `092e2e5` plus a line saying the binaries were copied, not rebuilt.

**Ratio series** (`gid_ratio` = gid-space high-water / global V; rank 0, the
value is global). Cadence np1 and np4 identical; `diag` identical to them;
`diag0` identical to the control:

```
cadence (renumber 50)                     control (renumber 0)
step  V   F    gid_space→after ratio      step  V   F    gid_space ratio
  1  462  920  3082            6.670996     1  462  920  3082      6.670996
  2  514 1024  3514            6.836576     2  514 1024  3514      6.836576
  3  518 1032  3522            6.799228     3  518 1032  3522      6.799228
 50  518 1032  3522→3098 (R)   6.799228   246  519 1034  3530      6.801541
 51                            5.980695   257  520 1036  3538      6.803846
100,150,200 R, no change       5.980695   398  521 1038  3546      6.806142
246  519 1034  3106            5.984586   516  522 1040  3554      6.808429
250  R 3106→3104; 251          5.980732   (constant between these steps)
257  520 1036  3112            5.984615
300  R 3112→3110; 301          5.980769
350  R, no change
398  521 1038  3118            5.984645
400  R 3118→3116; 401          5.980806
450,500 R, no change
516  522 1040  3124            5.984674
550  R 3124→3122; 551          5.980843
600  R, no change              5.980843
```

Window 400–600: cadence 5.984645 → 5.980843, two drops and one rise, so it
does not grow monotonically, at np1 and np4. Control 6.806142 → 6.808429, zero
drops and one rise (step 516).

**Per-rank peak RSS** (`/usr/bin/time -v` Maximum resident set size, kB, per rank):

| Launch | Peak RSS per rank | VmRSS step 1 → 600 (rank 0..3) |
| --- | --- | --- |
| cadence_HIP_np1 | 863476 | 862220 → 863476 |
| cadence_HIP_np4 | 1014924, 1023676, 1023656, 1022368 | +12.8, +21.7, +21.3, +20.4 MB |
| control_HIP_np1 | 863176 | 862184 → 863176 |
| umbrella_SERIAL_np1 | 704848 | (50 steps) +0.3 MB |
| umbrella_SERIAL_np4 | 709176, 709140, 708468, 708076 | +0.8–1.1 MB |
| umbrella_HIP_np1 | 863464 | +1.3 MB |
| umbrella_HIP_np4 | 1004144, 1004576, 1004100, 1004608 | +2.2–2.4 MB |
| diag_HIP_np4 (`f3iYLanaSJHd`) | 1003704, 1004352, 1006524, 1005076 | +6.4, +6.7, +9.2, +8.2 MB |
| diag_HIP_np4 (`f3iYMv1J4fSw`) | 1003752, 1004968, 1006884, 1004304 | +6.4, +7.5, +9.5, +7.4 MB |
| diag0_HIP_np4 (`f3iYMv1J4fSw`) | 1003840, 1005464, 1005200, 1003628 | +6.7, +8.0, +7.5, +6.4 MB |

RSS series, read from the `rss_kb=` fields:
- 1 rank (cadence): changes only at steps 1, 10 (F 920→1032 at step 3), 50,
  250 and 270: two 264 kB steps at constant face count (50, 270). The control
  changes at 1, 10 and 250 only. This counts as flat.
- 4 ranks with checkpoints (cadence): 51–60 changes per rank. 11.8 / 19.0 /
  18.8 / 17.7 MB arrive at constant F, 11.6 / 17.2 / 17.3 / 15.6 MB of it on
  checkpoint steps. Rank 1: 1001792 (1), 1003516 (10), 1007420 (200),
  1011404 (300), 1015896 (400), 1019464 (500), 1023540 (600).
- 4 ranks without checkpoints: about 264 kB increments, at constant F 4.1–7.2 MB
  (`diag`, first job), 4.9–7.7 MB (`diag`, second), 3.7–5.4 MB (`diag0`), mostly
  after step ~300 and often on the same steps on every rank (389, 413/414,
  437/438, 462, 486/487, 510/511, 535, 559/560, 583/584). None of those steps
  makes a remesh edit or changes the gid space. `diag0` rank 0: 997124 (1),
  998076 (3), 998248 (246), 999408 (389), 1000460 (398), 1002036 (511),
  1003064 (516), 1003840 (584).

**Umbrella** (all four launches, identical): cumulative at step 50
splits / collapses / flips / smooth_steps = **364 / 8 / 76 / 3**, all non-zero
(not M1-R7). Min quality over every step and rank 0.6939016 against
`--remesh-min-quality` 0.18 (`src/Beatnik_Params.hpp:655`). Max |volume drift|
3.62e-12 at step 1 and 3.1e-14 after it, below `1e-10`. The prompt's "at most
2.2e-16" understated the step-1 value; the clause holds either way.
`requireSupportedConfiguration` rejects nothing: every launch completed.
The 600-step runs' totals at step 600: 368 / 8 / 76 / 7.

**Measured 600-step walls** (launch wall from `LAUNCH_WALL`, including
startup; driver `wall=` in parentheses): cadence np1 13 s (5.9 s), cadence np4
17 s (9.7 s), control np1 10 s (7.0 s), diag np4 17 s and 20 s (7.8 s), diag0
np4 16 s. T4d6j's estimate was 900 s per launch.

**Decisions** (given to the task and not reopened):
- **"Grows monotonically" in the control clause means non-decreasing at every
  step with a net rise over steps 400–600.** The mesh is nearly static at the
  milestone defaults: the control holds 6.806142 from step 398 and steps to
  6.808429 at 516. That meets the clause.
- **The per-rank RSS clause is judged after a diagnostic launch.** The first
  diag (`f3iYLanaSJHd`, renumber 50, no checkpoints) still grew. The entry says
  that fails the clause, so the session stopped and asked. **User's ruling:**
  attribute first by adding a `diag0` row (4 ranks, HIP, 600 steps,
  checkpoints 0, renumber 0). If it grows the same way, rule the clause met with
  respect to the gid leak and record the 4-rank growth (checkpoint and
  non-checkpoint parts) as a pre-existing README Known Issue plus an Affects
  follow-up; if not, stop. It grew the same way (`f3iYMv1J4fSw`), so the clause is
  met with respect to the gid leak and the Known Issue is in README.

**Implementation choices** (inside **Fill in**): the selector is the script's
second positional, `measure` (default, the seven T4d6j rows, output
`T4d6j/out`) or `diag` (the `diag` and `diag0` rows, output `T4d6/out`); an
unknown value exits 1. The provenance block echoes the selector, the selected
rows and the arguments. Row estimates for `diag`/`diag0` are 300 s; `-t 60m`
unchanged. The diag group names begin with `diag` and the filter matches on
that prefix.

**Affects:**
- **M1-T2j:** a 600-step milestone-defaults launch takes 10–20 s of wall
  (5.9–9.7 s in the driver) at 1 or 4 ranks on HIP, against T4d6j's 900 s
  estimate; size its walltime from these numbers (it also writes the Python
  comparison, which this job did not).
- **M1-A1j:** per-rank peak RSS 0.86 GB (HIP np1), 1.00–1.02 GB (HIP np4), 0.71
  GB (SERIAL), with a 4-rank growth of 6–10 MB per 600 steps without
  checkpoints and 13–22 MB with checkpoints every 10; table above.
- **Follow-up:** the 4-rank per-rank RSS growth at constant face count
  (README Known Issues), pre-existing, independent of the gid cadence. About
  two thirds of it is on checkpoint steps and the rest is not. It is unmeasured
  whether it plateaus over a longer run. Not fixed here (checkpoint writer out of scope).

## T4d6 review

**Verdict:** Passed
**Reviewed:** `6d4896c^..6d4896c` (the one commit; `345c8c8` before it touches
only `tasks/milestone1.md`). Outside `tasks/` it touches
`scripts/tuolumne/t4d6_gid_cadence.flux` (the launch selector and the `diag`
rows) and `README.md` (the Known Issues entry), both in **Fill in**. No `src/`
or `tests/` change, so the code under judgment is T4d6j's `092e2e5`.
**Evidence:** a post-job task, verified from its jobs' output, read in full
and re-derived independently from every `launch.log`:
- `f3iXs3sWtvv3` (`t4d6_gid_cadence.flux`, `measure`), stage
  `/p/lustre5/stewartj/beatnik/T4d6j/stage`, echoed `COMMIT` `092e2e580a91…`
  with the staged binaries' sha256 (SERIAL `d279e9b…`). COMPLETED rc 0;
  submitted 11:17:18, started 11:38:22 (21 min queue wait), 1.55 min run.
  `launched=7 skipped=0`, every `FINAL` line `completed=` its budget.
- `f3iYLanaSJHd` (`diag`, one row) and `f3iYMv1J4fSw` (`diag` and `diag0`),
  stage `/p/lustre5/stewartj/beatnik/T4d6/stage`, `COMMIT` `092e2e5` with a
  "copied, not rebuilt" line; SERIAL sha256 identical to the T4d6j stage's.
  Both COMPLETED rc 0, no queue wait, 42 s and 1.01 min. The second job's
  provenance block echoes `selector: diag` and both rows, matching the
  committed script.

Re-derived from the logs (rank 0's `gid_ratio`, global):
- **Bounded direction.** Cadence np1, cadence np4 and both `diag` launches:
  steps 400–600 go 5.984645 → 5.980843, one rise and two drops. Not monotone.
- **Leak direction.** Control np1 and `diag0` np4: 6.806142 → 6.808429, one
  rise, zero drops; non-decreasing with a net rise, as the log's settled
  reading of "monotonically" requires.
- **Per-rank RSS.** 1 rank: +1.2 MB (cadence) and +1.0 MB (control) over 600
  steps. 4 ranks without checkpoints: +6.2–9.3 MB per rank, 3.6–7.5 MB at
  constant F, the same with renumbering every 50 and never; with checkpoints
  +12.5–21.2 MB (step-1 `VmRSS` to step-600; the log's table uses step-1 to
  peak, a few hundred kB higher). The clause is met with respect to the gid
  leak by the user's ruling recorded under **Decisions**, and the growth is
  in README Known Issues with a reproduction and on the **Affects** line.
- **Umbrella.** All four 50-step launches complete; cumulative splits /
  collapses / flips / smooth_steps 364 / 8 / 76 / 3 in each; minimum quality
  0.6939016 (floor 0.18); largest |volume drift| 3.62e-12, below `1e-10`.

Mechanical pass (`review_checks.py`): no candidates. `bash -n` on the script
is clean. The selector rejects an unknown value with exit 1 before any work,
`measure` keeps exactly T4d6j's seven rows and output root, and the diag rows
write under their own root, so the T4d6j output is not overwritten. No prompt
file remains under `tasks/`. **Removes** is nothing.

**Affects:** none.

## M1-T2j

**Commits:** `347c2ef` (the two `src/` files, the test, `tests/CMakeLists.txt`,
`scripts/tuolumne/m1_milestone_run.flux`, `docs/testing.md`, README), and the
commit that adds this section (`milestone1.md`, this log).

**Jobs:**
- `f3iYat4nhyXd`, `scripts/tuolumne/run_regression_minset.flux` with
  `BEATNIK_TEST_SCRATCH=/p/lustre5/stewartj/beatnik/gate_scratch`, log
  `beatnik_regression_minset.f3iYat4nhyXd.log`. COMPLETED rc 0, 7.9 min,
  `[gate] PASS (label=regression)` over 60 launches, zero `[FAIL]` lines.
  Passing direction: the two new `cleanupFlips` sums change no trajectory.
- `f3iYjQ4E4J8X`, `scripts/tuolumne/m1_milestone_run.flux -- /p/lustre5/stewartj/beatnik/M1-T2j/stage`
  (`pbatch`, 1 node `--exclusive`, `-t 12m`, `--bank=unm`, dev env), stage
  `/p/lustre5/stewartj/beatnik/M1-T2j/stage/` (`COMMIT` = `347c2ef…`, porcelain
  count 4: the user's `run_regression_minset.flux`, `commands.sh`,
  `milestone1.md` and the prompt file, none compiled; sha256 HIP `f9ba6852…`,
  SERIAL `1daa9626…`), output
  `/p/lustre5/stewartj/beatnik/M1-T2j/out/<tag>/` (`launch.log` plus the
  checkpoints) and `beatnik_m1_milestone.f3iYjQ4E4J8X.log` in the repo root.
  **Pending: read by `M1-T2`.** Accepted, state `S`, submitted 13:07.

**Decisions** (given to the task and not reopened):
- **The cleanup flip count is added in `src/`.** `MeshQuality::isotropicCleanup`
  returns `int`, the global `valenceEqualizingFlips` count (0 when disabled);
  `Solver` sums it into `long long _cleanup_flips` at both call sites (refine
  branch and remesh branch) and exposes `long long cleanupFlips() const`. The
  two `Beatnik_Test_MeshEdits.cpp` callers discard the result, unchanged. The
  README documents neither API; its T4d coverage sentence, which called this
  test "not yet written", now names the member and `cleanupFlips()`.
- **Queue `pbatch`**, as **Submits:** names. The wrapper's policy allows only
  `pdebug`, so the first submission exited 3; **the user approved** submitting
  this script unchanged to `pbatch`, once.

**Signatures changed:**

```
// src/Beatnik_MeshQuality.hpp
int MeshQuality::isotropicCleanup( mesh_type& mesh ) const;   // was void
// src/Beatnik_Solver.hpp
long long Solver::cleanupFlips() const;                        // new
```

**Walltime, and how it was sized** (cosmetic correction of **Submits:**, which
said `-t 90m`, "about 50 minutes" from M1-G2's Python wall). T4d6 measured a
600-step milestone-defaults launch at 10–20 s of wall on HIP at 1 or 4 ranks;
SERIAL 50 steps took 1.7 s of driver wall (0.034 s/step, so about 20 s for 600);
one `compare_output.py` call on the login node takes 0.51–0.53 s (M0-T3
measured 0.42 s on compute nodes), 61 per comparison launch. Per comparison
launch about 45–60 s, the job about 6 min. `-t 12m` is about 2x, not 1.5x,
because SERIAL at 600 steps and the comparator on a compute node are
unmeasured here; the deadline guard's row estimates (100 s per comparison
launch and for `fail-no-cleanup`, 40 s for `fail-step0`) sum to 540 s of a
600 s budget. **Submits:** now says so.

**Implementation choices** (inside **Fill in**):
- **Failure directions** are the test's optional third argument: `fail-step0`
  runs 10 steps and compares the step-10 checkpoint against the step-0 gold,
  passing iff the comparator exits exactly 1; `fail-no-cleanup` forces
  `cleanup.enabled = false`, compares each checkpoint until one mismatches,
  and passes iff a step exits 1 with no defect and `cleanupFlips() == 0`. The
  manifest line passes no third argument (default `compare`).
- **The `time` liveness bound is `|t/t_gold - 1| <= 1.0`** (`kTimeLivenessRtol`),
  a new tolerance in a new test. T4d6's own driver output has Beatnik 0.16
  ahead of the gold at step 10 and **0.34 at step 50** (Beatnik holds 1032
  faces where the reference holds 1080, so its minimum edge and adaptive dt
  differ); a fixed `--dt 0.003` is 1.72, 2.43 and 2.23 off at steps 10, 50 and
  600. 1.0 separates the two; 0.25 would have reported a mesh difference as a
  liveness failure. The tight `time` comparison is the comparator's, at the
  per-step table.
- **The tolerance table** is `toleranceForStep(step)`, every step
  `{1e-10, 1e-12}`; M1-A1j edits it.
- **The gold step/time/V/F table** (`kGold`, 61 rows) was read from the
  `.npz` files at 17 digits; each step cross-checks the comparator's gold
  counts against it.
- **Runner output tags:** `<target>_np<N>` for comparisons and
  `<target>_np<N>_<mode>` for failure directions, under `M1-T2j/out/` (the
  default is the stage's sibling `out/`). The two failure-direction
  `[milestone]` lines carry `mode=<mode>` before `in`.
- **Gold install:** `gold/*_step*.npz` (61 files, no `checkpoint_latest.npz`)
  and `milestone1-bubble/README.md` one level up, beside `gold/`, with its
  `FATAL_ERROR`.

**Output M1-T2 reads** (rank 0, every line one line in the output):

```
[m1traj] step= time= V= F= splits= collapses= flips= smooth_steps=
  cleanup_flips= elapsed=                                  (every step)
[m1step] step= verdict=<match|mismatch|DEFECT> exit= rtol= atol= time=
  gold_time= time_rel= V_cpp= F_cpp= V_gold= F_gold= ambiguous=<c>/<g>
  faces_identical=<k>/<F> max_e[time]= max_e[vertices]= max_e[potential]=
  max_e[remesh_material_position]= cmp_s=                  (every 10 steps)
[cmp <step>] <the comparator's full report for that step>
[m1fail] direction=a|b ...                                 (failure modes)
[m1final] mode= space= np= steps=<done>/<budget> wall= cmp_wall= run_wall=
  s_per_step= splits= collapses= flips= smooth_steps= cleanup_flips=
  compared= matched= mismatched= defects= first_failing_step=
  time_liveness_failures=
```

`V_cpp`…`F_gold` are the two files' counts from the comparator's `structure:`
line (`-1` if it printed none); a field the comparator did not reach (it stops
at a count mismatch) has no `max_e[...]`. The four remesh counters are
cumulative `RemeshDiagnostics` sums; `cleanup_flips` is `cleanupFlips()` and is
not in `flips`. The job prints `[milestone] PASS|FAIL <target> np=<N>
[mode=<mode>] in <S>s`, `LAUNCH_WALL` per launch, and `[milestone] SUMMARY:`.
A compare launch that FAILs because a step mismatched is a measurement; the
failure-direction launches are expected to PASS.

**Session notes:** one `spack install`, rc 0, 13m42s, `HIPCC_*_FLAGS_APPEND`
unset; log `/p/lustre5/stewartj/beatnik/M1-T2j/m1t2j_spack_install.log`, build
output `m1t2j_build_out.txt` beside it. clang-tidy does not apply (spack mode);
clang-format was not run.

**Affects:**
- **M1-T2:** read `beatnik_m1_milestone.f3iYjQ4E4J8X.log` and
  `M1-T2j/out/<tag>/launch.log`; the line format above is the contract. First
  failing step and field are on `[m1final]`/`[m1step]`; counts agreement is
  `V_cpp`/`V_gold`/`F_cpp`/`F_gold`; M1-R8's signature is `max_e[vertices]`
  against the other fields with `ambiguous`; wall time is `run_wall` and the
  face-count trajectory is the `[m1traj]` `F=` column. Liveness failures are
  `[FAIL]` lines naming a counter or `time`.
- **M1-A1j:** edits `toleranceForStep` (or replaces it with a table) and
  `kSteps` via `Beatnik_Milestone1Params.hpp`; `kTimeLivenessRtol` is 1.0.
