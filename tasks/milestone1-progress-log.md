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
