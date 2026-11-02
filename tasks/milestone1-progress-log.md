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
