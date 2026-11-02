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
