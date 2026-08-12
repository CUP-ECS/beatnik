# Maintaining the build/run/test framework

Read this when adding a system, editing the resolver or a profile/runtime-env
file, changing the gate, or writing a `systems/<system>/claude.md`. These
meta-rules keep the framework's pieces consistent; breaking one does not fail
loudly — it just makes the framework quietly wrong later.

The rules that apply to ordinary code changes every session are in
[CLAUDE.md](../CLAUDE.md); the mechanism they govern is described in
[docs/environment-and-build.md](environment-and-build.md).

## Required sections in every `systems/<system>/claude.md`

1. **Environment** — the env path(s) *available on this machine*, plus any
   pre-activation step (e.g. `module load`). Where both a dev and a prod env
   exist, record both and note that new batch scripts must ask the user which one
   to target. Which env a given checkout is developed into is per-instance and
   belongs in that checkout's `profile.local.sh`.
2. **Build-config args** — system-specific args passed to `cmake` (or to the
   spack spec).
3. **Build command** — **one entry per build mode**, since the mode is a property
   of the checkout, not the machine: `spack install` for spack mode,
   `cmake --build` for manual mode. Describe both even if one is rare here, and
   scope any prohibition to a mode rather than to the machine or the repository.
4. **Run command for binaries** — the scheduler launch template, with the binary
   resolved via `beatnik_exe`.
5. **Job-scheduler batch template** — a concrete template under
   `scripts/<system>/`, sourcing the resolver first and branching on
   `BEATNIK_BIN_MODE` where behavior differs.
6. **Running non-test binaries** — when asked to run an `examples/` problem, ask
   the user for the example name and its arguments, then plug them into
   sections 4–5.
7. **Backends** — which compute backends build and run here, and which the gate
   runs here.

The tests themselves are project-wide and live in
[CLAUDE.md](../CLAUDE.md#minimum-test-set) and
[docs/testing.md](testing.md), not in a per-system doc. A system doc only says
*how* to run a test on that machine.

**Fallback when a system doc is missing or incomplete:** if the hostname matches
no row of CLAUDE.md's detection table, or the matching doc is missing one of the
sections above, **stop and ask the user** to fill the gap; do not guess.

## Adding a system

Do all of it in one change: create `systems/<system>/claude.md` with all seven
required sections; add a row to the CLAUDE.md "System detection" hostname table;
add a matching `case` branch in
[scripts/lib/beatnik_env.sh](../scripts/lib/beatnik_env.sh); add
`scripts/<system>/profile.defaults.sh` (and `runtime_env.sh` if the system needs
launch-time exports); add the gate wrapper
`scripts/<system>/run_regression_minset.<scheduler>`; declare the system's
backends and which of them the gate runs there; and commit an env snapshot under
`systems/<system>/`.

## Meta-rules

- **Never record a per-instance fact in a committed file.** A committed file
  asserts something about *every* clone. This checkout's build mode, the specific
  environment it is developed into, and its build directory go in the gitignored
  `scripts/<system>/profile.local.sh` — not in `profile.defaults.sh`, not in
  `systems/<system>/claude.md`, and not in CLAUDE.md. When adding guidance, ask
  which of the [three scopes](environment-and-build.md#three-scopes--put-each-fact-in-the-right-one)
  it belongs to; if the answer is "it depends on the checkout", it is
  per-instance. Corollary: policy statements in committed files must be scoped to
  the **active mode** (`BEATNIK_BUILD_MODE`) rather than written as blanket
  repository rules, because a blanket rule is a per-instance fact in disguise.
- **Env snapshots stay in sync.** Each `systems/<system>/` holds a committed
  snapshot of that system's spack environment (`spack.yaml`, and
  `spack-production.yaml` where a production env exists). Changing a live
  environment means updating `systems/<system>/spack*.yaml` in the same change. A
  stale snapshot is worse than none, because it looks authoritative.
- **The hostname table is mirrored in code.** CLAUDE.md's "System detection"
  table and the `case` statement in `scripts/lib/beatnik_env.sh` must always
  agree.
- **No inline runtime-env in batch scripts.** Source the resolver; edit
  `scripts/<system>/runtime_env.sh` when a launch-time variable changes.
- **The gate definition is single-sourced.** The tier label, backend(s) and rank
  set must read identically in CLAUDE.md, in
  [tests/CMakeLists.txt](../tests/CMakeLists.txt), in the
  `run_regression_minset.*` wrapper(s), in each system doc's Backends section,
  and in CI if one is ever added. Changing it is deliberate and confirmed with
  the user — never a side effect.
- **Example argument changes mirror into README.**
- **New files carry the license/SPDX header.**
- **Keep the two directory trees apart.** `systems/<system>/` holds
  hostname-keyed machine instructions and env snapshots and nothing else;
  `docs/` holds project documentation and framework reference, and no
  hostname-keyed machine instructions.

## Task logs

Ongoing multi-phase problems live in `tasks/`, one file per topic, each recording
*why* a problem is being worked and *how* it is being attacked, with a dated
progress log so a later session can resume. **At the start of a session that
touches one of these topics, Read `tasks/<topic>.md` first**, and append progress
to it as work lands — not at the end, when the reasoning has been forgotten.
`tasks/TEMPLATE.md` is the starting point for a new topic.

**Checkpoint commits.** When planning a large change, put explicit checkpoints in
the task log where progress should be committed, so a later failure can roll back
to the nearest one instead of unwinding everything.
