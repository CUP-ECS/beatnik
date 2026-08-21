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
 * @file Beatnik_Test_Milestone0FrozenL4.cpp
 * @brief **THE MILESTONE-0 TEST AT SUBDIVISION 4** — M0-A1's second member.
 *        The whole test body is `Beatnik_Test_Milestone0Frozen.cpp`; this file
 *        is the level and nothing else.
 *
 * WHY A SECOND SOURCE STEM AND NOT A SECOND ARGUMENT LIST. The milestone tier
 * keys its argument lists by source stem
 * ([tests/CMakeLists.txt](../CMakeLists.txt)), one per registered source, and
 * each member needs its OWN gold directory — `milestone0-sub4-2000-steps/gold`
 * here against `milestone0-sub3-2000-steps/gold` there. Two stems give two
 * honest `_beatnik_args_<stem>_abs` / `_rel` pairs and leave that registration
 * loop untouched; one stem with two argument lists would have to teach the loop
 * to carry more than one per source, and an argument list naming the wrong
 * level's gold set is exactly what its `FATAL_ERROR` guard exists to prevent.
 *
 * Every per-level literal — the entity counts, the two carried scalars, the
 * polyhedral deficit, the final `time` and the 81-entry reference volume-drift
 * series — is selected by `BEATNIK_M0_LEVEL` inside the included file and was
 * re-derived from THIS level's gold set. Nothing is transferred from level 3.
 *
 * Level 4 is the second member rather than the primary one for two measured
 * reasons (M0-A1 decision 2): it diverges *earlier* at the tight `1e-12` rung
 * (step 775 against level 3's 1325), and its four tier launches are
 * `1722.066143` s of solve against level 3's `166.842830` s. It is a member at
 * all — rather than a recorded measurement — because it is the level that
 * resolves `--eps 0.025` (initial minimum edge `1.729575e-02` against
 * `3.457079e-02`) and the healthier frozen mesh, and leaving the trustworthy
 * resolution regime asserted by nothing was the wrong way round.
 *
 * Arguments, the scratch convention and `BEATNIK_PYTHON` are all as documented
 * in the included file's header.
 */

#define BEATNIK_M0_LEVEL 4
#include "Beatnik_Test_Milestone0Frozen.cpp"
