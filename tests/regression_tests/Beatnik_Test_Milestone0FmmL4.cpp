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
 * @file Beatnik_Test_Milestone0FmmL4.cpp
 * @brief **THE FMM MILESTONE MEMBER AT SUBDIVISION 4** — T6's second member,
 *        and the one the far-field accuracy claim actually rests on. The whole
 *        test body is `Beatnik_Test_Milestone0Fmm.cpp`; this file is the level
 *        and nothing else.
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
 * polyhedral deficit, the final `time`, the 81-entry reference volume-drift
 * series, **the realized P2P pair fraction and the divergence-horizon
 * envelope** — is selected by `BEATNIK_M0_FMM_LEVEL` inside the included file
 * and is this level's. Nothing is transferred from level 3.
 *
 * **THIS IS WHERE THE FAR-FIELD ACCURACY CLAIM LIVES**, and that is the whole
 * reason the pair is not symmetric. At 2562 vertices and `ncrit = 8` T5
 * measured **74.6% of pairs going through M2L** (`p2p_pair_fraction` 0.253650)
 * and `tau_A = 5.007746e-04` on the velocity; at 642 the M2L share is **14.5%**
 * and level 3's claim A is mostly a P2P comparison. Level 3 is still a member,
 * because it is the round trip, the tag handshake and the contraction under
 * test at a decomposition this level never visits — but a reader looking for
 * the far-field bound should be looking here.
 *
 * It is also the expensive member by a wide margin: one 2000-step FMM-driven
 * level-4 trajectory is **2377 s** at HIP np1 and **1473 s** at HIP np4 (T5),
 * against level 3's 256 s at HIP np1, and this member runs one of those on top
 * of a full direct trajectory. That is what moved the tier runner to
 * `-q pbatch`.
 *
 * Arguments, the scratch convention and `BEATNIK_PYTHON` are all as documented
 * in the included file's header.
 */

#define BEATNIK_M0_FMM_LEVEL 4
#include "Beatnik_Test_Milestone0Fmm.cpp"
