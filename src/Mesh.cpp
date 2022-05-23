/****************************************************************************
 * Copyright (c) 2021 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Mesh.hpp>

namespace Beatnik
{
//---------------------------------------------------------------------------//
#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<Kokkos::Cuda, Kokkos::CudaSpace>;
#else
template class Mesh<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Beatnik
