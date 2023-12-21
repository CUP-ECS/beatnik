/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Thomas Hines <thomas-hines-01@utc.edu>
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Supporting functions for Z-Model calculations, primarily Simple differential 
 * and other mathematical MigratorUtilities but also some utility functions that 
 * we may want to later contribute back to Cabana_Grid or other supporting libraries.
 */

#ifndef BEATNIK_MIGRATOR_UTILITIES_HPP
#define BEATNIK_MIGRATOR_UTILITIES_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Migrator.hpp>

#include <memory>

namespace Beatnik
{

namespace MigratorUtilities
{
    void runParticleMigrate()
    {
        Migrator::migrateParticles();
    }
    
}; // namespace MigratorUtilities

}; // namespace beatnik

#endif // BEATNIK_MIGRATOR_UTILITIES_HPP
