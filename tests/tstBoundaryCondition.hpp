#ifndef _TSTBOUNDARYCONDITION_HPP_
#define _TSTBOUNDARYCONDITION_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryCondition.hpp>

#include <mpi.h>

#include "tstMesh.hpp"
#include "tstDriver.hpp"

namespace BeatnikTest
{

template <class T>
class BoundaryConditionTest : public MeshTest<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;

  protected:

    Beatnik::BoundaryCondition bc_;
    std::shared_ptr<local_grid_type> local_grid_;
    // _local_grid = MeshTest<T>::testMeshNonPeriodic_->localGrid();
    // Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<double, 2>> _local_grid;
    
    void SetUp() override
    {
        MeshTest<T>::SetUp();
        for (int i = 0; i < 6; i++)
        {
            bc_.bounding_box[i] = MeshTest<T>::globalBoundingBox_[i];
        }
        local_grid_ = MeshTest<T>::testMeshNonPeriodic_->localGrid();
        
        // bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};
    }

    void TearDown() override
    {
        MeshTest<T>::TearDown();
    }

};

} // end namespace BeatnikTest

#endif // _TSTBOUNDARYCONDITION_HPP_
