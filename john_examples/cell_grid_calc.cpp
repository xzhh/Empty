//
// Created by jn98zk on 6/25/22.
//
#include "decomp.hpp"

int main(int argc, char* argv[]){
    using namespace espressopp;
    initMPIEnv();

    Real3D boxL(1.0, 2.0, 3.0);
    std::shared_ptr<System> system;
    system = std::make_shared<System>();
    system->rng = std::make_shared<esutil::RNG>();
    system->bc = std::make_shared<bc::SlabBC>(system->rng, boxL);
    auto halfCellInt = 1;


    Int3D nodeGrid(2, 2, 1);
    Int3D cellGrid(26, 26, 26);

    espressopp::storage::DomainDecomposition domdec(system, nodeGrid, cellGrid, halfCellInt);

}