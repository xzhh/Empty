//#include <iostream>
//#include "ut.hpp"
//
//#include "System.hpp"
//#include "mpi.hpp"
//#include "esutil/RNG.hpp"
//#include "storage/DomainDecomposition.hpp"
//#include "interaction/LennardJones.hpp"
//#include "iterator/CellListIterator.hpp"
//#include "bc/SlabBC.hpp"
//#include "VerletList.hpp"
//#include "Real3D.hpp"
//#include "decomp.hpp"
//#include "main/espressopp_common.hpp"
//
//int main()
//{
//    //########################################################################
//    //# 1. specification of the main simulation parameters                   #
//    //########################################################################
//    auto Npart = 32768;
//    auto rho = 0.8442;
//    auto L = pow(Npart / rho, 1.0 / 3.0);
//    auto box =espressopp::Real3D(L, L, L);
//    auto r_cutoff = 2.5;
//    auto skin = 0.4;
//    auto temperature = 1.0;
//    auto dt = 0.005;
//    auto epsilon = 1.0;
//    auto sigma = 1.0;
//    auto warmup_cutoff = pow(2.0, 1.0 / 6.0);
//    auto warmup_nloops = 100;
//    auto warmup_isteps = 200;
//    auto total_warmup_steps = warmup_nloops * warmup_isteps;
//    auto epsilon_start = 0.1;
//    auto epsilon_end = 1.0;
//    auto epsilon_delta = (epsilon_end - epsilon_start) / warmup_nloops;
//    auto capradius = 0.6;
//
//    auto equil_nloops = 100;
//
//    auto equil_isteps = 100;
//    initMPIEnv();
//
//    //#######################################################################
//    //# 2. setup of the system, random number geneartor and parallelisation  #
//    //########################################################################
//
//
//   auto sys = std::make_shared< espressopp::System>();
//
//    sys->rng = std::make_shared<espressopp::esutil::RNG>(espressopp::esutil::RNG());
//
//    sys->bc = std::make_shared<espressopp::bc::SlabBC>(sys->rng,box);
//
//    sys->setSkin(skin);
//
//
//    nodeGrid(mpiWorld->size(),box,r_cutoff,skin,1,0,1.1);
//
//
//    espressopp::Int3D cellgrid(0,0,0); // TODO Implement
////
////    sys->storage = std::make_shared<espressopp::storage::DomainDecomposition>(system, nodeGrid, cellGrid);
////    //########################################################################
////    //# 3. adding the particles                                              #
////    //########################################################################
////
////    std::cout <<"adding {} particles to the system ...\n";
////    particle_list = [(pid, sys->bc.getRandomPos()) for pid in range(int(Npart))];
////    sys->storage->addParticle(); // TODO In loop
////    sys->storage->decompose();
////
////    //########################################################################
////    //# 4. setting up interaction potential for the equilibration            #
////    //########################################################################
////
////    auto verletlist  = espressopp::VerletList(sys,r_cutoff, true);
////
//
//    finalizeMPIEnv();
//    return 0;
//}
