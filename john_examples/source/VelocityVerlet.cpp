/*
  Copyright (C) 2020-2022
      Max Planck Institute for Polymer Research & JGU Mainz

  This file is part of ESPResSo++.

  ESPResSo++ is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo++ is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "../../john_examples/log_level.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <hpx/config.hpp>

#include "hpx4espp/include/logging.hpp"
#include "hpx4espp/include/errors.hpp"
#include "hpx4espp/utils/algorithms/for_loop.hpp"
#include "hpx4espp/utils/algorithms/transform_reduce.hpp"
#include "hpx4espp/utils/multithreading.hpp"
#include "hpx4espp/integrator/VelocityVerlet.hpp"

#include "vec/integrator/VelocityVerlet.hpp"
#include "iterator/CellListIterator.hpp"
#include "interaction/Interaction.hpp"
#include "System.hpp"
#include "storage/Storage.hpp"
#include "mpi.hpp"
#include "python.hpp"
#include <hpx/modules/collectives.hpp>

#include <iomanip>

#ifdef VTRACE
#include "vampirtrace/vt_user.h"
#else
#define VT_TRACER(name)
#endif

#define Stringize(L) #L
#define MakeString(M, L) M(L)
#define $Line MakeString(Stringize, __LINE__)
#define Locator __FILE__ "(" $Line ")"
//#include "../../john_examples/json_lib.hpp"
#include <fstream>
#include "../src/storage/FullNeighbourNodeGrid.hpp"
#include <hpx/future.hpp>
#include <hpx/modules/collectives.hpp>
#include "../../john_examples/global_data.hpp"
std::size_t iteration_number = 0;
//HPXThreadsafeJson sending_json;
static double max_reduce(espressopp::real maxSqDist)
{
    return hpx::collectives::all_reduce("MaxDistanceMoved", maxSqDist,
                                        boost::mpi::maximum<espressopp::real>())
        .get();
}

HPX_PLAIN_ACTION(max_reduce)

namespace espressopp
{
namespace hpx4espp
{
namespace integrator
{
using namespace interaction;
using namespace iterator;
using namespace esutil;

LOG4ESPP_LOGGER(VelocityVerlet::theLogger, "VelocityVerlet");

VelocityVerlet::VelocityVerlet(shared_ptr<SystemHPX> system,
                               shared_ptr<storage::StorageHPX> storageHPX)
    : baseClass(system, storageHPX)
{
    LOG4ESPP_INFO(theLogger, "construct VelocityVerlet");
    resortFlag = true;
    maxDist = 0.0;
    nResorts = 0;
}

void VelocityVerlet::run(int nsteps)
{
    HPX4ESPP_DEBUG_MSG("VelocityVerlet::run");
    hpx4espp::utils::runAsHPXThread([this, nsteps] { this->run_(nsteps); });
}
void VelocityVerlet::run_(int nsteps)
{
    nResorts = 0;
    real time;
    timeIntegrate.reset();
    resetTimers();
    System& system = getSystemRef();
    espressopp::storage::Storage& storage = *system.storage;
    real skinHalf = 0.5 * system.getSkin();

    // signal
    baseClass::runInit();

    // Before start make sure that particles are on the right processor





    if (resortFlag)
    {
        SPDLOG_TRACE("Start Resort");
        spdlog::stopwatch sw;

        const real time = timeIntegrate.getElapsedTime();

        LOG4ESPP_INFO(theLogger, "resort particles");
        storageHPX->decomposeHPX();  // has send operatons

        maxDist = 0.0;
        resortFlag = false;

        timeResort += timeIntegrate.getElapsedTime() - time;
        SPDLOG_TRACE("Resort took {}", sw);
    }
    {
        const real time = timeIntegrate.getElapsedTime();
        storageHPX->loadCells();
        timeOtherLoadCells += timeIntegrate.getElapsedTime() - time;
    }
    bool recalcForces = true;  // TODO: more intelligent
    if (recalcForces)
    {
        LOG4ESPP_INFO(theLogger, "recalc forces before starting main integration loop");
        SPDLOG_TRACE("Start recalc1");
        spdlog::stopwatch sw;
        // signal
        baseClass::recalc1();
        SPDLOG_TRACE("recalc1 took {}", sw);
        updateForces();
        SPDLOG_TRACE("Start recalc2");
        sw.reset();

        // signal
        baseClass::recalc2();
        SPDLOG_TRACE("recalc2 took {}", sw);
    }


    auto grid_size = system.storage->getInt3DCellGrid();
    John::FullNeighbourNodeGrid grid(grid_size, hpx::get_locality_id(), Real3D(1, 1, 1));
//    json histogram;
    for (int i = 1; i < nsteps +1 ; i++)
    {
        iteration_number = i;
        step = i;
        iteration_num+=1;

//        if(iteration_num%100==0){
//            for (size_t particle_list = 0; particle_list < storage.getRealCells().size();
//                 ++particle_list)
//            {
//                auto unflatten =
//                    John::FullNeighbourNodeGrid::calculate_grid_position(particle_list, grid_size);
//                auto particle_list_self = storage.getRealCells()[particle_list];
//                histogram[particle_list + 1] =
//                    json::array({{"particle_list", particle_list},
//                                 {"filling_factor", particle_list_self->particles.size()},
//                                 {"3d_grid_pos", unflatten[0], unflatten[1], unflatten[2]}});
//            }
//
//            std::ofstream file(spdlog::fmt_lib::format("{} iteration {} rank {}.json", "histogram",
//                                                       iteration_num, hpx::get_locality_id()),
//                               std::ios::trunc);
//
//            file << histogram;
//        }

        {
            const real time = timeIntegrate.getElapsedTime();
            const real maxSqDist = integrate1();
            // collective call to allreduce for dmax
            real maxAllSqDist = max_reduce(maxSqDist);
            spdlog::stopwatch sw_all_reduce;
            //            mpi::all_reduce(*system.comm, maxSqDist, maxAllSqDist,
            //            boost::mpi::maximum<real>());
            maxDist += std::sqrt(maxAllSqDist);
            spdlog::info("HPX took {} iteration {}", sw_all_reduce, i);
            timeInt1 += timeIntegrate.getElapsedTime() - time;
        }
        if(enable_resort){
            if (maxDist > skinHalf) resortFlag = true;
        }
        if (resortFlag)
        {
            spdlog::stopwatch resort_sw;

            spdlog::info("Start Resport {}",i);

            const real time = timeIntegrate.getElapsedTime();

            storageHPX->unloadCells();

            // storage.decompose();
            storageHPX->decomposeHPX();

            storageHPX->loadCells();

            maxDist = 0.0;
            resortFlag = false;
            nResorts++;
            SPDLOG_INFO("took {}", resort_sw);

            timeResort += timeIntegrate.getElapsedTime() - time;
        }

        // update forces
        {
            updateForces();
        }

        {
            const real time = timeIntegrate.getElapsedTime();
            // second-half integration
            integrate2();
            step++;
            timeInt2 += timeIntegrate.getElapsedTime() - time;
        }
    }
    {
        const real time = timeIntegrate.getElapsedTime();
        storageHPX->unloadCells();

        timeOtherUnloadCells += timeIntegrate.getElapsedTime() - time;
    }


    timeRun = timeIntegrate.getElapsedTime();
    timeLost = timeRun - (timeForceComp[0] + timeForceComp[1] + timeForceComp[2] + timeComm1 +
                          timeComm2 + timeInt1 + timeInt2 + timeResort);

//        std::ofstream file(
//            spdlog::fmt_lib::format("CommTime rank {}.json",hpx::get_locality_id()),
//            std::ios::trunc);
//
//        file << sending_json.data;
}
/*void VelocityVerlet::log_to_json(espressopp::storage::Storage& storage,
                                 const Int3D& grid_size,
                                 std::size_t i,std::string what_to_store) const
{
//    json particle_status;
//    for (size_t particle_list = 0; particle_list < storage.getRealCells().size(); ++particle_list)
//    {
//        auto unflatten =
//            John::FullNeighbourNodeGrid::calculate_grid_position(particle_list, grid_size);
//        for (size_t particle_num = 0;
//             particle_num < storage.getRealCells()[particle_list]->particles.size(); ++particle_num)
//        {
//            const auto& particle = storage.getRealCells()[particle_list]->particles[particle_num];
//            particle_status[i][particle.id()] = json::array(
//                {{"id", particle.id()},
//                 {"particle_list", particle_list},
//                 {"cell_grid_3d", unflatten[0], unflatten[1], unflatten[2]},
//                 {"velocity", particle.velocity()[0], particle.velocity()[1],
//                  particle.velocity()[2]},
//                 {"position", particle.position()[0], particle.position()[1],
//                  particle.position()[2]},
//                 {"force", particle.force()[0], particle.force()[1], particle.force()[2]}});
//        }
//    }
//
//    std::ofstream file(
//        spdlog::fmt_lib::format("{} iteration {} rank {}.json",what_to_store,i,hpx::get_locality_id()),
//        std::ios::trunc);
//
//    file <<particle_status;
}*/

real VelocityVerlet::integrate1()
{
    auto& vs = this->storageHPX->virtualStorage;
    auto f_integrate1_vs = [this, &vs](size_t const& ivs)
    {
        auto& particles = vs[ivs].particles;
        return vec::integrator::VelocityVerlet::integrate1(particles, dt);
    };

    /// workaround to const& requirement for argument to convert operation
    /// see: https://github.com/STEllAR-GROUP/hpx/issues/3651
    if (vsidx.size() != vs.size())
    {
        vsidx.resize(vs.size());
        for (size_t i = 0; i < vsidx.size(); i++) vsidx[i] = i;
    }

    return utils::parallelTransformReduce(
        vsidx.begin(), vsidx.end(), 0.0,
        [](real const& a, real const& b) { return std::max(a, b); }, f_integrate1_vs);
}

void VelocityVerlet::integrate2()
{
    auto& vs = this->storageHPX->virtualStorage;
    auto f_integrate2_vs = [this, &vs](size_t const& ivs)
    { vec::integrator::VelocityVerlet::integrate2(vs[ivs].particles, dt); };

    utils::parallelForLoop(size_t(0), vs.size(), f_integrate2_vs);
}

void VelocityVerlet::initForcesParray()
{
    real time = timeIntegrate.getElapsedTime();

    auto& vs = this->storageHPX->virtualStorage;
    auto f_initf_vs = [this, &vs](size_t const& ivs) { vs[ivs].particles.zeroForces(); };

    utils::parallelForLoop(size_t(0), vs.size(), f_initf_vs);

    timeOtherInitForcesParray += timeIntegrate.getElapsedTime() - time;
}

void VelocityVerlet::calcForces()
{
    initForcesParray();
    {
        // TODO: Might need to place interaction list in HPXRuntime
        System& sys = getSystemRef();
        const espressopp::interaction::InteractionList& srIL = sys.shortRangeInteractions;

        for (size_t i = 0; i < srIL.size(); i++)
        {
            LOG4ESPP_INFO(theLogger, "compute forces for srIL " << i << " of " << srIL.size());
            real time;
            time = timeIntegrate.getElapsedTime();
            srIL[i]->addForces();
            timeForceComp[i] += timeIntegrate.getElapsedTime() - time;
        }
        // aftCalcFLocal();
    }
}

void VelocityVerlet::updateForces()
{
    // Implement force update here
    // Initial implementation: blocking update following original

    real time;
    System& system = getSystemRef();
    espressopp::storage::Storage& storage = *system.storage;
    time = timeIntegrate.getElapsedTime();
    auto grid_size = system.storage->getInt3DCellGrid();

    John::FullNeighbourNodeGrid grid(grid_size, hpx::get_locality_id(), Real3D(1, 1, 1));
    //log_to_json(storage, grid_size, iteration_num,"before_update_ghosts");

    storageHPX->updateGhostsBlocking();
    //log_to_json(storage, grid_size, iteration_num,"after_update_ghosts");

    timeComm1 += timeIntegrate.getElapsedTime() - time;

    time = timeIntegrate.getElapsedTime();
    calcForces();
    //log_to_json(storage, grid_size, iteration_num,"before_collect_forces");
    // just does some math
    timeForce += timeIntegrate.getElapsedTime() - time;

    time = timeIntegrate.getElapsedTime();
    storageHPX->collectGhostForcesBlocking();  // Exchange ghost information
    //log_to_json(storage, grid_size, iteration_num,"after_collect_forces");

    timeComm2 += timeIntegrate.getElapsedTime() - time;

    time = timeIntegrate.getElapsedTime();
    baseClass::aftCalcF();
    timeOtherAftCalcF += timeIntegrate.getElapsedTime() - time;
}

void VelocityVerlet::initForcesPlist()
{
    // forces are initialized for real + ghost particles

    System& system = getSystemRef();
    CellList localCells = system.storage->getLocalCells();

    LOG4ESPP_INFO(theLogger, "init forces for real + ghost particles");

    for (CellListIterator cit(localCells); !cit.isDone(); ++cit)
    {
        cit->force() = 0.0;
        cit->drift() = 0.0;  // Can in principle be commented, when drift is not used.
    }
}

void VelocityVerlet::resetTimers()
{
    timeForce = 0.0;
    for (int i = 0; i < 100; i++) timeForceComp[i] = 0.0;
    timeComm1 = 0.0;
    timeComm2 = 0.0;
    timeInt1 = 0.0;
    timeInt2 = 0.0;
    timeResort = 0.0;

    //--------------------------------------------------------//
    timeOtherInitResort = 0.0;
    timeOtherInitForcesPlist = 0.0;
    timeOtherLoadCells = 0.0;
    timeOtherRecalcForces = 0.0;
    timeOtherInitForcesParray = 0.0;
    timeOtherUnloadCells = 0.0;
    timeOtherAftCalcF = 0.0;
    //--------------------------------------------------------//
}

using namespace boost::python;

static object wrapGetTimers(class VelocityVerlet* obj)
{
    real tms[10];
    obj->loadTimers(tms);
    return boost::python::make_tuple(tms[0], tms[1], tms[2], tms[3], tms[4], tms[5], tms[6], tms[7],
                                     tms[8], tms[9]);
}

void VelocityVerlet::loadTimers(real t[10])
{
    t[0] = timeRun;
    t[1] = timeForceComp[0];
    t[2] = timeForceComp[1];
    t[3] = timeForceComp[2];
    t[4] = timeComm1;
    t[5] = timeComm2;
    t[6] = timeInt1;
    t[7] = timeInt2;
    t[8] = timeResort;
    t[9] = timeLost;
}

int VelocityVerlet::getNumResorts() const { return nResorts; }

void VelocityVerlet::loadOtherTimers(real* t)
{
    t[0] = timeOtherInitResort;
    t[1] = timeOtherInitForcesPlist;
    t[2] = timeOtherLoadCells;
    t[3] = timeOtherRecalcForces;
    t[4] = timeOtherInitForcesParray;
    t[5] = timeOtherUnloadCells;
    t[6] = timeOtherAftCalcF;
}

static object wrapGetOtherTimers(class VelocityVerlet* obj)
{
    real tms[7];
    obj->loadOtherTimers(tms);
    return boost::python::make_tuple(tms[0], tms[1], tms[2], tms[3], tms[4], tms[5], tms[6]);
}

/****************************************************
** REGISTRATION WITH PYTHON
****************************************************/

void VelocityVerlet::registerPython()
{
    using namespace espressopp::python;

    // Note: use noncopyable and no_init for abstract classes
    class_<hpx4espp::integrator::VelocityVerlet,
           bases<espressopp::integrator::MDIntegrator, MDIntegratorHPX>, boost::noncopyable>(
        "hpx4espp_integrator_VelocityVerlet",
        init<shared_ptr<SystemHPX>, shared_ptr<storage::StorageHPX> >())
        .def("run", &hpx4espp::integrator::VelocityVerlet::run)
        .def("getTimers", &wrapGetTimers)
        .def("getOtherTimers", &wrapGetOtherTimers)
        .def("resetTimers", &VelocityVerlet::resetTimers)
        .def("getNumResorts", &VelocityVerlet::getNumResorts);
}

void VelocityVerlet::updateForcesBlock() {}
}  // namespace integrator
}  // namespace hpx4espp
}  // namespace espressopp
