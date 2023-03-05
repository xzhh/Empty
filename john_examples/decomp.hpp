//
// Created by jn98zk on 5/5/22.
//
#include "log_level.hpp"

#ifndef ESPRESSOPP_DECOMP_HPP
#define ESPRESSOPP_DECOMP_HPP


#include <hpx/hpx_init.hpp>
#include "hpx4espp/SystemHPX.hpp"
#include "Real3D.hpp"
#include "Int3D.hpp"
#include <optional>
#include <random>
#include <fmt/printf.h>
#include "esutil/RNG.hpp"
#include "bc/SlabBC.hpp"
#include "hpx4espp/storage/DomainDecomposition.hpp"
#include "hpx4espp/integrator/VelocityVerlet.hpp"
#include "analysis/NPart.hpp"
#include "analysis/Temperature.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/PressureTensor.hpp"
#include "main/espressopp_common.hpp"
#include "hpx4espp/VerletList.hpp"
#include "hpx4espp/interaction/LennardJones.hpp"
#include "hpx4espp/interaction/VerletListLennardJones.hpp"
#include "hpx4espp/HPXRuntime.hpp"
#include <fmt/format.h>
#include "hpx4espp/esutil/RNGThread.hpp"
#include "hpx4espp/integrator/LangevinThermostat.hpp"
#include <spdlog/spdlog.h>
#include "hpx4espp/storage/JohnDomainDecomposition.hpp"
auto qbicity(const espressopp::Real3D &box_size, double rc, double skin, double cellDomTol = 2.0) {
    auto rc_skin = rc + skin;
    std::array<double, 3> box_aux = {box_size[0], box_size[1], box_size[2]};
    auto indMax = (int) (std::max_element(box_aux.begin(), box_aux.end()) - box_aux.begin());
    auto indMin = (int) (std::max_element(box_aux.begin(), box_aux.end()) - box_aux.begin());
    return ((box_size[indMax] - box_size[indMin]) < (cellDomTol * rc_skin));

}


auto nodeGridSimple(size_t n) {
    auto ijkmax = 3 * n * n + 1;
    auto d1 = 1;
    auto d2 = 1;
    auto d3 = 1;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k <= n; ++k) {
                if ((i * j * k == n) && (i * i + j * j + k * k < ijkmax)) {
                    d1 = k;
                    d2 = j;
                    d3 = i;
                    ijkmax = i * i + j * j + k * k;
                }
            }
        }
    }

    return espressopp::Int3D(d1, d2, d3);
}

struct CubicDataRep {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    double Lx;
    double Ly;
    double Lz;

    std::array<double, 3> make_box() const {
        return {Lx, Ly, Lz};
    }


};

auto replicate(const CubicDataRep &rep, int xdim, int ydim, int zdim) {
    CubicDataRep data_copy(rep);
    for (int i = 0; i < xdim; ++i) {
        for (int j = 0; j < ydim; ++j) {
            for (int k = 0; k < zdim; ++k) {
                if (i + j + k != 0) {
                    for (size_t index = 0; index < rep.x.size(); ++index) {
                        auto _x = rep.x[index];
                        auto _y = rep.y[index];
                        auto _z = rep.z[index];
                        data_copy.x.emplace_back(_x + i * rep.Lx);
                        data_copy.y.emplace_back(_y + j * rep.Ly);
                        data_copy.z.emplace_back(_z + k * rep.Lz);

                    }
                }
            }
        }
    }
    // Skip bonds and angles as we are not using them in the calculation
    data_copy.Lx = xdim * rep.Lx;
    data_copy.Ly = ydim * rep.Ly;
    data_copy.Lz = zdim * rep.Lz;

    return data_copy;
}


auto createCubic(int N, double rho, bool perfect) {

    std::vector<double> cubes(100, 0);
    for (int i = 0; i < cubes.size(); ++i) {
        cubes[i] = i * i * i;
    }
    double third = 1.0 / 3.0;
    auto L = std::pow(N / rho, third);
    int a = (int) std::pow(N, third);
    if (std::pow(a, 3) < N) {
        ++a;
    }
    auto lattice_spacing = L / a;

    auto magn = perfect ? 0.0 : lattice_spacing / 10.0;
    auto ct = 0;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist6(0, 1); // distribution in range [1, 6]

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < a; ++j) {
            for (int k = 0; k < a; ++k) {
                if (ct < N) {
                    x.emplace_back(0.5 * lattice_spacing + i * lattice_spacing + (magn * (2 * dist6(rng) - 1)));
                    y.emplace_back(0.5 * lattice_spacing + j * lattice_spacing + (magn * (2 * dist6(rng) - 1)));
                    z.emplace_back(0.5 * lattice_spacing + k * lattice_spacing + (magn * (2 * dist6(rng) - 1)));
                    ++ct;
                }

            }
        }
    }

    return CubicDataRep{x, y, z, L, L, L};
}


auto cellGrid_func(std::array<double, 3> rep, espressopp::Int3D node_grid, double rc, double skin, int halfCellInt) {
    auto rc_skin = rc + skin; // minimum size of cell
    int ix = (int) ((rep[0] * halfCellInt) / (rc_skin * node_grid[0]));
    int iy = (int) ((rep[1] * halfCellInt) / (rc_skin * node_grid[1]));
    int iz = (int) ((rep[2] * halfCellInt) / (rc_skin * node_grid[2]));
    if (ix < 1 or iy < 1 or iz < 1) {
        throw std::runtime_error(
                "local box size in direction 0 (=%6f) is smaller than interaction range (cutoff + skin = %6f)");
    }
    return espressopp::Int3D(ix, iy, iz);

}

espressopp::Int3D nodeGridMultiple(int n, espressopp::Int3D const &c) {
    int ijkmax = 3 * n * n + 1;
    int d1 = 1;
    int d2 = 1;
    int d3 = 1;
    for (int i = 1; i < n + 1; i++) {
        for (int j = i; j < n + 1; j++) {
            for (int k = j; k < n + 1; k++) {
                const int ijk = i * i + j * j + k * k;
                const int rem = c[0] % i + c[1] % j + c[2] % k;
                if ((i * j * k == n) && (ijk < ijkmax) && rem == 0) {
                    d1 = i;
                    d2 = j;
                    d3 = k;
                    ijkmax = ijk;
                }
            }
        }
    }

    const auto n_res = d1 * d2 * d3;

    return {d1, d2, d3};
}


auto make_hpx_system(const std::array<double, 3> &box, int numSubs, int numCommSubs, double rc, double skin, double dt,
                     int halfCellInt, double temperature, int threads) {
    auto system = std::make_shared<espressopp::hpx4espp::SystemHPX>();
    auto rng = espressopp::esutil::RNG();
    system->rng = std::make_shared<espressopp::esutil::RNG>();
    system->bc = std::make_shared<espressopp::bc::SlabBC>(system->rng,
                                                                  espressopp::Real3D(box[0], box[1], box[2]));
    system->setSkin(skin);
    auto nodeGrid = nodeGridSimple(mpiWorld->size());
    auto cellGrid = cellGrid_func(box, nodeGrid, rc, skin, halfCellInt);
    auto commAsync = false;
    auto excgAligned = false;
    auto commUseChannels = false;
    auto decompUseParFor = true;

    spdlog::info("commAsync\n""excgAligned\n""commUseChannels\n""decompUseParFor", commAsync, excgAligned,
                 commUseChannels, decompUseParFor);


    auto HPX4ESPP_OPTIMIZE_COMM = 1;
    std::array<double, 3> subSize = {(box[0] / nodeGrid[0]), (box[1] / nodeGrid[1]),
                                     (box[1] / nodeGrid[2])};
    auto subNodeGrid = nodeGridMultiple(numSubs, cellGrid);
    auto subCellGrid = cellGrid_func(subSize, subNodeGrid, rc, skin, 1);
    std::array<int, 3> expCellGrid = {subCellGrid[0] * subNodeGrid[0], subCellGrid[1] * subNodeGrid[1],
                                      subCellGrid[2] * subNodeGrid[2]};
    const auto storage = std::make_shared<espressopp::hpx4espp::storage::FullDomainDecomposition>(
            system, nodeGrid, cellGrid,
            halfCellInt, subCellGrid,
            numCommSubs, commAsync,
            excgAligned,
            commUseChannels,
            decompUseParFor,
            HPX4ESPP_OPTIMIZE_COMM);
    system->storage = storage;

    auto integrator = std::make_shared<espressopp::hpx4espp::integrator::VelocityVerlet>(system,
                                                                                         storage);
    integrator->setTimeStep(dt);
    if (temperature > 0) {

        system->rngThread = std::make_shared<espressopp::hpx4espp::esutil::RNGThread>(threads);
        auto thermostat = std::make_shared<espressopp::hpx4espp::integrator::LangevinThermostat>(system, storage);
        thermostat->setGamma(1.0);
        thermostat->setTemperature(temperature);
        thermostat->setIntegrator(integrator);
        thermostat->connect();
        integrator->addExtension(thermostat);
    }
    std::cout<<"System online" << std::endl;
    return std::make_tuple(system, integrator, storage);
}


std::stringstream info(const std::shared_ptr<espressopp::hpx4espp::SystemHPX> &system,
                       const std::shared_ptr<espressopp::hpx4espp::integrator::VelocityVerlet> &integrator,
                       bool per_atom) {
    auto NPart = espressopp::analysis::NPart(system).compute_real();
    auto T = espressopp::analysis::Temperature(system).compute_real();
    auto P_t = espressopp::analysis::Pressure(system);
    auto P = P_t.compute();
    auto Pij = espressopp::analysis::PressureTensor(system).computeRaw();
    auto step = integrator->getStep();
    auto Ek = (3.0 / 2.0) * NPart * T;
    auto Etotal = 0.0;
    std::string tot;
    std::stringstream os;
    if (per_atom) {
        tot = fmt::sprintf("%5d %10.4f %10.6f %10.6f %12.8f", step, T, P, Pij[3], Ek / NPart);
    } else {
        tot = fmt::sprintf("%5d %10.4f %10.6f %10.6f %12.3f", step, T, P, Pij[3], Ek);

    }
    std::string tt;
    for (int k = 0; k < system->getNumberOfInteractions(); ++k) {
        auto e = system->getInteraction(k)->computeEnergy();
        Etotal += e;
        if (per_atom) {
            tot += fmt::sprintf("%12.8f", (e / NPart));
            tt += fmt::sprintf("     e%i/N    ", k);
        } else {
            tot += fmt::sprintf("%12.3f", e);
            tt += fmt::sprintf("     e%i      ", k);
        }
    }

    if (per_atom) {
        tot += fmt::sprintf(" %12.8f", (Etotal / NPart + Ek / NPart));
        tt += "   etotal/N  ";

    } else {
        tot += fmt::sprintf(" %12.3f", Etotal + Ek);
        tt += "    etotal   ";
    }

    tot += fmt::sprintf("%12.8f\n", system->bc->getBoxL()[0]);
    tt += "    boxL     \n";
    if (step == 0) {
        os << " step      T          P        Pxy         ekin/N  " << tt;
    } else {
        os << " step      T          P        Pxy         ekin    " << tt;
    }

    os << tot;
    return os;
}

std::stringstream final_info(const std::shared_ptr<espressopp::hpx4espp::SystemHPX> &system,
                             const std::shared_ptr<espressopp::hpx4espp::integrator::VelocityVerlet> &integrator,
                             const std::shared_ptr<espressopp::hpx4espp::VerletList> &vl) {
    std::stringstream stream;
    auto NPart = espressopp::analysis::NPart(system).compute_real();
    stream << fmt::format("Total # of neighbors = {}\n", vl->totalSize());
    stream << fmt::format("Ave neighs/atom = {}\n", (double) vl->totalSize() / double(NPart));
    stream << fmt::format("Neighbor list builds = {}\n", vl->getBuilds());
    stream << fmt::format("Integration steps = {}\n", integrator->getStep());
    return stream;
}


#endif  // ESPRESSOPP_DECOMP_HPP
