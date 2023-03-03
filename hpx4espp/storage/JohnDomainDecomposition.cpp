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

#include "hpx4espp/HPXRuntime.hpp"
#include "hpx4espp/storage/JohnDomainDecomposition.hpp"
#include "hpx4espp/utils/CuboidStructureCalc.hpp"
#include "hpx4espp/include/logging.hpp"
#include "hpx4espp/include/errors.hpp"
#include "hpx4espp/utils/algorithms/for_loop.hpp"
#include "hpx4espp/utils/multithreading.hpp"
#include "hpx4espp/utils/assert.hpp"
#include "hpx4espp/utils/assert_msg.hpp"
#include <hpx/format.hpp>
#include "../../john_examples/global_data.hpp"
#include <fstream>
#include "bc/BC.hpp"

#include <boost/python/numpy.hpp>

using hpx::execution::par;
using hpx::experimental::for_loop;
#include "integrator/MDIntegrator.hpp"

namespace espressopp::hpx4espp::storage
{
LOG4ESPP_LOGGER(FullDomainDecomposition::logger, "FullDomainDecomposition");

const Real3D FullDomainDecomposition::SHIFT_ZERO{0., 0., 0.};

FullDomainDecomposition::FullDomainDecomposition(shared_ptr<System> system,
                                                 const Int3D& _nodeGrid,
                                                 const Int3D& _cellGrid,
                                                 int _halfCellInt,
                                                 const Int3D& _subCellGrid,
                                                 int _numCommSubs,
                                                 bool _commAsync,
                                                 bool _excgAligned,
                                                 bool _commUseChannels,
                                                 bool _decompUseParFor,
                                                 bool _HPX4ESPP_OPTIMIZE_COMM)
    : baseClass(system, _nodeGrid, _cellGrid, _halfCellInt),
      subCellGrid({(std::size_t)_subCellGrid[0], (std::size_t)_subCellGrid[1],
                   (std::size_t)_subCellGrid[2]}),
      numCommSubs(_numCommSubs),
      commAsync(_commAsync),
      inBuf(*system->comm),
      outBuf(*system->comm),
      excgAligned(_excgAligned),
      commUseChannels(_commUseChannels),
      decompUseParFor(_decompUseParFor),
      HPX4ESPP_OPTIMIZE_COMM(_HPX4ESPP_OPTIMIZE_COMM),
      exchange_ghost_channels(nodeGrid, subNodeGrid)
{
    //    ,ghostCommunication_impl_channels(nodeGrid, subNodeGrid)
    if (halfCellInt != 1) throw std::runtime_error("hpx4espp: Not implemented for halfCellInt!=1.");
    resetTimers();
    const int rank = getSystem()->comm->rank();

    John::FullNeighbourNodeGrid full_grid(_nodeGrid, rank, {1, 1, 1});
    ch_decompose_real_sizes =
        espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::longint>, 10>(
            full_grid, rank);
    ch_decompose_real_particles =
        espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::Particle>, 11>(
            full_grid, rank);

    ch_ghostCommunication_impl =
        espressopp::hpx4espp::channels::JohnFullChannels<vec::AlignedVector<real>, 20>(full_grid,
                                                                                       rank);
    ch_exchangeGhosts_impl_sizes =
        espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::longint>, 30>(
            full_grid, rank);

    ch_exchangeGhosts_impl_particles =
        espressopp::hpx4espp::channels::JohnFullChannels<hpx::serialization::serialize_buffer<char>,
                                                         31>(full_grid, rank);
    for (const auto& neighbour : John::NeighbourEnumIterable)
    {
        outBuffers.emplace_back(BufferFixed(*(getSystem()->comm)));
        outBuffers_futs.emplace_back(hpx::make_ready_future());
    }

    exchangeGhosts_impl_john = exchangeGhosts_impl_john_opt;
    ghostCommunication_impl_john = ghostCommunication_impl_john_opt;
    decomposeRealParticlesHPXParFor_john = decomposeRealParticlesHPXParFor_john_opt;

    connect();
    resetVirtualStorage();
    HPX4ESPP_DEBUG_MSG("FullDomainDecomposition()");
}

FullDomainDecomposition::~FullDomainDecomposition() { disconnect(); }

void FullDomainDecomposition::resetVirtualStorage()
{
    SPDLOG_TRACE("Start");
    spdlog::stopwatch sw;
    const int rank = getSystem()->comm->rank();
    if (!rank) std::cout << __FUNCTION__ << std::endl;

    if (localCells.empty())
    {
        std::cerr << __FUNCTION__ << " called but there are 0 cells" << std::endl;
        return;
    }

    /// Requires subCellGrid to be multiples of mainCellGrid and maintains
    /// compatibility with functionality in espressopp.tools.decomp
    Int3D mainCellGrid = this->getInt3DCellGrid();
    for (size_t i = 0; i < 3; i++)
    {
        HPX4ESPP_ASSERT_EQUAL((size_t(mainCellGrid[i]) % subCellGrid[i]), 0);
    }

    numSubNodes = 1;  /// number of virtual subdomains
    numSubCells = 1;  /// number of real cells in each subdomain
    for (size_t i = 0; i < 3; i++)
    {
        subNodeGrid[i] = (mainCellGrid[i] / subCellGrid[i]);
        HPX4ESPP_ASSERT(subNodeGrid[i] > 0);
        numSubNodes *= subNodeGrid[i];
        numSubCells *= subCellGrid[i];
    }

    if (true && !rank)
    {
        std::cout << rank << ": mainCellGrid: " << mainCellGrid << std::endl;
        std::cout << rank << ": subCellGrid: " << subCellGrid[0] << ", " << subCellGrid[1] << ", "
                  << subCellGrid[2] << std::endl;
        std::cout << rank << ": subNodeGrid: " << subNodeGrid[0] << ", " << subNodeGrid[1] << ", "
                  << subNodeGrid[2] << std::endl;
        std::cout << rank << ": numSubCells: " << numSubCells << std::endl;
        std::cout << rank << ": numSubNodes: " << numSubNodes << std::endl;
        std::cout << rank << ": realCells.size(): " << realCells.size() << std::endl;
    }

    /// expected number of cells in the local frame grid
    size_t numSubLocalCells = 1;
    for (size_t i = 0; i < 3; i++) numSubLocalCells *= (subCellGrid[i] + 2 * halfCellInt);

    /// determine real cells for each box in advance since they are uniform
    std::vector<size_t> realCellsIdx;
    realCellsIdx.reserve(numSubCells);
    // Copy these loops
    // Half cell int affects comm cells
    // Be lazy halfcell=1 == frame
    for (size_t j2 = 0, j2end = subCellGrid[2] + 2 * halfCellInt, ctr = 0; j2 < j2end; j2++)
    {
        const bool r2 = (halfCellInt <= j2) && (j2 < (subCellGrid[2] + halfCellInt));
        for (size_t j1 = 0, j1end = subCellGrid[1] + 2 * halfCellInt; j1 < j1end; j1++)
        {
            const bool r1 = (halfCellInt <= j1) && (j1 < (subCellGrid[1] + halfCellInt));
            for (size_t j0 = 0, j0end = subCellGrid[0] + 2 * halfCellInt; j0 < j0end; j0++, ctr++)
            {
                const bool r0 = (halfCellInt <= j0) && (j0 < (subCellGrid[0] + halfCellInt));
                if (r0 && r1 && r2)
                {
                    realCellsIdx.push_back(ctr);
                }
            }
        }
    }

    if (false && !rank)
    {
        std::cout << "realCellsIdx.size(): " << realCellsIdx.size() << std::endl;
        std::cout << "    >>> realCellsIdx: ";
        for (auto const& v : realCellsIdx)
        {
            std::cout << " " << v;
        }
        std::cout << std::endl;
    }

    virtualStorage.clear();
    virtualStorage.reserve(numSubNodes);

    const auto fgs0 = cellGrid.getFrameGridSize(0);
    const auto fgs1 = cellGrid.getFrameGridSize(1);
    const auto fw = cellGrid.getFrameWidth();

    /// Expected frameGridSize of each subnode
    const size_t lfgs0 = subCellGrid[0] + 2 * fw;
    const size_t lfgs1 = subCellGrid[1] + 2 * fw;

    Cell* lc0 = this->localCells[0];

    /// Instantiate virtual storage, virtual real cells and virtual rank
    // Take this as is
    for (size_t i2 = 0; i2 < subNodeGrid[2]; i2++)  // z
    {
        for (size_t i1 = 0; i1 < subNodeGrid[1]; i1++)  // y
        {
            for (size_t i0 = 0; i0 < subNodeGrid[0]; i0++)  // x
            {
                virtualStorage.emplace_back(VirtualStorage());
                auto& vs = virtualStorage.back();
                vs.particles.markRealCells(realCellsIdx, numSubLocalCells);

                /// local cells in cellGrid
                {
                    auto& vc = vs.localCells;
                    vc.reserve(numSubLocalCells);
                    VirtualStorage::MapType cellMap;

                    const size_t j2start = i2 * subCellGrid[2];
                    const size_t j1start = i1 * subCellGrid[1];
                    const size_t j0start = i0 * subCellGrid[0];

                    const size_t j2end = (i2 + 1) * subCellGrid[2] + 2 * fw;
                    const size_t j1end = (i1 + 1) * subCellGrid[1] + 2 * fw;
                    const size_t j0end = (i0 + 1) * subCellGrid[0] + 2 * fw;

                    for (size_t j2 = j2start; j2 < j2end; j2++)
                    {
                        for (size_t j1 = j1start; j1 < j1end; j1++)
                        {
                            for (size_t j0 = j0start; j0 < j0end; j0++)
                            {
                                const size_t gidx = j0 + fgs0 * (j1 + fgs1 * j2);
                                const size_t lidx = vc.size();
                                cellMap[gidx] = lidx;
                                vc.push_back(localCells.at(gidx));
                            }
                        }
                    }
                    vs.setCellMap(std::forward<VirtualStorage::MapType>(cellMap));
                    vs.cellNeighborList =
                        vec::CellNeighborList(lc0, vc, vs.particles.realCells(), vs.getCellMap());
                }

                /// real cells
                {
                    auto& vr = vs.realCellsIdx;
                    vr.reserve(numSubCells);

                    const size_t j2start = i2 * subCellGrid[2];
                    const size_t j1start = i1 * subCellGrid[1];
                    const size_t j0start = i0 * subCellGrid[0];

                    const size_t j2end = (i2 + 1) * subCellGrid[2];
                    const size_t j1end = (i1 + 1) * subCellGrid[1];
                    const size_t j0end = (i0 + 1) * subCellGrid[0];

                    for (size_t j2 = j2start; j2 < j2end; j2++)
                    {
                        for (size_t j1 = j1start; j1 < j1end; j1++)
                        {
                            for (size_t j0 = j0start; j0 < j0end; j0++)
                            {
                                const size_t gidx = j0 + fgs0 * (j1 + fgs1 * j2);
                                vr.push_back(gidx);
                            }
                        }
                    }
                }

                /// owned cells
                {
                    /// Set shift start
                    auto fw_casted = (std::size_t)fw;
                    std::array<size_t, 3> lStart = {fw_casted, fw_casted, fw_casted};
                    std::array<size_t, 3> lShiftEnd = {fw_casted, fw_casted, fw_casted};

                    if (i2 == 0) lStart[2] = 0;
                    if (i1 == 0) lStart[1] = 0;
                    if (i0 == 0) lStart[0] = 0;

                    if (i2 == (subNodeGrid[2] - 1)) lShiftEnd[2] = 2 * fw;
                    if (i1 == (subNodeGrid[1] - 1)) lShiftEnd[1] = 2 * fw;
                    if (i0 == (subNodeGrid[0] - 1)) lShiftEnd[0] = 2 * fw;

                    const size_t j2start = i2 * subCellGrid[2] + lStart[2];
                    const size_t j1start = i1 * subCellGrid[1] + lStart[1];
                    const size_t j0start = i0 * subCellGrid[0] + lStart[0];

                    const size_t j2end = (i2 + 1) * subCellGrid[2] + lShiftEnd[2];
                    const size_t j1end = (i1 + 1) * subCellGrid[1] + lShiftEnd[1];
                    const size_t j0end = (i0 + 1) * subCellGrid[0] + lShiftEnd[0];

                    auto& vo = vs.ownCellsIdx;
                    vo.reserve(vs.localCells.size());
                    for (size_t j2 = j2start, l2 = lStart[2]; j2 < j2end; j2++, l2++)
                    {
                        for (size_t j1 = j1start, l1 = lStart[1]; j1 < j1end; j1++, l1++)
                        {
                            for (size_t j0 = j0start, l0 = lStart[0]; j0 < j0end; j0++, l0++)
                            {
                                const size_t gidx = j0 + fgs0 * (j1 + fgs1 * j2);
                                const size_t lidx = l0 + lfgs0 * (l1 + lfgs1 * l2);
                                vo.push_back(lidx);

                                if (true)
                                {
                                    HPX4ESPP_ASSERT_EQUAL(vs.localCells[lidx], localCells[gidx]);
                                    HPX4ESPP_ASSERT_EQUAL(vs.localCells[lidx] - lc0, gidx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// verification of vs.ownCellsIdx
    if (true)
    {
        size_t totalCells = 0;
        std::vector<int> ownCellsIdx(localCells.size(), -1);
        for (size_t inode = 0; inode < virtualStorage.size(); inode++)
        {
            auto const& vs = virtualStorage[inode];
            for (auto const& oc : vs.ownCellsIdx)
            {
                const auto gidx = vs.localCells.at(oc) - lc0;
                HPX4ESPP_ASSERT_EQUAL_MSG(ownCellsIdx.at(gidx), -1, "Overlap of own cells at: ");
                ownCellsIdx.at(gidx) = inode;
            }
            totalCells += vs.ownCellsIdx.size();
        }
        HPX4ESPP_ASSERT_EQUAL(totalCells, localCells.size());

        for (const auto& oc : ownCellsIdx)
        {
            HPX4ESPP_ASSERT_NEQ(oc, -1);
        }
    }

    /// mark owned cells
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        for (size_t inode = 0; inode < virtualStorage.size(); inode++)
        {
            auto const& vs = virtualStorage[inode];
            virtualStorage[inode].particles.markOwnCells(vs.localCells, vs.ownCellsIdx);
        }
    }

    if (false)  // ENABLE_CHECK
    {
        std::cout << rank << ": localCells.size(): " << localCells.size() << std::endl;
        std::cout << rank << ": virtualStorage -> vs.localCells.size(): ";
        for (auto const& vs : virtualStorage) std::cout << " " << vs.localCells.size();
        std::cout << "\n";

        std::cout << rank << ": virtualStorage -> vs.vrank: ";
        for (auto const& vs : virtualStorage) std::cout << " " << vs.vrank;
        std::cout << "\n";
    }

    /// Fill in subnode parameters
    {
        Int3D mainNodeGrid = getInt3DNodeGrid();
        Int3D grid = mainNodeGrid;
        for (int i = 0; i < 3; i++) grid[i] *= subNodeGrid[i];

        /// vrank assignment divided into nodeGrid
        std::vector<int> myVranks;
        {
            if (false && !rank)
            {
                std::cout << rank << ": grid: " << grid << std::endl;
                std::cout << rank << ": vrank assignment: " << grid << std::endl;
            }

            myVranks.reserve(numSubNodes);
            for (int i2 = 0; i2 < grid[2]; i2++)
            {
                int j2 = i2 / subNodeGrid[2];
                for (int i1 = 0; i1 < grid[1]; i1++)
                {
                    int j1 = i1 / subNodeGrid[1];
                    for (int i0 = 0; i0 < grid[0]; i0++)
                    {
                        int j0 = i0 / subNodeGrid[0];

                        const int vrank = i0 + grid[0] * (i1 + grid[1] * (i2));
                        const int mnode = j0 + mainNodeGrid[0] * (j1 + mainNodeGrid[1] * (j2));

                        if (rank == mnode)
                        {
                            myVranks.push_back(vrank);
                        }

                        if (false && !rank)
                        {
                            std::cout << rank << ":    "
                                      << " i: (" << j0 << "," << i1 << "," << i2 << ")"
                                      << " j: (" << j0 << "," << j1 << "," << j2 << ")"
                                      << " vrank: " << vrank << " mnode: " << mnode << std::endl;
                        }
                    }
                }
            }
            HPX4ESPP_ASSERT_EQUAL(myVranks.size(), numSubNodes);
        }

        const Real3D domainSize = getSystem()->bc->getBoxL();
        const Int3D subCellGridInt3D(subCellGrid[0], subCellGrid[1], subCellGrid[2]);

        Int3D gridLocal(subNodeGrid[0], subNodeGrid[1], subNodeGrid[2]);

        vdd.clear();
        vdd.reserve(virtualStorage.size());

        using espressopp::CellGrid;
        using espressopp::storage::NodeGrid;
        for (size_t inode = 0; inode < virtualStorage.size(); inode++)
        {
            virtualStorage[inode].vrank = myVranks[inode];
            auto const& vs = virtualStorage[inode];
            vdd.emplace_back(FVDD());
            auto& myVdd = vdd.back();
            myVdd.nodeGrid = NodeGrid(grid, vs.vrank, domainSize);
            real myLeft[3];
            real myRight[3];
            for (int i = 0; i < 3; ++i)
            {
                myLeft[i] = myVdd.nodeGrid.getMyLeft(i);
                myRight[i] = myVdd.nodeGrid.getMyRight(i);
            }
            myVdd.cellGrid = CellGrid(subCellGridInt3D, myLeft, myRight, halfCellInt);

            Real3D domainSizeLocal;
            for (int i = 0; i < 3; ++i) domainSizeLocal[i] = myRight[i] - myLeft[i];
            myVdd.nodeGridLocal = NodeGrid(gridLocal, inode, domainSizeLocal);

            myVdd.fullNodeGridLocal =
                John::FullNeighbourNodeGrid(gridLocal, inode, domainSizeLocal);
        }

        if (false)  // ENABLE_CHECK
        {
            const int mpiSize = getSystem()->comm->size();
            for (int irank = 0; irank < mpiSize; irank++)
            {
                if (rank == irank)
                {
                    auto f_grid_info = [](const auto& grid)
                    {
                        std::ostringstream ss;
                        ss << "(" << grid.getGridSize(0) << ", " << grid.getGridSize(1) << ", "
                           << grid.getGridSize(2) << ") "
                           << "(" << grid.getMyLeft(0) << ", " << grid.getMyLeft(1) << ", "
                           << grid.getMyLeft(2) << ") "
                           << "(" << grid.getMyRight(0) << ", " << grid.getMyRight(1) << ", "
                           << grid.getMyRight(2) << ") ";
                        return ss.str();
                    };

                    std::cout << rank << ": domainSize: " << domainSize << std::endl;
                    std::cout << rank << ": GridInfo" << std::endl;
                    std::cout << rank << ": nodeGrid: " << f_grid_info(nodeGrid) << std::endl;
                    std::cout << rank << ": cellGrid: " << f_grid_info(cellGrid) << std::endl;
                    for (size_t inode = 0; inode < vdd.size(); inode++)
                    {
                        const auto& vs = virtualStorage[inode];
                        const auto& vd = vdd[inode];
                        std::cout << rank << ": "
                                  << "inode:" << inode << " "
                                  << "vrank:" << vs.vrank << " "
                                  << "vd.nodeGrid: " << f_grid_info(vd.nodeGrid) << "\n    "
                                  << "vd.cellGrid: " << f_grid_info(vd.cellGrid) << std::endl;
                    }
                }
                getSystem()->comm->barrier();
            }
        }

        /// Map back own cells (global Cell index) to subNode + offset (subnode index in localCells)
        /// TODO: Expose as member variable
        /// Should be as big as storage.localcells
        std::vector<std::pair<size_t, size_t>> mapCellToSubNode;
        constexpr auto size_t_max = std::numeric_limits<size_t>::max();
        {
            mapCellToSubNode.resize(localCells.size());
            std::fill(mapCellToSubNode.begin(), mapCellToSubNode.end(),
                      std::make_pair(size_t_max, size_t_max));
            for (size_t inode = 0; inode < numSubNodes; inode++)
            {
                auto const& vs = this->virtualStorage[inode];
                for (const auto& lidx : vs.ownCellsIdx)
                {
                    const size_t gidx = vs.localCells[lidx] - lc0;
                    mapCellToSubNode[gidx] = std::make_pair(inode, lidx);
                }
            }

            /// verification
            if (true)
            {
                for (const auto& pair : mapCellToSubNode)
                {
                    HPX4ESPP_ASSERT_NEQ(pair.first, size_t_max);
                    HPX4ESPP_ASSERT_NEQ(pair.second, size_t_max);
                }
            }
        }

        auto isInnerCell = [](auto const& cellGrid, auto const index)
        {
            int m, n, o;
            cellGrid.mapIndexToPosition(m, n, o, index);
            return cellGrid.isInnerCell(m, n, o);
        };

        /// Fill in cellGridInfo
        for (size_t inode = 0; inode < virtualStorage.size(); inode++)
        {
            /// init cell grid info
            {
                auto const& vs = this->virtualStorage[inode];
                auto const& vd = this->vdd[inode];
                const size_t gridNumCells =
                    vd.cellGrid.getNumberOfCells();  // Number of cells in dimension (x*y*z)
                // std::vector<FVDD::CellInfo> cellGridInfo(gridNumCells);

                auto& cellGridInfo = this->vdd[inode].cellGridInfo;
                cellGridInfo.clear();
                cellGridInfo.resize(gridNumCells);

                if (vs.localCells.size() != gridNumCells)
                    throw std::runtime_error("(vs.localCells.size()!=gridNumCells)");

                for (longint icell = 0; icell < gridNumCells; icell++)
                {
                    /// icell to gidx
                    const auto cell_gidx = vs.localCells.at(icell) - localCells[0];
                    /// cell_gidx to inode, offset
                    const auto cell_node_lidx = mapCellToSubNode.at(cell_gidx);
                    const auto cell_node = cell_node_lidx.first;
                    const auto cell_lidx = cell_node_lidx.second;

                    if (cell_node == inode)
                    {
                        if (isInnerCell(vd.cellGrid, icell))
                        {
                            cellGridInfo[icell].cellType = FVDD::CellInfo::CELL_REAL;
                        }
                        else
                        {
                            cellGridInfo[icell].cellType = FVDD::CellInfo::CELL_GHOST_EXTERNAL;
                        }
                        cellGridInfo[icell].subNode = inode;
                        cellGridInfo[icell].subNodeCell = icell;
                    }
                    else
                    {
                        cellGridInfo[icell].cellType = FVDD::CellInfo::CELL_GHOST_INTERNAL;
                        cellGridInfo[icell].subNode = cell_node;
                        cellGridInfo[icell].subNodeCell = cell_lidx;
                    }
                }
            }
        }

        /// Fill in realNbrs, externalNbrs, internalNbrs
        for (size_t inode = 0; inode < virtualStorage.size(); inode++)
        {
            /// init cell neighbors
            {
                const auto& vs = this->virtualStorage[inode];
                const auto& vd = this->vdd[inode];

                const auto cell0 = this->localCells[0];
                const auto& localCells = vs.localCells;
                const auto& realCellIdx = vs.particles.realCells();
                const size_t numCells = realCellIdx.size();

                auto& rnbr = this->virtualStorage[inode].realNbrs;
                auto& enbr = this->virtualStorage[inode].externalNbrs;
                auto& inbr = this->virtualStorage[inode].internalNbrs;

                rnbr.clear();
                enbr.clear();
                inbr.clear();

                /// key: inode, value: (p, list of nps)
                std::map<size_t, std::map<size_t, std::vector<size_t>>> temp;

                /// loop through cellGrid cells
                for (size_t ic = 0; ic < vs.localCells.size(); ic++)
                {
                    const Cell* cell = vs.localCells[ic];

                    /// accept only if this cell is real in the full cellgrid
                    const size_t icGidx = cell - cell0;
                    if (!isInnerCell(this->cellGrid, icGidx)) continue;

                    /// who owns this cell
                    const auto& icNodeCell = mapCellToSubNode.at(icGidx);
                    const auto& icNode = icNodeCell.first;
                    const auto& icCell = icNodeCell.second;
                    const bool icIsOwned = (icNode == inode);

                    /// ensures an entry for all real cells
                    temp[inode][icCell];

                    for (const auto& nc : cell->neighborCells)
                    {
                        if (!nc.useForAllPairs)
                        {
                            const auto ncGidx = nc.cell - cell0;

                            /// who owns this neighbor cell
                            const auto& ncNodeCell = mapCellToSubNode.at(ncGidx);
                            const auto& ncNode = ncNodeCell.first;
                            const auto& ncCell = ncNodeCell.second;
                            const bool ncIsOwned = (ncNode == inode);

                            if (icIsOwned && ncIsOwned)
                            {
                                /// n3l pair in reals in inode
                                temp[inode][icCell].push_back(ncCell);
                            }
                            else if (icIsOwned && !ncIsOwned)
                            {
                                /// insert (icCell, ncCell) to ncNode list
                                temp[ncNode][icCell].push_back(ncCell);
                            }
                            else if (!icIsOwned && ncIsOwned)
                            {
                                /// insert (ncCell, icCell) to icNode list
                                temp[icNode][ncCell].push_back(icCell);
                            }
                        }
                    }
                }

                /// compress temp
                for (const auto& pair_node : temp)
                {
                    auto const& nnode = pair_node.first;

                    vec::CellNeighborList cnl;
                    cnl.clear();
                    for (const auto& pair_cell : pair_node.second)
                    {
                        auto const& lcell = pair_cell.first;
                        std::vector<size_t> const& nbrs = pair_cell.second;

                        cnl.beginCell(lcell);
                        for (const auto& nbr : nbrs)
                        {
                            cnl.insertNeighbor(nbr);
                        }
                        cnl.endCell();
                    }
                    cnl.validate();

                    if (nnode == inode)
                    {
                        rnbr = cnl;
                    }
                    else
                    {
                        inbr.push_back({nnode, cnl});
                    }
                }
            }
        }

        // Manages internal affairs
        /// validate
        if (true)
        {
            const auto numNodes = virtualStorage.size();
            std::vector<std::set<std::pair<size_t, size_t>>> nodeCellPairs(numNodes);

            auto setPairsInsertCheck = [](auto& set, std::pair<size_t, size_t> const& pair)
            {
                const auto result = set.insert(pair);
                HPX4ESPP_ASSERT_EQUAL_MSG(
                    result.second, true,
                    "Duplicate detected at {" << pair.first << "," << pair.second << "}");
            };

            /// insert pairs
            for (size_t inode = 0; inode < numNodes; inode++)
            {
                const auto& vs = virtualStorage[inode];
                const auto& vd = vdd[inode];

                /// collect vs.cellNeighborList pairs into nodeCellPairs
                auto& ncp = nodeCellPairs[inode];
                const auto& cnl = vs.cellNeighborList;
                for (size_t ic = 0; ic < cnl.numCells(); ic++)
                {
                    const auto pcell = cnl.cellId(ic);
                    for (size_t nn = 0; nn < cnl.numNeighbors(ic); nn++)
                    {
                        const auto ncell = cnl.at(ic, nn);
                        setPairsInsertCheck(ncp, {pcell, ncell});
                        setPairsInsertCheck(ncp, {ncell, pcell});
                    }
                }
            }

            auto print_check_ncp = [&nodeCellPairs, numNodes](bool print, bool check)
            {
                for (size_t inode = 0; inode < numNodes; inode++)
                {
                    const auto& ncp = nodeCellPairs[inode];

                    for (const auto& s : ncp)
                    {
                        const auto pcell = s.first;
                        const auto ncell = s.second;
                    }

                    if (check) HPX4ESPP_ASSERT_EQUAL(ncp.size(), 0);

                    if (print)
                        std::cout << " (inode: " << inode << " numpairs: " << ncp.size() << ") ";
                }
                if (print) std::cout << std::endl;
            };

            // print_check_ncp(true, false);

            auto search_erase = [](std::pair<size_t, size_t> const& pair, auto& ncp)
            {
                auto search = ncp.find(pair);
                if (search != ncp.end())
                {
                    ncp.erase(search);
                    return 0;
                }
                else
                {
                    return 1;
                }
            };

            auto isOwnCellGlobal = [&mapCellToSubNode](auto const inode, auto const gidx)
            { return mapCellToSubNode.at(gidx).first == inode; };

            auto isOwnCell = [&mapCellToSubNode](auto const& vd, auto const lidx)
            {
                const auto cgi = vd.cellGridInfo[lidx].cellType;
                return (cgi == FVDD::CellInfo::CELL_REAL) ||
                       (cgi == FVDD::CellInfo::CELL_GHOST_EXTERNAL);
            };

            for (size_t inode = 0; inode < numNodes; inode++)
            {
                const auto& vs = virtualStorage[inode];
                const auto& vd = vdd[inode];

                const auto& rnbr = vs.realNbrs;
                const auto& enbr = vs.externalNbrs;
                const auto& inbr = vs.internalNbrs;

                /// remove pairs with real and external ghosts (i,j) and (j,i)
                auto& ncp = nodeCellPairs[inode];

                auto remove_n3l = [&ncp, &search_erase, inode, &isOwnCell, &vd](auto const& cnl)
                {
                    for (size_t ic = 0; ic < cnl.numCells(); ic++)
                    {
                        const auto pcell = cnl.cellId(ic);

                        if (!isOwnCell(vd, pcell))
                        {
                            HPX4ESPP_THROW_EXCEPTION(
                                hpx::assertion_failure, __FUNCTION__,
                                "ncellNbr " << pcell << " not in owned cells of inode " << inode
                                            << ". Failed in inode=" << inode << " ic=" << ic);
                        }

                        for (size_t nn = 0; nn < cnl.numNeighbors(ic); nn++)
                        {
                            const auto ncell = cnl.at(ic, nn);

                            if (!isOwnCell(vd, ncell))
                            {
                                HPX4ESPP_THROW_EXCEPTION(
                                    hpx::assertion_failure, __FUNCTION__,
                                    "ncell " << ncell << " not in owned cells of nnode " << inode
                                             << ". Failed in inode=" << inode << " nnode=" << inode
                                             << " ic=" << ic << " nn=" << nn);
                            }

                            if (search_erase({pcell, ncell}, ncp))
                            {
                                HPX4ESPP_THROW_EXCEPTION(
                                    hpx::assertion_failure, __FUNCTION__,
                                    "Pair not found: {" << pcell << "," << ncell
                                                        << "}. Failed in remove_n3l inode=" << inode
                                                        << " ic=" << ic << " nn=" << nn);
                            };
                            if (search_erase({ncell, pcell}, ncp))
                            {
                                HPX4ESPP_THROW_EXCEPTION(
                                    hpx::assertion_failure, __FUNCTION__,
                                    "Pair not found: {" << ncell << "," << pcell
                                                        << "}. Failed in remove_n3l inode=" << inode
                                                        << " ic=" << ic << " nn=" << nn);
                            };
                        }
                    }
                };
                remove_n3l(rnbr);
                remove_n3l(enbr);

                /// remove pairs with internal ghosts (i,j) only
                auto remove_nnode = [inode, &nodeCellPairs, &search_erase, &mapCellToSubNode,
                                     &isOwnCell, this](const size_t nnode, auto const& cnl)
                {
                    const auto& vd = this->vdd[inode];
                    const auto& vs = this->virtualStorage[inode];
                    const auto& cellMap = vs.getCellMap();
                    const auto& reals = vs.particles.realCells();
                    const auto& ghosts = vs.particles.ghostCells();

                    const auto& vdNbr = this->vdd[nnode];
                    const auto& vsNbr = this->virtualStorage[nnode];
                    const auto& cellMapNbr = vsNbr.getCellMap();
                    const auto& realsNbr = vsNbr.particles.realCells();
                    const auto& ghostsNbr = vsNbr.particles.ghostCells();

                    auto& ncp = nodeCellPairs[inode];
                    auto& ncpNbr = nodeCellPairs[nnode];

                    for (size_t ic = 0; ic < cnl.numCells(); ic++)
                    {
                        const auto pcell = cnl.cellId(ic);

                        if (!isOwnCell(vd, pcell))
                        {
                            HPX4ESPP_THROW_EXCEPTION(
                                hpx::assertion_failure, __FUNCTION__,
                                "ncellNbr " << pcell << " not in owned cells of inode " << inode
                                            << ". Failed in inode=" << inode << " ic=" << ic);
                        }

                        for (size_t nn = 0; nn < cnl.numNeighbors(ic); nn++)
                        {
                            const auto ncellNbr = cnl.at(ic, nn);
                            {
                                if (!isOwnCell(vdNbr, ncellNbr))
                                {
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, __FUNCTION__,
                                        "ncellNbr " << ncellNbr << " not in owned cells of nnode "
                                                    << nnode << ". Failed in inode=" << inode
                                                    << " nnode=" << nnode << " ic=" << ic
                                                    << " nn=" << nn);
                                }

                                /// map ncell to nnode's list of localCells
                                const auto nGidx = vsNbr.localCells[ncellNbr] - this->localCells[0];

                                /// map that localCell back to inode index
                                const auto nVidx = cellMap.at(nGidx);
                                const auto searchGhost =
                                    std::find(ghosts.begin(), ghosts.end(), nVidx);
                                if (searchGhost == ghosts.end())
                                {
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, __FUNCTION__,
                                        "cell " << nVidx << " not in ghost cells of inode "
                                                << inode);
                                }

                                const auto ncell = nVidx;

                                if (pcell < ncell)
                                {
                                    /// remove from ncp of this node
                                    if (search_erase({pcell, ncell}, ncp))
                                    {
                                        HPX4ESPP_THROW_EXCEPTION(
                                            hpx::assertion_failure, __FUNCTION__,
                                            "Pair not found: {"
                                                << pcell << "," << ncell
                                                << "}. Failed in remove_n3l inode=" << inode
                                                << " nnode=" << nnode << " ic=" << ic << " nn="
                                                << nn << " pcell=" << pcell << " ncell=" << ncell);
                                    };
                                }
                                else
                                {
                                    /// remove from ncp of neighbor node
                                    const auto pGidx = vs.localCells[pcell] - this->localCells[0];
                                    const auto pVidxNbr = cellMapNbr.at(pGidx);
                                    const auto searchGhostNbr =
                                        std::find(ghostsNbr.begin(), ghostsNbr.end(), pVidxNbr);
                                    if (searchGhostNbr == ghostsNbr.end())
                                    {
                                        HPX4ESPP_THROW_EXCEPTION(
                                            hpx::assertion_failure, __FUNCTION__,
                                            "cell " << pVidxNbr << " not in ghost cells of nnode "
                                                    << nnode);
                                    }

                                    const auto pcellNbr = pVidxNbr;
                                    if (search_erase({pcellNbr, ncellNbr}, ncpNbr))
                                    {
                                        HPX4ESPP_THROW_EXCEPTION(
                                            hpx::assertion_failure, __FUNCTION__,
                                            "Pair not found: {"
                                                << pcellNbr << "," << ncellNbr
                                                << "} in nnode. Failed in remove_n3l inode="
                                                << inode << " nnode=" << nnode << " ic=" << ic
                                                << " nn=" << nn << " pcell=" << pcell
                                                << " ncell=" << ncell);
                                    }
                                }
                            }
                        }
                    }
                };

                for (const auto& nnode_cnl : inbr) remove_nnode(nnode_cnl.first, nnode_cnl.second);
            }
            print_check_ncp(false, true);

            /// verify based on all interacting pairs of entire subdomain

            std::set<std::pair<size_t, size_t>> allCellPairs;
            const auto lc0 = this->localCells[0];
            for (const auto rc : this->realCells)
            {
                const size_t pcell = rc - lc0;
                for (const auto& nc : rc->neighborCells)
                {
                    if (!nc.useForAllPairs)
                    {
                        const size_t ncell = nc.cell - lc0;
                        setPairsInsertCheck(allCellPairs, {pcell, ncell});
                        setPairsInsertCheck(allCellPairs, {ncell, pcell});
                    }
                }
            }

            auto print_check_acp = [&allCellPairs](bool print, bool check)
            {
                if (print)
                    std::cout << "allCellPairs numpairs: " << allCellPairs.size() << std::endl;
                if (check) HPX4ESPP_ASSERT_EQUAL(allCellPairs.size(), 0);
            };

            // print_check_acp(true, false);

            auto remove = [&allCellPairs, &search_erase, this](const auto& cnl, const auto& vs,
                                                               const auto& vsNbr, const bool n3l)
            {
                for (size_t ic = 0; ic < cnl.numCells(); ic++)
                {
                    const auto pcell = cnl.cellId(ic);
                    const auto pcell_gidx = vs.localCells[pcell] - this->localCells[0];
                    for (size_t nn = 0; nn < cnl.numNeighbors(ic); nn++)
                    {
                        const auto ncellNbr = cnl.at(ic, nn);
                        const auto ncell_gidx = vsNbr.localCells[ncellNbr] - this->localCells[0];

                        auto genMsg = [&, this](auto pair)
                        {
                            std::ostringstream ss;
                            ss << "Pair not found: {" << pair.first << "," << pair.second
                               << "}. Failed in remove inode=" << (&vs - &virtualStorage[0])
                               << " nnode=" << (&vsNbr - &virtualStorage[0]) << " ic=" << ic
                               << " nn=" << nn << " pcell=" << pcell << " ncellNbr=" << ncellNbr;
                            return ss.str();
                        };

                        {
                            const auto pair = std::make_pair(pcell_gidx, ncell_gidx);
                            if (search_erase(pair, allCellPairs))
                            {
                                HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, __FUNCTION__,
                                                         genMsg(pair));
                            }
                        }

                        if (n3l)
                        {
                            const auto pair = std::make_pair(ncell_gidx, pcell_gidx);
                            if (search_erase(pair, allCellPairs))
                            {
                                HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, __FUNCTION__,
                                                         genMsg(pair));
                            }
                        }
                    }
                }
            };

            for (size_t inode = 0; inode < numNodes; inode++)
            {
                const auto& vs = virtualStorage[inode];

                const auto& rnbr = vs.realNbrs;
                const auto& enbr = vs.externalNbrs;
                const auto& inbr = vs.internalNbrs;

                /// TODO: Remove pairs from allCellPairs

                remove(rnbr, virtualStorage[inode], virtualStorage[inode], true);
                remove(enbr, virtualStorage[inode], virtualStorage[inode], true);

                for (const auto& nnode_cnl : inbr)
                    remove(nnode_cnl.second, virtualStorage[inode], virtualStorage[nnode_cnl.first],
                           false);
            }

            print_check_acp(false, true);
        }

        /// print stats about cnls
        if (false)  // ENABLE_CHECK
        {
            auto cnlSize = [](const auto& cnl)
            {
                size_t size = 0;
                for (size_t ic = 0; ic < cnl.numCells(); ic++)
                {
                    size += cnl.numNeighbors(ic);
                }
                return size;
            };

            for (size_t inode = 0; inode < virtualStorage.size(); inode++)
            {
                const auto& vs = virtualStorage[inode];
                const auto& rnbr = vs.realNbrs;
                const auto& enbr = vs.externalNbrs;
                const auto& inbr = vs.internalNbrs;

                std::cout << "  inode=" << inode << " rnbr.size()=" << cnlSize(rnbr)
                          << " enbr.size()=" << cnlSize(enbr) << " inbr:";

                for (const auto& nnode_cnl : inbr)
                    std::cout << " (" << nnode_cnl.first << "," << cnlSize(nnode_cnl.second) << ")";
                std::cout << std::endl;
            }
        }

        /// print stats about specific cells
        if (false && !rank)
        {
            std::set<size_t> checkNodes = {0, 1, 3, 4};
            constexpr size_t checkCell = 93;

            auto checkCNL = [](const auto& cnl)
            {
                std::ostringstream oss;
                for (size_t ic = 0; ic < cnl.numCells(); ic++)
                {
                    if (cnl.cellId(ic) != checkCell) continue;
                    for (size_t nn = 0; nn < cnl.numNeighbors(ic); nn++)
                    {
                        const auto ncellNbr = cnl.at(ic, nn);

                        oss << " " << ncellNbr;
                    }
                }
                return oss.str();
            };

            for (size_t inode = 0; inode < virtualStorage.size(); inode++)
            {
                const auto find = checkNodes.find(inode);
                if (find == checkNodes.end()) continue;

                const auto& vs = virtualStorage[inode];
                const auto& rnbr = vs.realNbrs;
                const auto& enbr = vs.externalNbrs;
                const auto& inbr = vs.internalNbrs;

                std::cout << "inode=(" << inode;
                std::cout << ") rnbr: (";
                std::cout << checkCNL(rnbr);
                std::cout << ") enbr: (";
                std::cout << checkCNL(enbr);
                std::cout << ")" << std::endl;

                for (const auto& nnode_cnl : inbr)
                {
                    const auto ss = checkCNL(nnode_cnl.second);
                    if (ss.size() > 0)
                        std::cout << "  nnode=" << nnode_cnl.first << " (" << ss << ")"
                                  << std::endl;
                }
            }
        }

        if (!rank)
        {
            std::cout << "HPX4ESPP_OPTIMIZE_COMM=" << HPX4ESPP_OPTIMIZE_COMM << std::endl;
        }

        if (!HPX4ESPP_OPTIMIZE_COMM)
        {
            for (auto& vs : this->virtualStorage)
            {
                vs.realNbrs = vs.cellNeighborList;
                vs.externalNbrs.clear();
                vs.internalNbrs.clear();
            }
        }
    }

    /// prepare ghost communication
    {
        auto vFillCells = [](std::vector<size_t>& cv, const int leftBoundary[3],
                             const int rightBoundary[3], CellGrid const& cg)
        {
            longint total = 1;
            for (int i = 0; i < 3; ++i)
            {
                if (leftBoundary[i] < 0 || leftBoundary[i] > cg.getFrameGridSize(i) ||
                    rightBoundary[i] < 0 || rightBoundary[i] > cg.getFrameGridSize(i) ||
                    leftBoundary[i] >= rightBoundary[i])
                {
                    throw std::runtime_error(
                        "FullDomainDecomposition::vFillCells: wrong cell grid specified "
                        "internally");
                }
                total *= (rightBoundary[i] - leftBoundary[i]);
            }
            cv.reserve(total);

            for (int k = leftBoundary[2]; k < rightBoundary[2]; ++k)
            {
                for (int j = leftBoundary[1]; j < rightBoundary[1]; ++j)
                {
                    for (int i = leftBoundary[0]; i < rightBoundary[0]; ++i)
                    {
                        const size_t flattened_cell_index = cg.mapPositionToIndex(i, j, k);
                        cv.push_back(flattened_cell_index);
                    }
                }
            }
        };

        auto vPrepareGhostCommunication = [&vFillCells](FVDD& vd)
        {
            auto& vcg = vd.cellGrid;
            auto& vcc = vd.commCells;

            auto& vcc_full = vd.fullCommCells;

            for (auto& c : vcc)
            {
                c.reals.clear();
                c.ghosts.clear();
            }

            for (auto& cell_idx_info : vcc_full)
            {
                cell_idx_info.reals.clear();
                cell_idx_info.ghosts.clear();
            }

            // direction loop: x, y, z
            // JOHN_OLD (1): This loop builds the reals and ghost of the vdd 6 neighbour.
            for (int coord = 0; coord < 3; ++coord)
            {
                // boundaries of area to send
                int leftBoundary[3], rightBoundary[3];
                /* boundaries perpendicular directions are the same for left/right send.
                We also send the ghost frame that we have already, so the data amount
                increase with each cycle.

                For a direction that was done already, i.e. is smaller than dir,
                we take the full ghost frame, otherwise only the inner frame.  */
                for (int offset = 1; offset <= 2; ++offset)
                {
                    int otherCoord = (coord + offset) % 3;
                    if (otherCoord < coord)
                    {
                        leftBoundary[otherCoord] = 0;
                        rightBoundary[otherCoord] = vcg.getFrameGridSize(otherCoord);
                    }
                    else
                    {
                        leftBoundary[otherCoord] = vcg.getInnerCellsBegin(otherCoord);
                        rightBoundary[otherCoord] = vcg.getInnerCellsEnd(otherCoord);
                    }
                }

                //  lr loop: left right - loop
                for (int lr = 0; lr < 2; ++lr)
                {
                    int dir = 2 * coord + lr;

                    if (lr == 0)
                    {
                        leftBoundary[coord] = vcg.getInnerCellsBegin(coord);
                        rightBoundary[coord] = vcg.getInnerCellsBegin(coord) + vcg.getFrameWidth();
                    }
                    else
                    {
                        leftBoundary[coord] = vcg.getInnerCellsEnd(coord) - vcg.getFrameWidth();
                        rightBoundary[coord] = vcg.getInnerCellsEnd(coord);
                    }
                    auto& vcc_reals = vcc[dir].reals;
                    vFillCells(vcc_reals, leftBoundary, rightBoundary, vcg);
                    std::sort(vcc_reals.begin(), vcc_reals.end());

                    if (lr == 0)
                    {
                        leftBoundary[coord] = vcg.getInnerCellsEnd(coord);
                        rightBoundary[coord] = vcg.getInnerCellsEnd(coord) + vcg.getFrameWidth();
                    }
                    else
                    {
                        leftBoundary[coord] = vcg.getInnerCellsBegin(coord) - vcg.getFrameWidth();
                        rightBoundary[coord] = vcg.getInnerCellsBegin(coord);
                    }
                    auto& vcc_ghosts = vcc[dir].ghosts;
                    vFillCells(vcc_ghosts, leftBoundary, rightBoundary, vcg);
                    std::sort(vcc_ghosts.begin(), vcc_ghosts.end());
                }
            }

            // JOHN_NEW (1): This loop builds the reals and ghost of the vdd 26 neighbour.
            using neighour_calc = Geometry::CuboidStructureCalc;
            neighour_calc v_cell_grid_calculator{Geometry::CuboidPair{
                {vcg.getInnerCellsBegin(0), vcg.getInnerCellsBegin(1), vcg.getInnerCellsBegin(2)},
                {vcg.getInnerCellsEnd(0), vcg.getInnerCellsEnd(1), vcg.getInnerCellsEnd(2)}}};
            int leftBoundary_arr[3], rightBoundary_arr[3];
            for (const auto& neighbour : John::NeighbourEnumIterable)
            {
                const auto [leftBoundary_real, rightBoundary_real] =
                    v_cell_grid_calculator
                        .getCuboidItems()[Geometry::cuboidFacefromNeighbourRelation(neighbour)];
                auto& vcc_reals = vcc_full[neighbour].reals;
                leftBoundary_real.to_raw_array(leftBoundary_arr);
                rightBoundary_real.to_raw_array(rightBoundary_arr);
                vFillCells(vcc_reals, leftBoundary_arr, rightBoundary_arr, vcg);
                const auto [leftBoundary_ghost, rightBoundary_ghost] =
                    v_cell_grid_calculator
                        .getCuboidFrameExtension()[Geometry::cuboidFacefromNeighbourRelation(
                            neighbour)];
                const auto opposite_neighbour = John::periodic_neighbour[neighbour];

                auto& vcc_ghosts = vcc_full[opposite_neighbour].ghosts;
                leftBoundary_ghost.to_raw_array(leftBoundary_arr);
                rightBoundary_ghost.to_raw_array(rightBoundary_arr);
                vFillCells(vcc_ghosts, leftBoundary_arr, rightBoundary_arr, vcg);
            }
        };

        const size_t nvd = vdd.size();
        for (size_t i = 0; i < nvd; i++) vPrepareGhostCommunication(vdd.at(i));

        /// For inter-node ghost updates
        {
            /// Assert pre-condition that vnode is at least 2*halfCellInt (satisfied in resortWaves)
            for (size_t i = 0; i < 3; i++)
            {
                HPX4ESPP_ASSERT_GEQ((subCellGrid[i]), (2 * halfCellInt));
            }
            /// Determine which vnodes and directions participate in inter-node updates
            int const rank = getSystem()->comm->rank();

            if (false && !rank)
            {
                std::cout << __FUNCTION__ << std::endl;
                std::cout << "subNodeGrid:"
                          << " " << subNodeGrid[0] << " " << subNodeGrid[1] << " " << subNodeGrid[2]
                          << std::endl;
                std::cout << "subNodeGrid: ";
                std::copy(std::begin(subNodeGrid), std::end(subNodeGrid),
                          std::ostream_iterator<size_t>(std::cout, " "));
                std::cout << std::endl;
            }

            for (auto& c : commNodesReal) c.clear();
            for (auto& c : commNodesGhost) c.clear();
            for (auto& c : fullCommNodesReal) c.clear();
            for (auto& c : fullCommNodesGhost) c.clear();
            // JOHN_OLD (2) : Calculate the flattened sub-nodes that participate in communication
            for (size_t thisCoord = 0; thisCoord < 3; ++thisCoord)
            {
                size_t const left = 0;
                size_t const right = subNodeGrid[thisCoord] - 1;
                size_t const mod1 = (thisCoord + 1) % 3;  // x,y,z coords
                size_t const mod2 = (thisCoord + 2) % 3;  // x,y,z coords
                size_t const otherCoord1 = std::min(mod1, mod2);
                size_t const otherCoord2 = std::max(mod1, mod2);

                /// loop through the other coords, fixing this coord
                // TODO: Change to 26 thing
                auto fillPlane = [this, &otherCoord2, &otherCoord1, thisCoord](
                                     size_t initCoord, std::vector<size_t>& plane)
                {
                    std::array<size_t, 3> nodeCoord{};
                    nodeCoord[thisCoord] = initCoord;
                    for (size_t i2 = 0; i2 < subNodeGrid[otherCoord2]; i2++)
                    {
                        nodeCoord[otherCoord2] = i2;
                        for (size_t i1 = 0; i1 < subNodeGrid[otherCoord1]; i1++)
                        {
                            nodeCoord[otherCoord1] = i1;
                            // Flattens node grid position
                            SPDLOG_TRACE("this Coord {} Node Grid {}", thisCoord,
                                         spdlog::fmt_lib::join(nodeCoord, ","));
                            plane.push_back(nodeCoord[0] +
                                            subNodeGrid[0] *
                                                (nodeCoord[1] + subNodeGrid[1] * nodeCoord[2]));
                        }
                    }
                };

                std::vector<size_t>& planeL = commNodesReal[2 * thisCoord + 0];
                fillPlane(left, planeL);
                commNodesGhost[2 * thisCoord + 1] = planeL;

                std::vector<size_t>& planeR = commNodesReal[2 * thisCoord + 1];
                fillPlane(right, planeR);
                commNodesGhost[2 * thisCoord + 0] = planeR;

                if (true && !rank)
                {
                    std::cout << "thisCoord:    " << thisCoord << std::endl;
                    std::cout << " otherCoord1: " << otherCoord1 << std::endl;
                    std::cout << " otherCoord2: " << otherCoord2 << std::endl;
                    std::cout << " left:        " << left << std::endl;
                    std::cout << " right:       " << right << std::endl;

                    auto printList = [](auto const& list, auto const& name)
                    {
                        std::cout << " " << name << ": ";
                        std::copy(std::begin(list), std::end(list),
                                  std::ostream_iterator<size_t>(std::cout, " "));
                        std::cout << std::endl;
                    };

                    printList(
                        commNodesReal[2 * thisCoord + 0],
                        std::string("commNodesReal[") + std::to_string(2 * thisCoord + 0) + "]");
                    printList(
                        commNodesReal[2 * thisCoord + 1],
                        std::string("commNodesReal[") + std::to_string(2 * thisCoord + 1) + "]");
                    printList(
                        commNodesGhost[2 * thisCoord + 0],
                        std::string("commNodesGhost[") + std::to_string(2 * thisCoord + 0) + "]");
                    printList(
                        commNodesGhost[2 * thisCoord + 1],
                        std::string("commNodesGhost[") + std::to_string(2 * thisCoord + 1) + "]");
                }
            }
            /// Calculates the partitions in case of subnodes communications

            // JOHN_NEW (2) : Calculate the flattened sub-nodes that participate in communication
            using face_calc = Geometry::CuboidStructureCalc;
            face_calc subNodeCalc{face_calc::coordinate_type{static_cast<int>(subNodeGrid[0]),
                                                             static_cast<int>(subNodeGrid[1]),
                                                             static_cast<int>(subNodeGrid[2])}};
            for (const auto& neighbour : John::NeighbourEnumIterable)
            {
                const auto [bottom_left, top_right] =
                    subNodeCalc
                        .getCuboidItems()[Geometry::cuboidFacefromNeighbourRelation(neighbour)];

                // Inversed order to produce a sorted list
                for (size_t k = bottom_left.z; k < top_right.z; ++k)
                {
                    for (size_t j = bottom_left.y; j < top_right.y; ++j)
                    {
                        for (size_t i = bottom_left.x; i < top_right.x; ++i)
                        {
                            auto flattened_coordinate =
                                i + subNodeGrid[0] * (j + subNodeGrid[1] * k);
                            fullCommNodesReal[neighbour].emplace_back(flattened_coordinate);
                            const auto opposite_neighbour = John::periodic_neighbour[neighbour];

                            fullCommNodesGhost[opposite_neighbour].emplace_back(
                                flattened_coordinate);
                            HPX4ESPP_ASSERT_LT_MSG(
                                flattened_coordinate, numSubNodes,
                                "flattened coordinate has to be inside the node grid")
                        }
                    }
                }
            }

            if (true)
            {
                if (commNodesReal[0] != fullCommNodesReal[0])
                {
                    HPX4ESPP_ASSERT(false);
                }
                if (commNodesReal[1] != fullCommNodesReal[1])
                {
                    HPX4ESPP_ASSERT(false);
                }
                if (commNodesGhost[0] != fullCommNodesGhost[0])
                {
                    HPX4ESPP_ASSERT(false);
                }
                if (commNodesGhost[1] != fullCommNodesGhost[1])
                {
                    HPX4ESPP_ASSERT(false);
                }
            }
            /// Not supported now

            if (true && !rank)
            {
                std::cout << rank << ": numCommNodesReal:  ";
                for (auto& c : commNodesReal) std::cout << " " << c.size();
                std::cout << std::endl;
                std::cout << rank << ": numCommNodesGhost: ";
                for (auto& c : commNodesGhost) std::cout << " " << c.size();
                std::cout << std::endl;
                std::cout << rank << ": numCommCellsReal:  ";
                for (int i = 0; i < 6; i++) std::cout << " " << commCells[i].reals.size();
                std::cout << std::endl;
                std::cout << rank << ": numCommCellsGhost: ";
                for (int i = 0; i < 6; i++) std::cout << " " << commCells[i].ghosts.size();
                std::cout << std::endl;
                std::cout << rank << ": numCommSubs: " << numCommSubs << std::endl;
                /// Calculates the partitions in case of subnodes communications
                /// Not used
                {
                    //                if (numCommSubs > 1)
                    //                {
                    //                    std::cout << rank << ": subdividing commNodes further: ";
                    //                    for (size_t dir = 0; dir < 6; dir++)
                    //                    {
                    //                        std::cout << "\n "
                    //                                  << "dir " << dir << ": ";
                    //                        const size_t maxNodes = false ? vdd.size() : 1;
                    //                        for (size_t inode = 0; inode < maxNodes; inode++)
                    //                        {
                    //                            const auto& vd = vdd[inode];
                    //                            const auto& cc =
                    //                                true ? vd.commCells[dir].reals :
                    //                                vd.commCells[dir].ghosts;
                    //                            const auto numCells = cc.size();
                    ////                            const auto& sizes = vd.numCommSubCells[dir];
                    ////                            const auto& ranges = vd.commSubCellRanges[dir];
                    //
                    //                            std::cout << "\n    ";
                    //                            std::cout << "inode"
                    //                                      << "=" << std::setw(4) << std::left <<
                    //                                      inode;
                    //                            std::cout << "numCells"
                    //                                      << "=" << numCells;
                    //                            std::cout << "\n        ";
                    //                            std::cout << "sizes:";
                    ////                            for (const auto& size : sizes) std::cout << " "
                    ///<< size; /                            std::cout << "\n        "; / std::cout
                    ///<< "ranges:"; /                            for (const auto& range : ranges)
                    ////                                std::cout << " (" << range.first << "," <<
                    /// range.second << ")";
                    //                        }
                    //                    }
                    //                    std::cout << std::endl;
                    //                }
                }

                /// print commCells
                if (false && !rank)
                {
                    auto printCommCells = [this](int dir, bool reals)
                    {
                        std::ostringstream ss;
                        ss << "commCells[" << dir << "]." << (reals ? "reals" : "ghosts");
                        const auto& cc = reals ? commCells[dir].reals : commCells[dir].ghosts;
                        std::cout << " " << ss.str() << ": ";
                        for (auto const& c : cc)
                        {
                            int icell = c - localCells[0];
                            std::cout << icell << " ";
                        }
                        std::cout << std::endl;
                    };

                    printCommCells(0, true);
                    printCommCells(0, false);
                }

                /// verify that commCells for this direction are fully contained within these planes
                {
                    auto verifyCommNodes =
                        [this](size_t dir, std::vector<size_t> const& plane, bool commReal)
                    {
                        /// collect all localCells for this plane
                        std::set<Cell*> lcSet;
                        for (const auto& n : plane)
                        {
                            auto const& lc = this->virtualStorage.at(n).localCells;
                            std::copy(lc.begin(), lc.end(), std::inserter(lcSet, lcSet.end()));
                        }

                        /// make sure that every commCell in this direction (ghost and real) belongs
                        /// to lcSet
                        if (commReal)
                        {
                            for (Cell* c : commCells[dir].reals)
                            {
                                HPX4ESPP_ASSERT(!(lcSet.find(c) == lcSet.end()));
                            }
                        }
                        else
                        {
                            for (Cell* c : commCells[dir].ghosts)
                            {
                                HPX4ESPP_ASSERT(!(lcSet.find(c) == lcSet.end()));
                            }
                        }
                    };

                    for (size_t coord = 0; coord < 3; ++coord)
                    {
                        for (int lr = 0; lr < 2; ++lr)
                        {
                            size_t const dir = 2 * coord + lr;
                            size_t const oppDir = 2 * coord + (1 - lr);
                            verifyCommNodes(dir, commNodesReal[dir], true);
                            verifyCommNodes(oppDir, commNodesGhost[oppDir], false);
                        }
                    }
                }
                {
                    auto verifyCommNodesFull = [this](John::NeighbourRelation relation,
                                                      std::vector<size_t> const& relation_arr,
                                                      bool commReal)
                    {
                        /// collect all localCells for this plane
                        std::set<Cell*> lcSet;
                        for (const auto& n : relation_arr)
                        {
                            auto const& lc = this->virtualStorage.at(n).localCells;
                            std::copy(lc.begin(), lc.end(), std::inserter(lcSet, lcSet.end()));
                        }

                        /// make sure that every commCell in this direction (ghost and real) belongs
                        /// to lcSet
                        if (commReal)
                        {
                            for (Cell* c : fullCommCells[relation].reals)
                            {
                                HPX4ESPP_ASSERT(!(lcSet.find(c) == lcSet.end()));
                            }
                        }
                        else
                        {
                            for (Cell* c : fullCommCells[relation].ghosts)
                            {
                                HPX4ESPP_ASSERT(!(lcSet.find(c) == lcSet.end()));
                            }
                        }
                    };

                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        verifyCommNodesFull(neighbour, fullCommNodesReal[neighbour], true);
                        const auto opposite_neighbour = John::periodic_neighbour[neighbour];

                        verifyCommNodesFull(opposite_neighbour,
                                            fullCommNodesGhost[opposite_neighbour], false);
                    }
                }

                /// verify that commCells in virtual Node match the commCells of the entire
                /// subdomain
                {
                    auto verifyCommCells =
                        [this](size_t dir, std::vector<size_t> const& plane, bool commReal)
                    {
                        auto const& ccr = commCells[dir].reals;
                        auto const& ccg = commCells[dir].ghosts;

                        for (const auto& n : plane)
                        {
                            auto const& vlc = this->virtualStorage.at(n).localCells;
                            auto const& vcc = this->vdd.at(n).commCells;

                            if (commReal)
                            {
                                for (const auto& ic : vcc[dir].reals)
                                {
                                    HPX4ESPP_ASSERT(
                                        !(std::find(ccr.begin(), ccr.end(), vlc[ic]) == ccr.end()));
                                }
                            }
                            else
                            {
                                for (const auto& ic : vcc[dir].ghosts)
                                {
                                    HPX4ESPP_ASSERT(
                                        !(std::find(ccg.begin(), ccg.end(), vlc[ic]) == ccg.end()));
                                }
                            }
                        }
                    };

                    for (size_t coord = 0; coord < 3; ++coord)
                    {
                        for (int lr = 0; lr < 2; ++lr)
                        {
                            size_t const dir = 2 * coord + lr;
                            size_t const oppDir = 2 * coord + (1 - lr);
                            verifyCommCells(dir, commNodesReal[dir], true);
                            verifyCommCells(oppDir, commNodesGhost[oppDir], false);
                        }
                    }
                }

                {
                    auto fullVerifyCommCells = [this](John::NeighbourRelation relation,
                                                      std::vector<size_t> const& relation_arr,
                                                      bool commReal)
                    {
                        auto const& ccr = fullCommCells[relation].reals;
                        auto const& ccg = fullCommCells[relation].ghosts;

                        for (const auto& n : relation_arr)
                        {
                            auto const& vlc = this->virtualStorage.at(n).localCells;
                            auto const& vcc = this->vdd.at(n).fullCommCells;

                            if (commReal)
                            {
                                for (const auto& ic : vcc[relation].reals)
                                {
                                    HPX4ESPP_ASSERT(
                                        !(std::find(ccr.begin(), ccr.end(), vlc[ic]) == ccr.end()));
                                }
                            }
                            else
                            {
                                for (const auto& ic : vcc[relation].ghosts)
                                {
                                    HPX4ESPP_ASSERT(
                                        !(std::find(ccg.begin(), ccg.end(), vlc[ic]) == ccg.end()));
                                }
                            }
                        }
                    };
                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        fullVerifyCommCells(neighbour, fullCommNodesReal[neighbour], true);
                        const auto opposite_neighbour = John::periodic_neighbour[neighbour];

                        fullVerifyCommCells(opposite_neighbour,
                                            fullCommNodesGhost[opposite_neighbour], false);
                    }
                }
            }

            /// For intra-node ghost updates
            {
                // JOHN_OLD (3) : Intra Node updates
                for (size_t coord = 0; coord < 3; ++coord)
                {
                    bool doPeriodic = (nodeGrid.getGridSize(coord) == 1);
                    for (size_t lr = 0; lr < 2; ++lr)
                    {
                        size_t const dir = 2 * coord + lr;

                        subNodePairsIntra[dir].clear();
                        {
                            std::array<size_t, 3> subNodeStart = {0, 0, 0};
                            std::array<size_t, 3> subNodeEnd = subNodeGrid;
                            if (lr == 0)
                            {
                                subNodeStart[coord] += 1;
                            }
                            else
                            {
                                subNodeEnd[coord] -= 1;
                            }
                            SPDLOG_TRACE(
                                "dir {} doPersiodic {} coord {} subnode_start {} subnode_end {}",
                                dir, doPeriodic, coord, spdlog::fmt_lib::join(subNodeStart, ","),
                                spdlog::fmt_lib::join(subNodeEnd, ","));
                            size_t numSubNodePairs = 1;
                            for (size_t i = 0; i < 3; i++)
                                numSubNodePairs *= (subNodeEnd[i] - subNodeStart[i]);
                            for (size_t i2 = subNodeStart[2]; i2 < subNodeEnd[2]; i2++)
                            {
                                for (size_t i1 = subNodeStart[1]; i1 < subNodeEnd[1]; i1++)
                                {
                                    for (size_t i0 = subNodeStart[0]; i0 < subNodeEnd[0]; i0++)
                                    {
                                        size_t sender =
                                            i0 + subNodeGrid[0] * (i1 + subNodeGrid[1] * i2);
                                        size_t recver =
                                            vdd[sender].nodeGridLocal.getNodeNeighborIndex(
                                                dir);  /// global index
                                        subNodePairsIntra[dir].push_back({sender, recver, false});
                                    }
                                }
                            }

                            if (false && !rank)
                            {
                                std::cout << "  coord: " << coord << std::endl;
                                std::cout << "    dir: " << dir << std::endl;
                                std::cout << "      subNodeStart:   ";
                                for (auto const& s : subNodeStart) std::cout << " " << s;
                                std::cout << std::endl;
                                std::cout << "      subNodeEnd:     ";
                                for (auto const& s : subNodeEnd) std::cout << " " << s;
                                std::cout << std::endl;
                                std::cout << "      numSubNodePairs: " << numSubNodePairs
                                          << std::endl;
                            }
                        }

                        if (doPeriodic)
                        {
                            std::array<size_t, 3> subNodeStart = {0, 0, 0};
                            std::array<size_t, 3> subNodeEnd = subNodeGrid;
                            if (lr == 0)
                            {
                                subNodeEnd[coord] = 1;
                            }
                            else
                            {
                                subNodeStart[coord] = subNodeEnd[coord] - 1;
                            }

                            SPDLOG_TRACE(
                                "dir {} IndoPersiodic {} coord {} subnode_start {} subnode_end {}",
                                dir, doPeriodic, coord, spdlog::fmt_lib::join(subNodeStart, ","),
                                spdlog::fmt_lib::join(subNodeEnd, ","));
                            size_t numSubNodePairs = 1;
                            for (size_t i = 0; i < 3; i++)
                                numSubNodePairs *= (subNodeEnd[i] - subNodeStart[i]);
                            for (size_t i2 = subNodeStart[2]; i2 < subNodeEnd[2]; i2++)
                            {
                                for (size_t i1 = subNodeStart[1]; i1 < subNodeEnd[1]; i1++)
                                {
                                    for (size_t i0 = subNodeStart[0]; i0 < subNodeEnd[0]; i0++)
                                    {
                                        size_t sender =
                                            i0 + subNodeGrid[0] * (i1 + subNodeGrid[1] * i2);
                                        size_t recver =
                                            vdd[sender].nodeGridLocal.getNodeNeighborIndex(
                                                dir);  /// global index
                                        subNodePairsIntra[dir].push_back({sender, recver, true});
                                    }
                                }
                            }

                            if (false && !rank)
                            {
                                std::cout << "  coord: " << coord << std::endl;
                                std::cout << "    doPeriodic: " << doPeriodic << std::endl;
                                std::cout << "    dir: " << dir << std::endl;
                                std::cout << "      subNodeStart:   ";
                                for (auto const& s : subNodeStart) std::cout << " " << s;
                                std::cout << std::endl;
                                std::cout << "      subNodeEnd:     ";
                                for (auto const& s : subNodeEnd) std::cout << " " << s;
                                std::cout << std::endl;
                                std::cout << "      numSubNodePairs: " << numSubNodePairs
                                          << std::endl;
                            }
                        }

                        if (false && !rank)
                        {
                            std::cout << "      subNodePairsIntra: ";
                            for (const auto& tup : subNodePairsIntra[dir])
                                std::cout << " (" << std::get<0>(tup) << "," << std::get<1>(tup)
                                          << "," << std::get<2>(tup) << ")";
                            std::cout << std::endl;
                        }
                    }
                }
                // JOHN_NEW (3) : Intra Node updates
                for (const auto& neighbour : John::NeighbourEnumIterable)
                {
                    fullSubNodePairsIntra[neighbour].clear();
                }
                for (const auto& virtual_domain : vdd)
                {
                    const auto virtual_rank = virtual_domain.fullNodeGridLocal.getMyRank();
                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        if (!(virtual_domain.fullNodeGridLocal.is_periodic_neighbour(neighbour)))
                        {
                            fullSubNodePairsIntra[neighbour].push_back(
                                {virtual_rank,
                                 static_cast<std::size_t>(virtual_domain.fullNodeGridLocal
                                                              .get_neighbour_rank(neighbour)),
                                 false});
                        }
                    }
                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        if (full_grid.is_periodic_neighbour(neighbour) &&
                            virtual_domain.fullNodeGridLocal.is_periodic_neighbour(neighbour))
                        {
                            fullSubNodePairsIntra[neighbour].push_back(
                                {virtual_rank,
                                 static_cast<std::size_t>(virtual_domain.fullNodeGridLocal
                                                              .get_neighbour_rank(neighbour)),
                                 true});
                        }
                    }
                }
            }

            /// Validate assumptions about ghost updates in HPX4ESPP_OPTIMIZE_COMM
            // JOHN_OLD (4) Validate assumptions about ghost updates
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                /// list down commCells in the full subdomain
                auto const fill_occ = [lc0, this](auto& occReals, auto& occGhosts)
                {
                    for (size_t dir = 0; dir < 6; dir++)
                    {
                        occReals[dir].clear();
                        for (const auto& c : this->commCells[dir].reals)
                        {
                            HPX4ESPP_ASSERT_GEQ(c, lc0);
                            occReals[dir].insert(c - lc0);
                        }

                        occGhosts[dir].clear();
                        for (const auto& c : this->commCells[dir].ghosts)
                        {
                            HPX4ESPP_ASSERT_GEQ(c, lc0);
                            occGhosts[dir].insert(c - lc0);
                        }
                    }
                };

                std::array<std::set<size_t>, 6> occReals, occGhosts;
                fill_occ(occReals, occGhosts);

                auto removeCommCell = [](auto& occ, auto const cc)
                {
                    const auto find = occ.find(cc);
                    if (find == occ.end()) return 1;
                    occ.erase(find);
                    return 0;
                };

                /// remove entries from inter-node updates
                auto test1 = [&, this](bool REAL_TO_GHOSTS)
                {
                    auto occReals1 = occReals;
                    auto occGhosts1 = occGhosts;

                    for (size_t _coord = 0; _coord < 3; ++_coord)
                    {
                        int coord = REAL_TO_GHOSTS ? _coord : (2 - _coord);
                        const real curCoordBoxL = getSystem()->bc->getBoxL()[coord];
                        const bool doPeriodic = (nodeGrid.getGridSize(coord) == 1);
                        for (size_t lr = 0; lr < 2; ++lr)
                        {
                            size_t const dir = 2 * coord + lr;

                            HPX4ESPP_ASSERT_NEQ(occReals1[dir].size(), 0);
                            HPX4ESPP_ASSERT_NEQ(occGhosts1[dir].size(), 0);

                            size_t const numCommNodes = REAL_TO_GHOSTS ? commNodesReal[dir].size()
                                                                       : commNodesGhost[dir].size();

                            for (size_t ii = 0; ii < numCommNodes; ii++)
                            {
                                const auto& idxCommNode = ii;

                                auto removeCommCells =
                                    [&, this](const bool commReal, auto& occ, const auto occ_label)
                                {
                                    const size_t inode = commReal
                                                             ? commNodesReal[dir][idxCommNode]
                                                             : commNodesGhost[dir][idxCommNode];

                                    const auto& vs = virtualStorage[inode];
                                    const auto& cr = vs.particles.cellRange();
                                    const auto& vd = vdd[inode];
                                    const auto& cc = commReal ? vd.commCells[dir].reals
                                                              : vd.commCells[dir].ghosts;

                                    for (const auto& ic : cc)
                                    {
                                        const auto gidx = vs.localCells[ic] - lc0;
                                        if (HPX4ESPP_OPTIMIZE_COMM)
                                        {
                                            const bool icOwn =
                                                (vd.cellGridInfo[ic].subNode == inode);
                                            if (icOwn)
                                            {
                                                /// remove from list of commCells
                                                if (removeCommCell(occ, gidx))
                                                {
                                                    HPX4ESPP_THROW_EXCEPTION(
                                                        hpx::assertion_failure, __FUNCTION__,
                                                        "ic not found in occ "
                                                            << occ_label << " dir=" << dir
                                                            << " ic=" << ic << " gidx=" << gidx);
                                                }
                                            }
                                        }
                                    }
                                };
                                removeCommCells(true, occReals1[dir], "reals");
                                removeCommCells(false, occGhosts1[dir], "ghosts");
                            }

                            HPX4ESPP_ASSERT_EQUAL(occReals1[dir].size(), 0);
                            HPX4ESPP_ASSERT_EQUAL(occGhosts1[dir].size(), 0);
                        }
                    }
                };
                test1(true);
                test1(false);

                /// make sure there are no intra-node updates that are not in the periodic
                /// boundaries

                //            test2(true);
                //            test2(false);
            }

            // JOHN_NEW (4) Validate assumptions about ghost updates
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                auto const fill_occ = [lc0, this](auto& occReals, auto& occGhosts)
                {
                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        occReals[neighbour].clear();
                        for (const auto& c : this->fullCommCells[neighbour].reals)
                        {
                            HPX4ESPP_ASSERT_GEQ(c, lc0);
                            occReals[neighbour].insert(c - lc0);
                        }

                        occGhosts[neighbour].clear();
                        for (const auto& c : this->fullCommCells[neighbour].ghosts)
                        {
                            HPX4ESPP_ASSERT_GEQ(c, lc0);
                            occGhosts[neighbour].insert(c - lc0);
                        }
                    }
                };
                std::array<std::set<size_t>, 26> occReals, occGhosts;
                fill_occ(occReals, occGhosts);

                auto removeCommCell = [](auto& occ, auto const cc)
                {
                    const auto find = occ.find(cc);
                    if (find == occ.end()) return 1;
                    occ.erase(find);
                    return 0;
                };

                /// remove entries from inter-node updates
                auto test1 = [&, this](bool REAL_TO_GHOSTS)
                {
                    auto occReals1 = occReals;
                    auto occGhosts1 = occGhosts;
                    for (const auto& neighbour : John::NeighbourEnumIterable)
                    {
                        HPX4ESPP_ASSERT_NEQ(occReals1[neighbour].size(), 0);
                        HPX4ESPP_ASSERT_NEQ(occGhosts1[neighbour].size(), 0);
                        size_t const numCommNodes = REAL_TO_GHOSTS
                                                        ? fullCommNodesReal[neighbour].size()
                                                        : fullCommNodesGhost[neighbour].size();
                        for (size_t ii = 0; ii < numCommNodes; ii++)
                        {
                            const auto& idxCommNode = ii;

                            auto removeCommCells =
                                [&, this](const bool commReal, auto& occ, const auto occ_label)
                            {
                                const size_t inode =
                                    commReal ? fullCommNodesReal[neighbour][idxCommNode]
                                             : fullCommNodesGhost[neighbour][idxCommNode];

                                const auto& vs = virtualStorage[inode];
                                const auto& cr = vs.particles.cellRange();
                                const auto& vd = vdd[inode];
                                const auto& cc = commReal ? vd.fullCommCells[neighbour].reals
                                                          : vd.fullCommCells[neighbour].ghosts;

                                for (const auto& ic : cc)
                                {
                                    const auto gidx = vs.localCells[ic] - lc0;
                                    if (HPX4ESPP_OPTIMIZE_COMM)
                                    {
                                        const bool icOwn = (vd.cellGridInfo[ic].subNode == inode);
                                        if (icOwn)
                                        {
                                            /// remove from list of commCells
                                            if (removeCommCell(occ, gidx))
                                            {
                                                HPX4ESPP_THROW_EXCEPTION(
                                                    hpx::assertion_failure, __FUNCTION__,
                                                    "ic not found in occ "
                                                        << occ_label << " neighbour=" << neighbour
                                                        << " ic=" << ic << " gidx=" << gidx);
                                            }
                                        }
                                    }
                                }
                            };
                            removeCommCells(true, occReals1[neighbour], "reals");
                            removeCommCells(false, occGhosts1[neighbour], "ghosts");
                        }

                        HPX4ESPP_ASSERT_EQUAL(occReals1[neighbour].size(), 0);
                        HPX4ESPP_ASSERT_EQUAL(occGhosts1[neighbour].size(), 0);
                    }
                };
                test1(true);
                test1(false);
            }

            /// JOHN_OLD (5): duplicate and filter subNodePairsIntra for each subnode pair
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                bool constexpr REAL_TO_GHOSTS = true;
                for (size_t _coord = 0; _coord < 3; ++_coord)
                {
                    for (size_t lr = 0; lr < 2; ++lr)
                    {
                        int coord = REAL_TO_GHOSTS ? _coord : (2 - _coord);
                        size_t const dir = 2 * coord + lr;

                        subNodePairsIntraPeriodic[dir].clear();

                        size_t const numPairs = subNodePairsIntra[dir].size();
                        for (size_t i = 0; i < numPairs * numCommSubs; i++)
                        {
                            const size_t ip = i / numCommSubs;
                            const size_t is = i % numCommSubs;

                            const auto& pair = subNodePairsIntra[dir][ip];
                            if (std::get<2>(pair))
                            {
                                subNodePairsIntraPeriodic[dir].push_back(pair);
                            }
                        }

                        // std::cout << " dir=" << dir << " numPairs=" << numPairs
                        //           << " subNodePairsIntraPeriodic[dir].size()="
                        //           << subNodePairsIntraPeriodic[dir].size() << std::endl;
                    }
                }
            }

            /// JOHN_NEW (5): duplicate and filter subNodePairsIntra for each subnode pair
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                for (const auto& neighbour : John::NeighbourEnumIterable)
                {
                    fullSubNodePairsIntraPeriodic[neighbour].clear();
                    size_t const numPairs = fullSubNodePairsIntra[neighbour].size();
                    for (size_t i = 0; i < numPairs * numCommSubs; i++)
                    {
                        const size_t ip = i / numCommSubs;
                        const size_t is = i % numCommSubs;

                        const auto& pair = fullSubNodePairsIntra[neighbour][ip];
                        if (std::get<2>(pair))
                        {
                            fullSubNodePairsIntraPeriodic[neighbour].push_back(pair);
                        }
                    }
                }
            }

            /// JOHN_OLD (6) : fill in vdd.commCellsOwned with correct entries
            /// NOTE: works only if numCommSubs==1
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                HPX4ESPP_ASSERT_EQUAL_MSG(numCommSubs, 1,
                                          "Use of vdd.commCellsOwned works only for "
                                          "numCommSubs=1");

                bool constexpr REAL_TO_GHOSTS = true;
                for (size_t _coord = 0; _coord < 3; ++_coord)
                {
                    for (size_t lr = 0; lr < 2; ++lr)
                    {
                        int coord = REAL_TO_GHOSTS ? _coord : (2 - _coord);
                        size_t const dir = 2 * coord + lr;

                        const auto& pairs = subNodePairsIntraPeriodic[dir];
                        size_t const numPairs = pairs.size();
                        for (size_t i = 0; i < numPairs; i++)
                        {
                            const size_t ip = i;
                            const size_t is = 0;

                            const auto& pair = pairs[ip];
                            HPX4ESPP_ASSERT_EQUAL(std::get<2>(pair), 1);

                            const size_t ir = std::get<0>(pair);
                            const size_t ig = std::get<1>(pair);

                            const auto& ccrOrig = vdd[ir].commCells[dir].reals;
                            const auto& ccgOrig = vdd[ig].commCells[dir].ghosts;

                            HPX4ESPP_ASSERT_EQUAL(ccrOrig.size(), ccgOrig.size());
                            const size_t numCells = ccrOrig.size();

                            auto& ccrOwn = vdd[ir].commCellsOwned[dir].reals;
                            auto& ccgOwn = vdd[ig].commCellsOwned[dir].ghosts;

                            for (size_t ic = 0; ic < numCells; ic++)
                            {
                                const size_t icr = ccrOrig[ic];
                                const size_t icg = ccgOrig[ic];

                                const bool irOwn = (vdd[ir].cellGridInfo[icr].subNode == ir);
                                const bool igOwn = (vdd[ig].cellGridInfo[icg].subNode == ig);

                                HPX4ESPP_ASSERT_EQUAL(irOwn, igOwn);

                                if (!igOwn) continue;

                                ccrOwn.push_back(icr);
                                ccgOwn.push_back(icg);
                            }
                        }
                    }
                }
            }

            /// JOHN_NEW (6) : fill in vdd.commCellsOwned with correct entries
            /// NOTE: works only if numCommSubs==1
            if (HPX4ESPP_OPTIMIZE_COMM)
            {
                for (const auto& neighbour : John::NeighbourEnumIterable)
                {
                    const auto& pairs = fullSubNodePairsIntraPeriodic[neighbour];
                    size_t const numPairs = pairs.size();
                    for (const auto& [ir, ig, isPeriodic] : pairs)
                    {
                        const auto& ccrOrig = vdd[ir].fullCommCells[neighbour].reals;
                        const auto& ccgOrig = vdd[ig].fullCommCells[neighbour].ghosts;

                        HPX4ESPP_ASSERT_EQUAL(isPeriodic, true);

                        HPX4ESPP_ASSERT_EQUAL(ccrOrig.size(), ccgOrig.size());
                        const size_t numCells = ccrOrig.size();

                        auto& ccrOwn = vdd[ir].fullCommCellsOwned[neighbour].reals;
                        auto& ccgOwn = vdd[ig].fullCommCellsOwned[neighbour].ghosts;

                        for (size_t ic = 0; ic < numCells; ic++)
                        {
                            const size_t icr = ccrOrig[ic];
                            const size_t icg = ccgOrig[ic];

                            const bool irOwn = (vdd[ir].cellGridInfo[icr].subNode == ir);
                            const bool igOwn = (vdd[ig].cellGridInfo[icg].subNode == ig);

                            //                            SPDLOG_TRACE("irOwn {} igOwn {} neighbour
                            //                            {}", irOwn, igOwn, neighbour);
                            //                            HPX4ESPP_ASSERT_EQUAL(irOwn, igOwn);

                            if (!igOwn) continue;

                            ccrOwn.push_back(icr);
                            ccgOwn.push_back(icg);
                        }
                    }
                }
            }
            /// Fill out resortWaves
            {
                /// Requires subCellGrid to be at least twice the size of halfCellInt to
                /// allow non-overlapping cells to be resorted concurrently
                for (size_t i = 0; i < 3; i++)
                {
                    HPX4ESPP_ASSERT_GEQ((subCellGrid[i]), (2 * halfCellInt));
                }

                /// Group subNodes into "waves"
                resortWaves.clear();
                resortWaves.resize(8);
                for (size_t i2 = 0; i2 < subNodeGrid[2]; i2++)
                {
                    const size_t r2 = i2 % 2;
                    for (size_t i1 = 0; i1 < subNodeGrid[1]; i1++)
                    {
                        const size_t r1 = i1 % 2;
                        for (size_t i0 = 0; i0 < subNodeGrid[0]; i0++)
                        {
                            const size_t r0 = i0 % 2;
                            const size_t rwidx = r0 + 2 * (r1 + 2 * r2);
                            const size_t myidx = i0 + subNodeGrid[0] * (i1 + subNodeGrid[1] * i2);
                            resortWaves[rwidx].push_back(myidx);
                        }
                    }
                }

                /// Verification
                if (false)  // ENABLE_CHECK
                {
                    std::vector<size_t> resortWavesAssigned;
                    resortWavesAssigned.resize(numSubNodes);
                    for (size_t iwave = 0; iwave < resortWaves.size(); iwave++)
                    {
                        const auto& wave = resortWaves[iwave];
                        for (const auto& inode : wave)
                        {
                            resortWavesAssigned[inode] = iwave;
                        }
                    }

                    if (!rank)
                    {
                        std::cout << "resortWavesAssigned: ";
                        for (const auto& w : resortWavesAssigned) std::cout << " " << w;
                        std::cout << std::endl;
                    }

                    /// Check adjacents
                    for (std::int64_t i2 = 0; i2 < subNodeGrid[2]; i2++)
                    {
                        for (std::int64_t i1 = 0; i1 < subNodeGrid[1]; i1++)
                        {
                            for (std::int64_t i0 = 0; i0 < subNodeGrid[0]; i0++)
                            {
                                const std::int64_t myidx =
                                    i0 + subNodeGrid[0] * (i1 + subNodeGrid[1] * i2);
                                const auto mywave = resortWavesAssigned[myidx];

                                for (std::int64_t a2 = -1; a2 <= +1; a2++)
                                {
                                    const std::int64_t n2 = i2 + a2;
                                    if (n2 < 0 || n2 >= subNodeGrid[2]) continue;
                                    for (std::int64_t a1 = -1; a1 <= +1; a1++)
                                    {
                                        const std::int64_t n1 = i1 + a1;
                                        if (n1 < 0 || n1 >= subNodeGrid[1]) continue;
                                        for (std::int64_t a0 = -1; a0 <= +1; a0++)
                                        {
                                            const std::int64_t n0 = i0 + a0;
                                            if (n0 < 0 || n0 >= subNodeGrid[0]) continue;
                                            if (a0 == 0 && a1 == 0 && a2 == 0) continue;

                                            const std::int64_t nidx =
                                                n0 + subNodeGrid[0] * (n1 + subNodeGrid[1] * n2);
                                            HPX4ESPP_ASSERT_LT(nidx, numSubNodes);
                                            const auto nwave = resortWavesAssigned[nidx];

                                            HPX4ESPP_ASSERT_NEQ(mywave, nwave);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /// Prepare resort using hpx par-for
            {
                for (size_t inode = 0; inode < vdd.size(); inode++)
                {
                    auto& vs = virtualStorage[inode];
                    auto& vd = vdd[inode];
                    const size_t numLocals = vs.localCells.size();
                    const size_t numReals = vs.realCellsIdx.size();
                    const size_t numGhosts = numLocals - numReals;

                    vd.vGhostCells.clear();
                    vd.vGhostCells.resize(numGhosts);

                    vd.vLocalCells.clear();
                    vd.vLocalCells.reserve(numLocals);

                    vd.vRealCells.clear();
                    vd.vRealCells.reserve(numReals);

                    size_t ghostCtr = 0, ctr = 0;
                    for (size_t j2 = 0, j2end = subCellGrid[2] + 2 * halfCellInt; j2 < j2end; j2++)
                    {
                        const bool r2 =
                            (halfCellInt <= j2) && (j2 < (subCellGrid[2] + halfCellInt));
                        for (size_t j1 = 0, j1end = subCellGrid[1] + 2 * halfCellInt; j1 < j1end;
                             j1++)
                        {
                            const bool r1 =
                                (halfCellInt <= j1) && (j1 < (subCellGrid[1] + halfCellInt));
                            for (size_t j0 = 0, j0end = subCellGrid[0] + 2 * halfCellInt;
                                 j0 < j0end; j0++)
                            {
                                const bool r0 =
                                    (halfCellInt <= j0) && (j0 < (subCellGrid[0] + halfCellInt));

                                if (r0 && r1 && r2)
                                {
                                    Cell* realCell = vs.localCells[ctr];
                                    vd.vLocalCells.push_back(realCell);
                                    vd.vRealCells.push_back(realCell);
                                }
                                else
                                {
                                    /// push_back pointer to ghost cell cell
                                    vd.vLocalCells.push_back(&(vd.vGhostCells[ghostCtr++]));
                                }
                                ctr++;
                            }
                        }
                    }

                    HPX4ESPP_ASSERT_EQUAL(vd.vLocalCells.size(), numLocals);
                    HPX4ESPP_ASSERT_EQUAL(vd.vRealCells.size(), numReals);

                    if (false && !rank)
                    {
                        std::cout << "inode"
                                  << " : " << inode << "\n"
                                  << "  numLocals"
                                  << " : " << numLocals << "\n"
                                  << "  numReals "
                                  << " : " << numReals << "\n"
                                  << "  numGhosts"
                                  << " : " << numGhosts << std::endl;
                    }
                }

                /// JOHN_OLD (6): node and cell sizes and ranges for inter-node
                /// NOTE: assuming uniform subNode division
                for (int coord = 0; coord < 3; ++coord)
                {
                    for (int lr = 0; lr < 2; ++lr)
                    {
                        const int dir = 2 * coord + lr;
                        const int oppDir = 2 * coord + (1 - lr);
                        nodeCommCellRange[dir].clear();
                        HPX4ESPP_ASSERT_EQUAL(commNodesReal[dir].size(),
                                              commNodesGhost[dir].size());
                        nodeCommCellRange[dir].reserve(commNodesGhost[dir].size() + 1);
                        const int numCommNodes = commNodesGhost[dir].size();
                        size_t total = 0;
                        for (int idxCommNode = 0; idxCommNode < numCommNodes; idxCommNode++)
                        {
                            nodeCommCellRange[dir].push_back(total);
                            const size_t inode = commNodesGhost[dir][idxCommNode];
                            total += vdd[inode].commCells[dir].ghosts.size();

                            HPX4ESPP_ASSERT_EQUAL(vdd[inode].commCells[dir].reals.size(),
                                                  vdd[inode].commCells[dir].ghosts.size());
                            HPX4ESPP_ASSERT_EQUAL(vdd[inode].commCells[dir].reals.size(),
                                                  vdd[inode].commCells[oppDir].ghosts.size());
                        }
                        nodeCommCellRange[dir].push_back(total);

                        if (false && !rank)
                        {
                            std::cout << rank << ": nodeCommCellRange[" << dir << "]:  ";
                            for (auto r : nodeCommCellRange[dir]) std::cout << " " << r;
                            std::cout << std::endl;
                        }
                    }
                }
                /// JOHN_NEW (6): node and cell sizes and ranges for inter-node
                for (const auto& neighbour : John::NeighbourEnumIterable)
                {
                    fullNodeCommCellRange[neighbour].clear();
                    HPX4ESPP_ASSERT_EQUAL(fullCommNodesReal[neighbour].size(),
                                          fullCommNodesGhost[neighbour].size());

                    fullNodeCommCellRange[neighbour].reserve(
                        fullNodeCommCellRange[neighbour].size() + 1);
                    std::size_t total = 0;
                    const auto opposite_neighbour = John::periodic_neighbour[neighbour];
                    const auto numCommNodes = fullCommNodesGhost[neighbour].size();

                    for (auto idxCommNode = 0; idxCommNode < numCommNodes; idxCommNode++)
                    {
                        fullNodeCommCellRange[neighbour].push_back(total);
                        const size_t inode = fullCommNodesGhost[neighbour][idxCommNode];
                        total += vdd[inode].fullCommCells[neighbour].ghosts.size();

                        HPX4ESPP_ASSERT_EQUAL(vdd[inode].fullCommCells[neighbour].reals.size(),
                                              vdd[inode].fullCommCells[neighbour].ghosts.size());
                        HPX4ESPP_ASSERT_EQUAL(
                            vdd[inode].fullCommCells[neighbour].reals.size(),
                            vdd[inode].fullCommCells[opposite_neighbour].ghosts.size());
                    }
                    fullNodeCommCellRange[neighbour].push_back(total);
                }
            }
            SPDLOG_TRACE("took {}", sw);
            HPX4ESPP_DEBUG_MSG("FullDomainDecomposition::resetVirtualStorage()");
        }
    }
    //
}

void FullDomainDecomposition::initChannels()
{
    /// Initialize channel-based ghost updates
    /// Setup channels between adjacent non-local neighboring virtual subdomains
    /// Initial try: use a channel for buffer exchange between full subdomains
    HPX4ESPP_ASSERT(commUseChannels);
    //    HPX4ESPP_ASSERT(HPXRuntime::isRunning());
}

python::object FullDomainDecomposition::getChannelIndices() const
{
    HPX4ESPP_ASSERT(channelsInit);
    return exchange_ghost_channels.getChannelIndices();
}

void FullDomainDecomposition::loadCells()
{
    /// load new particle data (real and ghost cells)
    auto f = [this](size_t i)
    {
        auto& vs = this->virtualStorage[i];
        if (HPX4ESPP_OPTIMIZE_COMM)
            vs.particles.copyFromCellOwn(vs.localCells);
        else
            vs.particles.copyFrom(vs.localCells);
    };
    const size_t nvs = virtualStorage.size();
    {
        const real time = wallTimer.getElapsedTime();
        utils::parallelForLoop(0, nvs, f);
        timeLoadLoop += wallTimer.getElapsedTime() - time;
    }
    {
        const real time = wallTimer.getElapsedTime();
        prepareGhostBuffers();
        timeLoadPrepGhost += wallTimer.getElapsedTime() - time;
    }
    {
        const real time = wallTimer.getElapsedTime();
        onLoadCells();
        timeLoadSignal += wallTimer.getElapsedTime() - time;
    }
}

void FullDomainDecomposition::unloadCells()
{
    auto f = [this](size_t i)
    {
        auto& vs = this->virtualStorage[i];
        vs.particles.updateToPositionVelocity(vs.localCells, true);
        /// only real cells
    };
    const size_t nvs = virtualStorage.size();
    {
        const real time = wallTimer.getElapsedTime();
        utils::parallelForLoop(0, nvs, f);
        timeUnloadLoop += wallTimer.getElapsedTime() - time;
    }
    {
        const real time = wallTimer.getElapsedTime();
        onUnloadCells();
        timeUnloadSignal += wallTimer.getElapsedTime() - time;
    }
}

#define HPX4ESPP_PGB_DEBUG(STATEMENT) STATEMENT
// #define HPX4ESPP_PGB_DEBUG(STATEMENT)

void FullDomainDecomposition::prepareGhostBuffers()
// TODO:Replace with your function
// TODO: on particle changed signal modify
{
    maxReal = maxGhost = 0;
    fullMaxReal = fullMaxGhost = 0;
    for (size_t coord = 0; coord < 3; ++coord)
    {
        for (size_t lr = 0; lr < 2; ++lr)
        {
            size_t const dir = 2 * coord + lr;
            size_t const oppDir = 2 * coord + (1 - lr);

            auto f_count_particles =
                [this](bool commReals, size_t dir, std::vector<size_t>& nodeRange)
            {
                // bool commReals = true;
                std::vector<size_t> const& cn =
                    commReals ? commNodesReal[dir] : commNodesGhost[dir];  // What is that
                nodeRange.resize(cn.size() + 1);
                size_t dirTotal = 0;
                for (size_t in = 0; in < cn.size(); in++)
                {
                    const size_t& inode = cn[in];
                    const auto& vs = virtualStorage[inode];
                    const auto& cr = vs.particles.cellRange();
                    const auto& vd = vdd[inode];

                    const auto& cc = commReals ? vd.commCells[dir].reals : vd.commCells[dir].ghosts;
                    size_t nodeTotal = 0;
                    for (const auto& ic : cc)
                    {
                        nodeTotal += (cr[ic + 1] - cr[ic]);
                    }
                    nodeRange[in] = dirTotal;
                    dirTotal += nodeTotal;
                }
                nodeRange[cn.size()] = dirTotal;
            };

            f_count_particles(true, dir, nodeRangeReal[dir]);
            f_count_particles(false, oppDir, nodeRangeGhost[oppDir]);
            /// determine maximum number of particles
            maxReal = std::max(maxReal, nodeRangeReal[dir].back());
            maxGhost = std::max(maxGhost, nodeRangeGhost[oppDir].back());

            if (false)  // ENABLE_CHECK
            {
                const size_t dirTotalReal = nodeRangeReal[dir].back();
                const size_t dirTotalGhost = nodeRangeGhost[oppDir].back();
                std::cout << "  dir: " << dir << " dirTotalReal: " << dirTotalReal
                          << " nodeRangeReal[dir]:";
                for (const auto& p : nodeRangeReal[dir])
                {
                    std::cout << " " << p;
                }
                std::cout << std::endl;
                std::cout << "  oppDir: " << oppDir << " dirTotalGhost: " << dirTotalGhost
                          << " nodeRangeGhost[oppDir]:";
                for (const auto& p : nodeRangeGhost[oppDir])
                {
                    std::cout << " " << p;
                }
                std::cout << std::endl;
            }
        }
    }

    for (const auto& neighbour : John::NeighbourEnumIterable)
    {
        auto full_f_count_particles =
            [this, &neighbour](bool commReals, size_t dir, std::vector<size_t>& nodeRange)
        {
            // bool commReals = true;
            std::vector<size_t> const& cn = commReals
                                                ? fullCommNodesReal[neighbour]
                                                : fullCommNodesGhost[neighbour];  // What is that
            nodeRange.resize(cn.size() + 1);
            size_t dirTotal = 0;
            for (size_t in = 0; in < cn.size(); in++)
            {
                const size_t& inode = cn[in];
                const auto& vs = virtualStorage[inode];
                const auto& cr = vs.particles.cellRange();
                const auto& vd = vdd[inode];

                const auto& cc = commReals ? vd.fullCommCells[neighbour].reals
                                           : vd.fullCommCells[neighbour].ghosts;
                size_t nodeTotal = 0;
                for (const auto& ic : cc)
                {
                    nodeTotal += (cr[ic + 1] - cr[ic]);
                }
                nodeRange[in] = dirTotal;
                dirTotal += nodeTotal;
            }
            nodeRange[cn.size()] = dirTotal;
        };
        const auto opposite = opposite_neighbour(neighbour);
        full_f_count_particles(true, neighbour, fullNodeRangeReal[neighbour]);
        full_f_count_particles(false, opposite, fullNodeRangeGhost[opposite]);

        fullMaxReal = std::max(fullMaxReal, fullNodeRangeReal[neighbour].back());
        fullMaxGhost = std::max(fullMaxGhost, fullNodeRangeGhost[opposite].back());
    }

    /// size of each entry for each particle
    {
        /// force exchange (AOS: 4, SOA: 3)
        /// position (AOS: 4, SOA: 3)
        const size_t preallocReal = maxReal * vecModeFactor;
        const size_t preallocGhost = maxGhost * vecModeFactor;

        fullPreallocReal = fullMaxReal * vecModeFactor;
        fullPreallocGhost = fullMaxGhost * vecModeFactor;
        if (buffReal.size() < preallocReal) buffReal.resize(preallocReal);
        if (buffGhost.size() < preallocGhost) buffGhost.resize(preallocGhost);
    }

    if (false)  // ENABLE_CHECK
    {
        std::cout << " maxReal: " << maxReal << " maxGhost: " << maxGhost << std::endl;
        std::cout << " buffReal.size(): " << buffReal.size()
                  << " buffGhost.size(): " << buffGhost.size() << std::endl;
    }
}

void FullDomainDecomposition::updateGhostsBlocking() { ghostCommunication_impl<false, true, 0>(); }

void FullDomainDecomposition::collectGhostForcesBlocking()
{
    ghostCommunication_impl<false, false, 0>();
}

void FullDomainDecomposition::connect()
{
    if (!sigResetVirtualStorage.connected())
        sigResetVirtualStorage = this->onCellAdjust.connect(
            boost::signals2::at_back,
            boost::bind(&FullDomainDecomposition::resetVirtualStorage, this));
    HPX4ESPP_DEBUG_MSG("FullDomainDecomposition::connect()");
}

void FullDomainDecomposition::disconnect()
{
    if (sigResetVirtualStorage.connected()) sigResetVirtualStorage.disconnect();
    HPX4ESPP_DEBUG_MSG("FullDomainDecomposition::disconnect()");
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Members extending base FullDomainDecomposition class
void FullDomainDecomposition::decomposeHPX()
{
    // std::cout << "Called " << __FUNCTION__ << std::endl;
    // baseClass::decompose();
    {
        /// WARNING: Makes localParticles incorrect
        // baseClass::invalidateGhosts();
    } {
        const real time = wallTimer.getElapsedTime();

        if (decompUseParFor)  // is True for now see make_hpx_system
        {
            decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic();
        }
        else
        {
            decomposeRealParticlesHPXCellTask_MultiNode_NonPeriodic();
        }

        // baseClass::decomposeRealParticles();
        timeDecomposeReal += wallTimer.getElapsedTime() - time;
    }
    {
        const real time = wallTimer.getElapsedTime();
        // baseClass::exchangeGhosts();
        {
            if (excgAligned)
                exchangeGhosts_impl<true>();
            else
                exchangeGhosts_impl<false>();  // is false see code
        }

        timeDecomposeExcGhosts += wallTimer.getElapsedTime() - time;
    }
    {
        const real time = wallTimer.getElapsedTime();
        baseClass::onParticlesChanged();
        timeDecomposeSignal += wallTimer.getElapsedTime() - time;
        /// NOTE: vectorized rebuild should not be invoked here but in onLoadCells()
    }
}

// Is not used yet..
void FullDomainDecomposition::decomposeRealParticlesHPXCellTask_MultiNode_NonPeriodic()
{
    // std::cout << getSystem()->comm->rank() << ": " << "
    // decomposeRealParticlesHPXCellTask_MultiNode_NonPeriodic useDecomposeHPX="<<
    // useDecomposeHPX
    // << std::endl;

    const int DD_COMM_TAG = 0xab;

    ParticleList sendBuf, recvBuf;
    sendBuf.reserve(exchangeBufferSize);
    recvBuf.reserve(exchangeBufferSize);

    const size_t numRealCells = realCells.size();
    const size_t numGhostCells = ghostCells.size();
    const size_t numLocalCells = localCells.size();

    {
        auto moveIndexedParticleNonGlobal = [this](ParticleList& dl, ParticleList& sl, int i)
        {
            dl.push_back(sl[i]);
            const int newSize = sl.size() - 1;
            if (i != newSize)
            {
                sl[i] = sl.back();
            }
            sl.resize(newSize);
        };

        const real* cellSize = cellGrid.getCellSize();
        const real* invCellSize = cellGrid.getInverseCellSize();
        const int frame = cellGrid.getFrameWidth();
        auto mapPositionToLocalCell = [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
        {
            int cpos[3];

            for (int i = 0; i < 3; ++i)
            {
                // check that all particles are still within cellGrid
                if (false)  // ENABLE_CHECK
                {
                    const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                    const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                    if (pos[i] < myLeft || pos[i] >= myRight)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "",
                                                 "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                           << "," << myRight << ")");
                }

                real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                if (false)  // ENABLE_CHECK
                {
                    const int gridSize = cellGrid.getFrameGridSize(i);
                    if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                        HPX4ESPP_THROW_EXCEPTION(
                            hpx::assertion_failure, "",
                            "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
                }
            }
            return localCells[cellGrid.mapPositionToIndex(cpos)];
        };

        espressopp::bc::BC const& boundary = *(getSystem()->bc);
        boost::mpi::communicator const& comm = *(getSystem()->comm);
        auto resortRealCell =
            [this, &moveIndexedParticleNonGlobal, &mapPositionToLocalCell, &boundary](size_t ic)
        {
            Cell& cell = *realCells[ic];
            // do not use an iterator here, since we have need to take out particles during the
            // loop
            for (size_t p = 0; p < cell.particles.size(); ++p)
            {
                Particle& part = cell.particles[p];

                {
                    Cell* sortCell = mapPositionToLocalCell(part.position());

                    if (sortCell != &cell)
                    {
                        if (sortCell == 0)
                        {
                            HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "resortRealCell",
                                                     "sortCell == 0 not allowed");
                        }
                        else
                        {
                            moveIndexedParticleNonGlobal(sortCell->particles, cell.particles, p);
                            --p;
                        }
                    }
                }
            }
        };

        /// clear ghost cells and use as outgoing buffer cells
        // for(const auto cell: getGhostCells()) cell->particles.clear();
        for (size_t ig = 0; ig < numGhostCells; ig++)
        {
            ghostCells[ig]->particles.clear();
        }

        /// extra check to ensure ghost cells are empty
        if (false)  // ENABLE_CHECK
        {
            for (const auto cell : getGhostCells())
            {
                if (!(cell->particles.empty()))
                {
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure,
                        "decomposeRealParticlesHPXCellTask_SingleNode_NonPeriodic",
                        "ghost cell not empty. size = " << cell->particles.size());
                }
            }
        }

        real time = wallTimer.getElapsedTime();
        if (utils::isInsideHPXThread())
        {
            for (auto const& wave : resortWaves)
            {
                auto resortSubNode = [this, &wave, &resortRealCell](size_t inode)
                {
                    const auto nodeIdx = wave[inode];
                    /// list of real cells in this node
                    const auto& realCellsIdx = virtualStorage[nodeIdx].realCellsIdx;
                    for (const auto& ir : realCellsIdx) resortRealCell(ir);
                };
                const size_t numWaveNodes = wave.size();
                utils::parallelForLoop(0, numWaveNodes, resortSubNode);
            }
        }
        else
        {
            for (size_t ic = 0; ic < numRealCells; ic++) resortRealCell(ic);
        }
        timeDecomposeRealResort += wallTimer.getElapsedTime() - time;

        if (false)  // ENABLE_CHECK
        {
            /// TODO: check that every particle belongs to the correct cell
            Cell* c0 = localCells[0];
            for (int ic = 0; ic < numLocalCells; ic++)
            {
                Cell* currCell = localCells[ic];
                ParticleList& particles = currCell->particles;
                for (Particle& part : particles)
                {
                    Cell* sortCell = mapPositionToLocalCell(part.position());
                    if (sortCell != currCell)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,
                                                 "FullDomainDecomposition::"
                                                 "decomposeRealParticlesHPXCellTask_SingleNode_"
                                                 "NonPeriodic",
                                                 "sortCell!=currCell");
                }
            }

            /// TODO: add test to check that particles remain in the left-frame to right+frame
            /// volume
        }

        time = wallTimer.getElapsedTime();
        /// Move ghost cell particles to destination cells
        const bool realToGhosts = false;
        for (int _coord = 0; _coord < 3; ++_coord)
        {
            int coord = realToGhosts ? _coord : (2 - _coord);
            // real curCoordBoxL = boundary.getBoxL()[coord];

            // lr loop: left right
            for (int lr = 0; lr < 2; ++lr)
            {
                int dir = 2 * coord + lr;
                int oppositeDir = 2 * coord + (1 - lr);

                if (nodeGrid.getGridSize(coord) == 1)
                {
                    const int numCells = commCells[dir].ghosts.size();
                    auto moveGhostsToReals = [this, coord, dir, &boundary](int i)
                    {
                        ParticleList& reals = commCells[dir].reals[i]->particles;
                        ParticleList& ghosts = commCells[dir].ghosts[i]->particles;

                        reals.reserve(reals.size() + ghosts.size());
                        for (Particle& part : ghosts)
                        {
                            boundary.foldCoordinate(part.position(), part.image(), coord);

                            reals.push_back(part);
                        }
                        /// Delete particles from ghost cells
                        ghosts.clear();
                    };
                    utils::parallelForLoop(0, numCells, moveGhostsToReals);
                }
                else
                {
                    /// TODO: Calculate send/recv ranges to parallelize packing/unpacking
                    std::vector<longint> sendSizes, recvSizes;
                    std::vector<longint> sendRanges, recvRanges;
                    longint totalSend = 0, totalRecv = 0;
                    {
                        // prepare buffers
                        sendSizes.reserve(commCells[dir].ghosts.size());
                        for (int i = 0, end = commCells[dir].ghosts.size(); i < end; ++i)
                        {
                            sendSizes.push_back(commCells[dir].ghosts[i]->particles.size());
                        }
                        recvSizes.resize(commCells[dir].reals.size());

                        // exchange sizes, odd-even rule
                        if (nodeGrid.getNodePosition(coord) % 2 == 0)
                        {
                            LOG4ESPP_DEBUG(logger,
                                           "sending to node "
                                               << nodeGrid.getNodeNeighborIndex(dir)
                                               << ", then receiving from node "
                                               << nodeGrid.getNodeNeighborIndex(oppositeDir));
                            hpx::util::format("DD 2601 Sending to {} recieveing from {}",
                                              nodeGrid.getNodeNeighborIndex(dir),
                                              nodeGrid.getNodeNeighborIndex(oppositeDir));
                            comm.send(nodeGrid.getNodeNeighborIndex(oppositeDir), DD_COMM_TAG,
                                      sendSizes.data(), sendSizes.size());
                            comm.recv(nodeGrid.getNodeNeighborIndex(dir), DD_COMM_TAG,
                                      recvSizes.data(), recvSizes.size());
                        }
                        else
                        {
                            LOG4ESPP_DEBUG(logger, "receiving from node "
                                                       << nodeGrid.getNodeNeighborIndex(oppositeDir)
                                                       << ", then sending to node "
                                                       << nodeGrid.getNodeNeighborIndex(dir));
                            comm.recv(nodeGrid.getNodeNeighborIndex(dir), DD_COMM_TAG,
                                      recvSizes.data(), recvSizes.size());
                            comm.send(nodeGrid.getNodeNeighborIndex(oppositeDir), DD_COMM_TAG,
                                      sendSizes.data(), sendSizes.size());
                        }

                        sendRanges.reserve(commCells[dir].ghosts.size());
                        for (longint s : sendSizes)
                        {
                            sendRanges.push_back(totalSend);
                            totalSend += s;
                        }
                        recvRanges.reserve(commCells[dir].reals.size());
                        for (longint r : recvSizes)
                        {
                            recvRanges.push_back(totalRecv);
                            totalRecv += r;
                        }

                        LOG4ESPP_DEBUG(logger, "exchanging ghost cell sizes done");
                    }

                    sendBuf.clear();
                    sendBuf.resize(totalSend);
                    recvBuf.resize(totalRecv);
                    {
                        const int numCells = commCells[dir].ghosts.size();
                        auto f_pack = [dir, &sendRanges, &sendSizes, &sendBuf, this](int i)
                        {
                            ParticleList& ghosts = commCells[dir].ghosts[i]->particles;
                            longint const& start = sendRanges[i];
                            longint const& size = sendSizes[i];
                            for (int ig = 0, ip = start; ig < size; ip++, ig++)
                            {
                                sendBuf[ip] = ghosts[ig];
                            }
                            ghosts.clear();
                        };
                        /// NOTE: too few particles to parallelize
                        // utils::parallelForLoop(0, numCells, f_pack);
                        for (int i = 0; i < numCells; ++i) f_pack(i);

                        if (sendBuf.size() != totalSend)
                            HPX4ESPP_THROW_EXCEPTION(
                                hpx::assertion_failure,
                                "decomposeRealParticlesHPXCellTask_MultiNode_NonPeriodic",
                                "size mismatch sendBuf.size()!=totalSend");
                    }

                    const longint receiver = nodeGrid.getNodeNeighborIndex(oppositeDir);
                    const longint sender = nodeGrid.getNodeNeighborIndex(dir);

                    // exchange particles, odd-even rule
                    if (nodeGrid.getNodePosition(coord) % 2 == 0)
                    {
                        comm.send(receiver, DD_COMM_TAG, sendBuf.data(), sendBuf.size());
                        comm.recv(sender, DD_COMM_TAG, recvBuf.data(), recvBuf.size());
                    }
                    else
                    {
                        comm.recv(sender, DD_COMM_TAG, recvBuf.data(), recvBuf.size());
                        comm.send(receiver, DD_COMM_TAG, sendBuf.data(), sendBuf.size());
                    }

                    {
                        const int numCells = commCells[dir].reals.size();
                        // int ctr = 0;
                        auto f_unpack =
                            [dir, coord, &recvRanges, &recvSizes, &recvBuf, &boundary, this](int i)
                        {
                            ParticleList& reals = commCells[dir].reals[i]->particles;
                            const int start = recvRanges[i];
                            const int size = recvSizes[i];
                            const int end = start + size;
                            reals.reserve(reals.size() + size);
                            for (int ip = start; ip < end; ip++)
                            {
                                Particle& part = recvBuf[ip];
                                boundary.foldCoordinate(part.position(), part.image(), coord);
                                reals.push_back(part);
                            }
                        };
                        /// NOTE: too few particles to parallelize
                        // utils::parallelForLoop(0, numCells, f_unpack);
                        for (int i = 0; i < numCells; ++i) f_unpack(i);
                    }
                }
            }
        }
        timeDecomposeRealComm += wallTimer.getElapsedTime() - time;

        /// check that all particles ended up in the correct cell
        if (false)  // ENABLE_CHECK
        {
            /// check that every particle belongs to the correct cell
            Cell* c0 = localCells[0];
            for (int ic = 0; ic < numLocalCells; ic++)
            {
                Cell* currCell = localCells[ic];
                ParticleList& particles = currCell->particles;
                for (Particle& part : particles)
                {
                    Cell* sortCell = mapPositionToLocalCell(part.position());
                    if (sortCell != currCell)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,
                                                 "FullDomainDecomposition::"
                                                 "decomposeRealParticlesHPXCellTask_SingleNode_"
                                                 "NonPeriodic",
                                                 "sortCell!=currCell");
                }
            }
        }

        /// check that ghost cells are empty
        if (false)  // ENABLE_CHECK
        {
            for (size_t ig = 0; ig < numGhostCells; ig++)
            {
                ParticleList& ghosts = ghostCells[ig]->particles;
                if (ghosts.size() > 0)
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure,
                        "FullDomainDecomposition::decomposeRealParticlesHPXCellTask_SingleNode_"
                        "NonPeriodic",
                        "ghost cell must be empty. ghosts.size()=" << ghosts.size());
            }
        }
    }
}

// #define HPX4ESPP_DDPF_DEBUG(STATEMENT) STATEMENT
#define HPX4ESPP_DDPF_DEBUG(STATEMENT)

void FullDomainDecomposition::decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic()
{
    if (decomposeRealParticlesHPXParFor_john)
    {
        decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_full();
    }
    else
    {
        decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_six();
    }
}

void FullDomainDecomposition::decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_six()
{
    auto const comm = *(getSystem()->comm);
    const int rank = comm.rank();
    const int DD_COMM_TAG = 0xab;

    const size_t numRealCells = realCells.size();
    const size_t numGhostCells = ghostCells.size();
    const size_t numLocalCells = localCells.size();

    auto moveIndexedParticleNonGlobal = [](ParticleList& dl, ParticleList& sl, int i)
    {
        dl.push_back(sl[i]);
        const int newSize = sl.size() - 1;
        if (i != newSize)
        {
            sl[i] = sl.back();
        }
        sl.resize(newSize);
    };

    auto mapPositionToLocalCell = [](FVDD& vd, const Real3D& pos)
    {
        const auto& cg = vd.cellGrid;
        const real* cellSize = cg.getCellSize();
        const real* invCellSize = cg.getInverseCellSize();
        const int frame = cg.getFrameWidth();

        int cpos[3];

        for (int i = 0; i < 3; ++i)
        {
            // check that all particles are still within cg
            if (false)  // ENABLE_CHECK
            {
                const real myLeft = cg.getMyLeft(i) - cellSize[i];
                const real myRight = cg.getMyRight(i) + cellSize[i];
                if (pos[i] < myLeft || pos[i] >= myRight)
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "",
                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << "," << myRight << ")");
            }

            real lpos = pos[i] + cellSize[i] - cg.getMyLeft(i);
            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

            if (false)  // ENABLE_CHECK
            {
                const int gridSize = cg.getFrameGridSize(i);
                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "",
                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
            }
        }
        auto pos2 = cg.mapPositionToIndex(cpos);
        return vd.vLocalCells[pos2];
    };

    auto resortRealCell =
        [&moveIndexedParticleNonGlobal, &mapPositionToLocalCell](FVDD& vd, Cell& cell)
    {
        for (size_t p = 0; p < cell.particles.size(); ++p)
        {
            Particle& part = cell.particles[p];
            {
                Cell* sortCell = mapPositionToLocalCell(vd, part.position());

                if (sortCell != &cell)
                {
                    if (sortCell == 0)
                    {
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "resortRealCell",
                                                 "sortCell == 0 not allowed");
                        // HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,"resortRealCell","sortCell
                        // == 0 not allowed"
                        //   << " particle position (" << part.position() << ") mapped to cell "
                        //   << vd.cellGrid.mapPositionToIndex(part.position()));
                    }
                    else
                    {
                        moveIndexedParticleNonGlobal(sortCell->particles, cell.particles, p);
                        --p;
                    }
                }
            }
        }
    };

    auto resortSubNode = [this, &resortRealCell](size_t inode)
    {
        auto& vd = vdd[inode];
        auto const numRealCells = vd.vRealCells.size();
        for (Cell& c : vd.vGhostCells) c.particles.clear();
        for (Cell* c : vd.vRealCells) resortRealCell(vd, *c);
    };

    /// count the particles are not in their correct cells
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    // HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (srcCell != dstCell) ctr++;
                }
            }
        }
        std::cout << "Check ctr: " << ctr << std::endl << std::endl;
    }

    /// count the  particles are not in their subnode domain
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    // Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    const bool isInner = vd.cellGrid.isInnerPosition(p.position().get());
                    // HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (!isInner) ctr++;
                }
            }
        }
        std::cout << "Check ctr !isInnerPosition: " << ctr << std::endl << std::endl;
    }

    // for(Cell *cell: baseClass::ghostCells) cell->particles.clear();

    /// verify that particles all fall within the boundaries of their cellGrid
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& part : srcCell->particles)
                {
                    const real* cellSize = cellGrid.getCellSize();
                    const real* invCellSize = cellGrid.getInverseCellSize();
                    const int frame = cellGrid.getFrameWidth();
                    auto mapPositionToLocalCell =
                        [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                    {
                        int cpos[3];

                        for (int i = 0; i < 3; ++i)
                        {
                            // check that all particles are still within cellGrid
                            if (true)
                            {
                                const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                if (pos[i] < myLeft || pos[i] >= myRight)
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << ","
                                                  << myRight << ")");
                            }

                            real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                            if (true)
                            {
                                const int gridSize = cellGrid.getFrameGridSize(i);
                                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                   << gridSize << ")");
                            }
                        }
                        return localCells[cellGrid.mapPositionToIndex(cpos)];
                    };
                    const auto myCell = mapPositionToLocalCell(part.position());
                }
            }
        }
    }

    real time = wallTimer.getElapsedTime();
    utils::parallelForLoop(0, numSubNodes, resortSubNode);
    timeDecomposeRealResort += wallTimer.getElapsedTime() - time;

    /// verify that particles all fall within the boundaries of their cellGrid
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& part : srcCell->particles)
                {
                    const real* cellSize = cellGrid.getCellSize();
                    const real* invCellSize = cellGrid.getInverseCellSize();
                    const int frame = cellGrid.getFrameWidth();
                    auto mapPositionToLocalCell =
                        [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                    {
                        int cpos[3];

                        for (int i = 0; i < 3; ++i)
                        {
                            // check that all particles are still within cellGrid
                            if (true)
                            {
                                const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                if (pos[i] < myLeft || pos[i] >= myRight)
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << ","
                                                  << myRight << ")");
                            }

                            real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                            if (true)
                            {
                                const int gridSize = cellGrid.getFrameGridSize(i);
                                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                   << gridSize << ")");
                            }
                        }
                        return localCells[cellGrid.mapPositionToIndex(cpos)];
                    };
                    const auto myCell = mapPositionToLocalCell(part.position());
                }
            }
        }
    }

    /// verify that every particle is in its correct cell
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (srcCell != dstCell) ctr++;
                }
            }
        }
        std::cout << "Check ctr after resort: " << ctr << std::endl << std::endl;
    }

    time = wallTimer.getElapsedTime();
    {
        /// Move ghost cell particles to destination cells
        espressopp::bc::BC const& boundary = *(getSystem()->bc);
        constexpr bool REAL_TO_GHOSTS = false;
        for (int _coord = 0; _coord < 3; ++_coord)
        {
            int coord = REAL_TO_GHOSTS ? _coord : (2 - _coord);
            const bool DO_PERIODIC = (nodeGrid.getGridSize(coord) == 1);
            for (int lr = 0; lr < 2; ++lr)
            {
                int dir = 2 * coord + lr;
                int oppositeDir = 2 * coord + (1 - lr);

                /// Intra-node block (including shifted/periodic)
                auto f_intraNode = [this, coord, dir, &boundary]()
                {
                    auto f_pair = [this, coord, dir, &boundary](size_t i)
                    {
                        const auto& pair = subNodePairsIntra[dir][i];
                        const size_t ir = std::get<0>(pair);
                        const size_t ig = std::get<1>(pair);
                        const bool shift = std::get<2>(pair);
                        const auto& ccr = vdd[ir].commCells[dir].reals;
                        const auto& ccg = vdd[ig].commCells[dir].ghosts;
                        const auto& lcr = vdd[ir].vLocalCells;
                        const auto& lcg = vdd[ig].vLocalCells;

                        HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size()))

                        const size_t numCells = ccr.size();
                        for (size_t ic = 0; ic < numCells; ic++)
                        {
                            ParticleList& reals = lcr[ccr[ic]]->particles;
                            ParticleList& ghosts = lcg[ccg[ic]]->particles;
                            reals.reserve(reals.size() + ghosts.size());
                            for (Particle& part : ghosts)
                            {
                                if (shift)
                                    boundary.foldCoordinate(part.position(), part.image(), coord);
                                reals.push_back(part);
                            }
                        }
                    };
                    size_t const numPairs = subNodePairsIntra[dir].size();
                    utils::parallelForLoop(0, numPairs, f_pair);
                };

                auto f_interNode = [this, coord, dir, oppositeDir, rank, &boundary, &comm]()
                {
                    std::vector<longint> sendSizes, recvSizes;  /// number of particles per cell
                    std::vector<longint> sendRanges,
                        recvRanges;  /// starting index for each cell
                    longint totalSend = 0,
                            totalRecv = 0;  /// number of particles to send and recv
                    {
                        /// NOTE: assuming uniform subNode division
                        const int numCommNodes = commNodesGhost[dir].size();
                        const int numCommCells =
                            vdd[commNodesGhost[dir][0]].commCells[dir].ghosts.size();
                        const int numSendCells = numCommNodes * numCommCells;
                        sendSizes.reserve(numSendCells);
                        int numRecvCells = 0;
                        spdlog::stopwatch build_buffer;
                        for (int idxCommNode = 0; idxCommNode < numCommNodes; idxCommNode++)
                        {
                            {
                                const int inode = commNodesGhost[dir][idxCommNode];
                                const auto& vd = vdd[inode];
                                const auto& lc = vd.vLocalCells;
                                const auto& cc = vd.commCells[dir].ghosts;

                                const int nodeStart = nodeCommCellRange[dir][idxCommNode];
                                const int nodeEnd = nodeStart + cc.size();

                                HPX4ESPP_DDPF_DEBUG(
                                    HPX4ESPP_ASSERT_EQUAL(nodeStart, sendSizes.size());
                                    HPX4ESPP_ASSERT_EQUAL(nodeStart, idxCommNode * cc.size());)

                                for (int ic = 0; ic < cc.size(); ic++)
                                {
                                    sendSizes.push_back(lc[cc[ic]]->particles.size());
                                }

                                HPX4ESPP_DDPF_DEBUG(
                                    HPX4ESPP_ASSERT_EQUAL(nodeEnd, sendSizes.size());)
                            }
                            {
                                const int inode = commNodesReal[dir][idxCommNode];
                                const auto& vd = vdd[inode];
                                const auto& lc = vd.vLocalCells;
                                const auto& rc = vd.commCells[dir].reals;
                                numRecvCells += rc.size();
                            }
                        }
                        recvSizes.resize(numRecvCells);

                        HPX4ESPP_DDPF_DEBUG(
                            HPX4ESPP_ASSERT_EQUAL(numRecvCells, numSendCells);
                            HPX4ESPP_ASSERT_EQUAL(sendSizes.size(), numSendCells);
                            HPX4ESPP_ASSERT_EQUAL(recvSizes.size(), numSendCells);
                            HPX4ESPP_ASSERT_EQUAL(nodeCommCellRange[dir].back(), numSendCells);)
                        SPDLOG_TRACE("Sending Sizes");
                        spdlog::stopwatch size_exchange;
                        if (nodeGrid.getNodePosition(coord) % 2 == 0)
                        {
                            comm.send(nodeGrid.getNodeNeighborIndex(oppositeDir), DD_COMM_TAG,
                                      sendSizes.data(), sendSizes.size());
                            comm.recv(nodeGrid.getNodeNeighborIndex(dir), DD_COMM_TAG,
                                      recvSizes.data(), recvSizes.size());
                        }
                        else
                        {
                            comm.recv(nodeGrid.getNodeNeighborIndex(dir), DD_COMM_TAG,
                                      recvSizes.data(), recvSizes.size());

                            comm.send(nodeGrid.getNodeNeighborIndex(oppositeDir), DD_COMM_TAG,
                                      sendSizes.data(), sendSizes.size());

                        }
                        SPDLOG_TRACE("Sending Sizes took {}", size_exchange);

                        sendRanges.reserve(sendSizes.size() + 1);
                        for (longint s : sendSizes)
                        {
                            sendRanges.push_back(totalSend);
                            totalSend += s;
                        }
                        sendRanges.push_back(totalSend);

                        recvRanges.reserve(recvSizes.size() + 1);
                        for (longint r : recvSizes)
                        {
                            recvRanges.push_back(totalRecv);
                            totalRecv += r;
                        }
                        recvRanges.push_back(totalRecv);
                    }

                    auto& sendBuf = this->decompSendBuf;
                    auto& recvBuf = this->decompRecvBuf;
                    sendBuf.clear();
                    sendBuf.resize(totalSend);
                    recvBuf.resize(totalRecv);

                    {
                        const int numCommNodes = commNodesGhost[dir].size();
                        auto f_packNode = [dir, &sendRanges, &sendSizes, &sendBuf, totalSend,
                                           this](int idxCommNode)
                        {
                            /// commCells to pack
                            const int inode = commNodesGhost[dir][idxCommNode];
                            const auto& vd = vdd[inode];
                            const auto& lc = vd.vLocalCells;
                            const auto& gc = vd.commCells[dir].ghosts;

                            const int nodeStart = nodeCommCellRange[dir][idxCommNode];
                            const int nodeEnd = nodeStart + gc.size();

                            HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(
                                nodeEnd, nodeCommCellRange[dir][idxCommNode + 1]))

                            for (int ic = 0; ic < gc.size(); ic++)
                            {
                                const ParticleList& ghosts = lc[gc[ic]]->particles;
                                const int ii = nodeStart + ic;
                                const auto start = sendRanges[ii];
                                const auto size = sendSizes[ii];

                                HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(ghosts.size(), size))

                                for (int ig = 0; ig < size; ig++)
                                {
                                    HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_LT(ig + start, totalSend))

                                    sendBuf[ig + start] = ghosts[ig];
                                }
                            }
                        };

                        // for(int idxCommNode=0; idxCommNode<numCommNodes; idxCommNode++)
                        //   f_packNode(idxCommNode);
                        spdlog::stopwatch buffer_build;
                        utils::parallelForLoop(0, numCommNodes, f_packNode);
                        SPDLOG_TRACE("Buffer was build in ", buffer_build);
                    }

                    const longint receiver = nodeGrid.getNodeNeighborIndex(oppositeDir);
                    const longint sender = nodeGrid.getNodeNeighborIndex(dir);

                    if (false && rank == 1)
                    {
                        auto str_list = [](const auto& list)
                        {
                            std::ostringstream ss;
                            for (const auto& l : list) ss << " " << l;
                            return ss.str();
                        };
                        std::cout << rank << ": f_interNode dir " << dir
                                  << " sendSizes:  " << str_list(sendSizes) << "\n"
                                  << rank << ": f_interNode dir " << dir
                                  << " sendRanges: " << str_list(sendRanges) << "\n"
                                  << rank << ": f_interNode dir " << dir
                                  << " recvSizes:  " << str_list(recvSizes) << "\n"
                                  << rank << ": f_interNode dir " << dir
                                  << " recvRanges: " << str_list(recvRanges) << std::endl;
                    }

                    /// Check send particles here
                    if (false)  // ENABLE_CHECK
                    {
                        std::cout << rank << ": f_interNode dir " << dir << " send " << totalSend
                                  << ", recv " << totalRecv << std::endl;

                        /// check that the particle belongs to this cellGrid
                        for (const auto& part : sendBuf)
                        {
                            const real* cellSize = cellGrid.getCellSize();
                            const real* invCellSize = cellGrid.getInverseCellSize();
                            const int frame = cellGrid.getFrameWidth();
                            auto mapPositionToLocalCell =
                                [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                            {
                                int cpos[3];

                                for (int i = 0; i < 3; ++i)
                                {
                                    // check that all particles are still within cellGrid
                                    if (true)
                                    {
                                        const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                        const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                        if (pos[i] < myLeft || pos[i] >= myRight)
                                            HPX4ESPP_THROW_EXCEPTION(
                                                hpx::assertion_failure, "mapPositionToLocalCell",
                                                "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                          << "," << myRight << ")");
                                    }

                                    real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                                    cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                                    if (true)
                                    {
                                        const int gridSize = cellGrid.getFrameGridSize(i);
                                        if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                            HPX4ESPP_THROW_EXCEPTION(
                                                hpx::assertion_failure, "mapPositionToLocalCell",
                                                "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                           << gridSize << ")");
                                    }
                                }
                                return localCells[cellGrid.mapPositionToIndex(cpos)];
                            };
                            const auto myCell = mapPositionToLocalCell(part.position());
                        }
                    }

                    SPDLOG_TRACE("Sending Buffers!");
                    // exchange particles, odd-even rule
                    spdlog::stopwatch buffer_send;

                    if (nodeGrid.getNodePosition(coord) % 2 == 0)
                    {
                        comm.send(receiver, DD_COMM_TAG, sendBuf.data(), sendBuf.size());
                        comm.recv(sender, DD_COMM_TAG, recvBuf.data(), recvBuf.size());
                    }
                    else
                    {
                        comm.recv(sender, DD_COMM_TAG, recvBuf.data(), recvBuf.size());
                        comm.send(receiver, DD_COMM_TAG, sendBuf.data(), sendBuf.size());
                    }
                    SPDLOG_TRACE("Buffers were sent in {}", buffer_send);

                    HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(recvBuf.size(), totalRecv))

                    {
                        const int numCommNodes = commNodesReal[dir].size();
                        auto f_unpackNode = [coord, dir, totalRecv, &recvRanges, &recvSizes,
                                             &recvBuf, &boundary, this](int idxCommNode)
                        {
                            /// commCells to unpack
                            const int inode = commNodesReal[dir][idxCommNode];
                            const auto& vd = vdd[inode];
                            const auto& lc = vd.vLocalCells;
                            const auto& rc = vd.commCells[dir].reals;

                            const int nodeStart = nodeCommCellRange[dir][idxCommNode];
                            const int nodeEnd = nodeStart + rc.size();

                            HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(
                                nodeEnd, nodeCommCellRange[dir][idxCommNode + 1]))

                            for (int ic = 0; ic < rc.size(); ic++)
                            {
                                ParticleList& reals = lc[rc[ic]]->particles;
                                const int ii = nodeStart + ic;
                                const auto start = recvRanges[ii];
                                const auto size = recvSizes[ii];
                                const auto end = start + size;
                                reals.reserve(reals.size() + size);
                                for (int ip = start; ip < end; ip++)
                                {
                                    HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_LT(ip, totalRecv));

                                    Particle& part = recvBuf[ip];
                                    boundary.foldCoordinate(part.position(), part.image(), coord);

                                    /// check that the particle belongs to this cellGrid
                                    if (false)  // ENABLE_CHECK
                                    {
                                        const real* cellSize = cellGrid.getCellSize();
                                        const real* invCellSize = cellGrid.getInverseCellSize();
                                        const int frame = cellGrid.getFrameWidth();
                                        auto mapPositionToLocalCell = [this, &cellSize,
                                                                       &invCellSize,
                                                                       &frame](const Real3D& pos)
                                        {
                                            int cpos[3];

                                            for (int i = 0; i < 3; ++i)
                                            {
                                                // check that all particles are still within
                                                // cellGrid
                                                if (true)
                                                {
                                                    const real myLeft =
                                                        cellGrid.getMyLeft(i) - cellSize[i];
                                                    const real myRight =
                                                        cellGrid.getMyRight(i) + cellSize[i];
                                                    if (pos[i] < myLeft || pos[i] >= myRight)
                                                        HPX4ESPP_THROW_EXCEPTION(
                                                            hpx::assertion_failure,
                                                            "mapPositionToLocalCell",
                                                            "pos[i]=" << pos[i] << " allowed: ["
                                                                      << myLeft << "," << myRight
                                                                      << ")");
                                                }

                                                real lpos =
                                                    pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                                                cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                                                if (true)
                                                {
                                                    const int gridSize =
                                                        cellGrid.getFrameGridSize(i);
                                                    if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                                        HPX4ESPP_THROW_EXCEPTION(
                                                            hpx::assertion_failure,
                                                            "mapPositionToLocalCell",
                                                            "cpos[i]=" << cpos[i] << " allowed: ["
                                                                       << 0 << "," << gridSize
                                                                       << ")");
                                                }
                                            }
                                            return localCells[cellGrid.mapPositionToIndex(cpos)];
                                        };
                                        const auto myCell = mapPositionToLocalCell(part.position());
                                    }

                                    reals.push_back(part);
                                }
                            }
                        };

                        // for(int idxCommNode=0; idxCommNode<numCommNodes; idxCommNode++)
                        //   f_unpackNode(idxCommNode);
                        utils::parallelForLoop(0, numCommNodes, f_unpackNode);
                    }

                    // std::cout << rank << ": f_interNode dir " << dir << " end" << std::endl;
                };

                if (!DO_PERIODIC) f_interNode();
                f_intraNode();
            }
        }
    }
    timeDecomposeRealComm += wallTimer.getElapsedTime() - time;

    if (false)
    {
        for (FVDD& vd : vdd)
            for (Cell& c : vd.vGhostCells) c.particles.clear();

        for (Cell* c : ghostCells) c->particles.clear();
    }

    /// check that all particles ended up in the correct cell
    if (false)  // ENABLE_CHECK
    {
        const real* cellSize = cellGrid.getCellSize();
        const real* invCellSize = cellGrid.getInverseCellSize();
        const int frame = cellGrid.getFrameWidth();
        auto mapPositionToLocalCell = [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
        {
            int cpos[3];

            for (int i = 0; i < 3; ++i)
            {
                // check that all particles are still within cellGrid
                if (true)
                {
                    const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                    const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                    if (pos[i] < myLeft || pos[i] >= myRight)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "mapPositionToLocalCell",
                                                 "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                           << "," << myRight << ")");
                }

                real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                if (true)
                {
                    const int gridSize = cellGrid.getFrameGridSize(i);
                    if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                        HPX4ESPP_THROW_EXCEPTION(
                            hpx::assertion_failure, "mapPositionToLocalCell",
                            "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
                }
            }
            return localCells[cellGrid.mapPositionToIndex(cpos)];
        };

        auto mapPositionToLocalCelldbg = [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
        {
            int cpos[3];

            for (int i = 0; i < 3; ++i)
            {
                // check that all particles are still within cellGrid
                if (true)
                {
                    const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                    const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                    if (pos[i] < myLeft || pos[i] >= myRight)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "mapPositionToLocalCell",
                                                 "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                           << "," << myRight << ")");
                }

                real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                if (true)
                {
                    const int gridSize = cellGrid.getFrameGridSize(i);
                    if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                        HPX4ESPP_THROW_EXCEPTION(
                            hpx::assertion_failure, "mapPositionToLocalCell",
                            "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
                }
            }
            const auto vdd_index = cellGrid.mapPositionToIndex(cpos);
            const auto& vdd_spec = vdd.at(vdd_index);
            std::array<double, 3> left_bound{vdd_spec.cellGrid.getMyLeft(0),
                                             vdd_spec.cellGrid.getMyLeft(1),
                                             vdd_spec.cellGrid.getMyLeft(2)};
            std::array<double, 3> right_bound{vdd_spec.cellGrid.getMyRight(0),
                                              vdd_spec.cellGrid.getMyRight(1),
                                              vdd_spec.cellGrid.getMyRight(2)};
            std::array<double, 3> particle_pos{pos[0], pos[1], pos[2]};

            SPDLOG_CRITICAL(
                "Particle should be in cell {} vdd{} left {} right {} particle {}",
                spdlog::fmt_lib::join(cpos, ","), spdlog::fmt_lib::join(left_bound, ","),
                spdlog::fmt_lib::join(right_bound, ","), spdlog::fmt_lib::join(particle_pos, ","));
        };

        /// check that every particle belongs to the correct cell
        Cell* c0 = localCells[0];
        for (int ic = 0; ic < numLocalCells; ic++)
        {
            Cell* currCell = localCells[ic];
            ParticleList& particles = currCell->particles;
            for (Particle& part : particles)
            {
                Cell* sortCell = mapPositionToLocalCell(part.position());
                if (sortCell != currCell)
                {
                    //                    mapPositionToLocalCelldbg(part.position());
                    HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, __FUNCTION__,
                                             "sortCell!=currCell");
                }
            }
        }
    }
}

void FullDomainDecomposition::decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_full()
{
    auto const comm = *(getSystem()->comm);
    const int rank = comm.rank();
    const int DD_COMM_TAG = 0xab;

    const size_t numRealCells = realCells.size();
    const size_t numGhostCells = ghostCells.size();
    const size_t numLocalCells = localCells.size();

    auto moveIndexedParticleNonGlobal = [](ParticleList& dl, ParticleList& sl, int i)
    {
        dl.push_back(sl[i]);
        const int newSize = sl.size() - 1;
        if (i != newSize)
        {
            sl[i] = sl.back();
        }
        sl.resize(newSize);
    };

    auto mapPositionToLocalCell = [](FVDD& vd, const Real3D& pos)
    {
        const auto& cg = vd.cellGrid;
        const real* cellSize = cg.getCellSize();
        const real* invCellSize = cg.getInverseCellSize();
        const int frame = cg.getFrameWidth();

        int cpos[3];

        for (int i = 0; i < 3; ++i)
        {
            // check that all particles are still within cg
            if (false)  // ENABLE_CHECK
            {
                const real myLeft = cg.getMyLeft(i) - cellSize[i];
                const real myRight = cg.getMyRight(i) + cellSize[i];
                if (pos[i] < myLeft || pos[i] >= myRight)
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "",
                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << "," << myRight << ")");
            }

            real lpos = pos[i] + cellSize[i] - cg.getMyLeft(i);
            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

            if (false)  // ENABLE_CHECK
            {
                const int gridSize = cg.getFrameGridSize(i);
                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "",
                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
            }
        }
        auto pos2 = cg.mapPositionToIndex(cpos);
        return vd.vLocalCells[pos2];
    };

    auto resortRealCell =
        [&moveIndexedParticleNonGlobal, &mapPositionToLocalCell](FVDD& vd, Cell& cell)
    {
        for (size_t p = 0; p < cell.particles.size(); ++p)
        {
            Particle& part = cell.particles[p];
            {
                Cell* sortCell = mapPositionToLocalCell(vd, part.position());

                if (sortCell != &cell)
                {
                    if (sortCell == nullptr)
                    {
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "resortRealCell",
                                                 "sortCell == 0 not allowed");
                        // HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,"resortRealCell","sortCell
                        // == 0 not allowed"
                        //   << " particle position (" << part.position() << ") mapped to cell "
                        //   << vd.cellGrid.mapPositionToIndex(part.position()));
                    }
                    else
                    {
                        moveIndexedParticleNonGlobal(sortCell->particles, cell.particles, p);
                        --p;
                    }
                }
            }
        }
    };

    auto resortSubNode = [this, &resortRealCell](size_t inode)
    {
        auto& vd = vdd[inode];
        auto const numRealCells = vd.vRealCells.size();
        for (Cell& c : vd.vGhostCells) c.particles.clear();
        for (Cell* c : vd.vRealCells) resortRealCell(vd, *c);
    };

    /// count the particles are not in their correct cells
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    // HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (srcCell != dstCell) ctr++;
                }
            }
        }
        std::cout << "Check ctr: " << ctr << std::endl << std::endl;
    }

    /// count the  particles are not in their subnode domain
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    // Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    const bool isInner = vd.cellGrid.isInnerPosition(p.position().get());
                    // HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (!isInner) ctr++;
                }
            }
        }
        std::cout << "Check ctr !isInnerPosition: " << ctr << std::endl << std::endl;
    }

    // for(Cell *cell: baseClass::ghostCells) cell->particles.clear();

    /// verify that particles all fall within the boundaries of their cellGrid
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& part : srcCell->particles)
                {
                    const real* cellSize = cellGrid.getCellSize();
                    const real* invCellSize = cellGrid.getInverseCellSize();
                    const int frame = cellGrid.getFrameWidth();
                    auto mapPositionToLocalCell =
                        [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                    {
                        int cpos[3];

                        for (int i = 0; i < 3; ++i)
                        {
                            // check that all particles are still within cellGrid
                            if (false)  // ENABLE_CHECK
                            {
                                const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                if (pos[i] < myLeft || pos[i] >= myRight)
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << ","
                                                  << myRight << ")");
                            }

                            real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                            if (false)  // ENABLE_CHECK
                            {
                                const int gridSize = cellGrid.getFrameGridSize(i);
                                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                   << gridSize << ")");
                            }
                        }
                        return localCells[cellGrid.mapPositionToIndex(cpos)];
                    };
                    const auto myCell = mapPositionToLocalCell(part.position());
                }
            }
        }
    }

    real time = wallTimer.getElapsedTime();
    //    utils::parallelForLoop(0, numSubNodes, resortSubNode);
    for (int i = 0; i < numSubNodes; ++i)
    {
        resortSubNode(i);
    }
    timeDecomposeRealResort += wallTimer.getElapsedTime() - time;

    /// verify that particles all fall within the boundaries of their cellGrid
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& part : srcCell->particles)
                {
                    const real* cellSize = cellGrid.getCellSize();
                    const real* invCellSize = cellGrid.getInverseCellSize();
                    const int frame = cellGrid.getFrameWidth();
                    auto mapPositionToLocalCell =
                        [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                    {
                        int cpos[3];

                        for (int i = 0; i < 3; ++i)
                        {
                            // check that all particles are still within cellGrid
                            if (false)  // ENABLE_CHECK
                            {
                                const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                if (pos[i] < myLeft || pos[i] >= myRight)
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << ","
                                                  << myRight << ")");
                            }

                            real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                            if (false)  // ENABLE_CHECK
                            {
                                const int gridSize = cellGrid.getFrameGridSize(i);
                                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                   << gridSize << ")");
                            }
                        }
                        return localCells[cellGrid.mapPositionToIndex(cpos)];
                    };
                    const auto myCell = mapPositionToLocalCell(part.position());
                }
            }
        }
    }

    /// verify that every particle is in its correct cell
    if (false)  // ENABLE_CHECK
    {
        size_t ctr = 0;
        for (size_t inode = 0; inode < numSubNodes; inode++)
        {
            auto& vd = vdd[inode];
            for (Cell* srcCell : vd.vLocalCells)
            {
                for (Particle& p : srcCell->particles)
                {
                    Cell* dstCell = mapPositionToLocalCell(vd, p.position());
                    HPX4ESPP_ASSERT_EQUAL(srcCell, dstCell);
                    if (srcCell != dstCell) ctr++;
                }
            }
        }
        std::cout << "Check ctr after resort: " << ctr << std::endl << std::endl;
    }

    time = wallTimer.getElapsedTime();
    std::array<hpx::future<void>, 26> neighbour_comm_futures;

    {
        /// Move ghost cell particles to destination cells

        for (const auto& send_to_neighbour : John::NeighbourEnumIterable)
        {
            // Neighbour is periodic to itself, do intra_node_exchange
            if (full_grid.neighbour_is_self(send_to_neighbour))
            {
                neighbour_comm_futures[send_to_neighbour] = hpx::async(
                    [this, send_to_neighbour]()
                    {
                        size_t const numPairs = fullSubNodePairsIntra[send_to_neighbour].size();
                        for (size_t i = 0; i < numPairs; ++i)
                        {
                            decomposeRealParticles_full_intra(i, send_to_neighbour);
                        }
                    });
                //                neighbour_comm_futures[send_to_neighbour].get();
                continue;  // Skip InterNodeStuff
            }
            const bool DO_PERIODIC = full_grid.is_periodic_neighbour(send_to_neighbour);

            auto f_interNode = [this, send_to_neighbour, &neighbour_comm_futures]()
            {
                // SEND_HERE
                const auto receive_neighbour = send_to_neighbour;

                auto send_future =
                    hpx::async([this, send_to_neighbour]()
                               { decomposeRealParticles_send_routine(send_to_neighbour); });

                auto recv_futures =
                    hpx::async([this, receive_neighbour]()
                               { decomposeRealParticles_receive_routine(receive_neighbour); });

                neighbour_comm_futures[send_to_neighbour] =
                    hpx::when_all(send_future, recv_futures);
            };
            f_interNode();
        }
    }

    hpx::wait_all(neighbour_comm_futures);
    timeDecomposeRealComm += wallTimer.getElapsedTime() - time;

    if (false)
    {
        for (FVDD& vd : vdd)
            for (Cell& c : vd.vGhostCells) c.particles.clear();

        for (Cell* c : ghostCells) c->particles.clear();
    }

    /// check that all particles ended up in the correct cell
    if (false)  // ENABLE_CHECK
    {
        const real* cellSize = cellGrid.getCellSize();
        const real* invCellSize = cellGrid.getInverseCellSize();
        const int frame = cellGrid.getFrameWidth();
        auto mapPositionToLocalCell = [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
        {
            int cpos[3];

            for (int i = 0; i < 3; ++i)
            {
                // check that all particles are still within cellGrid
                if (false)  // ENABLE_CHECK
                {
                    const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                    const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                    if (pos[i] < myLeft || pos[i] >= myRight)
                        HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure, "mapPositionToLocalCell",
                                                 "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                           << "," << myRight << ")");
                }

                real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                if (false)  // ENABLE_CHECK
                {
                    const int gridSize = cellGrid.getFrameGridSize(i);
                    if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                        HPX4ESPP_THROW_EXCEPTION(
                            hpx::assertion_failure, "mapPositionToLocalCell",
                            "cpos[i]=" << cpos[i] << " allowed: [" << 0 << "," << gridSize << ")");
                }
            }
            return localCells[cellGrid.mapPositionToIndex(cpos)];
        };

        /// check that every particle belongs to the correct cell
        Cell* c0 = localCells[0];
        for (int ic = 0; ic < numLocalCells; ic++)
        {
            Cell* currCell = localCells[ic];
            ParticleList& particles = currCell->particles;
            for (Particle& part : particles)
            {
                Cell* sortCell = mapPositionToLocalCell(part.position());
                if (sortCell != currCell)
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, __FUNCTION__,
                        spdlog::fmt_lib::format("sortCell!=currCell ic {}, part.position {},{},{}",
                                                ic, part.getPos()[0], part.getPos()[1],
                                                part.getPos()[2]));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Utility functions for connecting to integrator to do stepwise offload
/// We need this when we only want to do force calculation on particleArrays
void FullDomainDecomposition::connectOffload(
    boost::shared_ptr<espressopp::integrator::MDIntegrator> mdintegrator)
{
    sigResetParticles = espressopp::storage::Storage::onParticlesChanged.connect(
        boost::signals2::at_front,  // call first due to reordering
        boost::bind(&FullDomainDecomposition::resetParticles, this));
    sigResetCells = espressopp::storage::Storage::onCellAdjust.connect(
        boost::signals2::at_back, boost::bind(&FullDomainDecomposition::resetCells, this));
    sigBefCalcForces = mdintegrator->aftInitF.connect(
        boost::signals2::at_back, boost::bind(&FullDomainDecomposition::befCalcForces, this));
    sigUpdateForces = mdintegrator->aftCalcFLocal.connect(
        boost::signals2::at_front, boost::bind(&FullDomainDecomposition::updateForces, this));
    offload = true;
}

void FullDomainDecomposition::disconnectOffload()
{
    if (sigResetParticles.connected()) sigResetParticles.disconnect();
    if (sigResetCells.connected()) sigResetCells.disconnect();
    if (sigBefCalcForces.connected()) sigBefCalcForces.disconnect();
    if (sigUpdateForces.connected()) sigUpdateForces.disconnect();
    offload = false;
}

void FullDomainDecomposition::resetParticles()
{
    /// load new particle data
    auto f = [this](size_t i)
    {
        auto& vs = this->virtualStorage[i];
        vs.particles.copyFrom(vs.localCells);
    };
    const size_t nvs = virtualStorage.size();
    utils::parallelForLoop(0, nvs, f);

    if (false)  // ENABLE_CHECK
    {
        std::ostringstream _ss;
        for (auto const& vs : virtualStorage)
        {
            _ss << " (";
            for (auto const& lc : vs.localCells)
            {
                _ss << " " << lc - localCells[0];
            }
            _ss << " )\n";
        }
        std::cout << " virtualStorage: vs.localCells: " << _ss.str() << std::endl;
    }

    if (false)  // ENABLE_CHECK
    {
        static const size_t chunk_size_ = vec::ESPP_VECTOR_WIDTH;
        std::ostringstream _ss;
        for (auto const& vs : virtualStorage)
        {
            size_t total = 0;
            for (auto const& lc : vs.localCells)
            {
                const size_t size = lc->particles.size();
                // total += (1 + ((size - 1) / chunk_size_)) * chunk_size_; ;
                total += size;
            }
            _ss << " " << total;
        }
        std::cout << " virtualStorage: vs.localCells: lc.particles.size " << _ss.str() << std::endl;
    }

    if (false)  // ENABLE_CHECK
    {
        std::ostringstream _ss;
        for (auto const& vs : virtualStorage)
        {
            _ss << " " << vs.particles.size();
        }
        std::cout << " virtualStorage: vs.particles.size(): " << _ss.str() << std::endl;
    }

    // std::cout << "Called " << __FUNCTION__ << std::endl;
}

void FullDomainDecomposition::resetCells()
{
    resetVirtualStorage();
    // std::cout << "Called " << __FUNCTION__ << std::endl;
}

void FullDomainDecomposition::befCalcForces()
{
    auto f = [this](size_t i)
    {
        auto& vs = this->virtualStorage[i];
        auto& particles = vs.particles;
        vs.particles.zeroForces();
        /// overwrite particleArray positon data
        particles.updateFromPositionOnly(vs.localCells);
    };
    const size_t nvs = virtualStorage.size();
    utils::parallelForLoop(0, nvs, f);

    // std::cout << "Called " << __FUNCTION__ << std::endl;
}

void FullDomainDecomposition::updateForces()
{
    /// FIXME: This part probably results in race conditions for ghost cells
    /// TODO: Parallelize only once a clear inner ghost<->real cell update is implemented
    for (auto& vs : virtualStorage)
    {
        vs.particles.addToForceOnly(vs.localCells);
    }
    // std::cout << "Called " << __FUNCTION__ << std::endl;
}

void FullDomainDecomposition::resetTimers()
{
    timeUpdateGhostsBlocking_comm = 0.0;       /* 0 */
    timeCollectGhostForcesBlocking_comm = 0.0; /* 1 */
    timeParticlesCopyFrom = 0.0;               /* 2 */
    timeParticlesUpdateTo = 0.0;               /* 3 */
    timeDecomposeInvGhosts = 0.0;              /* 4 */
    timeDecomposeReal = 0.0;                   /* 5 */
    timeDecomposeExcGhosts = 0.0;              /* 6 */
    timeDecomposeSignal = 0.0;                 /* 7 */
    timeDecomposeRealResort = 0.0;             /* 8 */
    timeDecomposeRealComm = 0.0;               /* 9 */
    timeLoadLoop = 0.0;                        /* 10 */
    timeLoadPrepGhost = 0.0;                   /* 11 */
    timeLoadSignal = 0.0;                      /* 12 */
    timeUnloadLoop = 0.0;                      /* 13 */
    timeUnloadSignal = 0.0;                    /* 14 */
    timeUpdateGhosts_InterNode_pack = 0.0;     /* 15 */
    timeUpdateGhosts_InterNode_comm = 0.0;     /* 16 */
    timeUpdateGhosts_InterNode_unpack = 0.0;   /* 17 */
    timeUpdateGhosts_IntraNode = 0.0;          /* 18 */
    timeExcGhosts_InterNode_sizes = 0.0;       /* 19 */
    timeExcGhosts_InterNode_pack = 0.0;        /* 20 */
    timeExcGhosts_InterNode_comm = 0.0;        /* 21 */
    timeExcGhosts_InterNode_unpack = 0.0;      /* 22 */
    timeExcGhosts_IntraNode = 0.0;             /* 23 */
    wallTimer.reset();
}

void FullDomainDecomposition::loadTimers(real t[24])
{
    t[0] = timeUpdateGhostsBlocking_comm;
    t[1] = timeCollectGhostForcesBlocking_comm;
    t[2] = timeParticlesCopyFrom;
    t[3] = timeParticlesUpdateTo;
    t[4] = timeDecomposeInvGhosts;
    t[5] = timeDecomposeReal;
    t[6] = timeDecomposeExcGhosts;
    t[7] = timeDecomposeSignal;
    t[8] = timeDecomposeRealResort;
    t[9] = timeDecomposeRealComm;
    t[10] = timeLoadLoop;
    t[11] = timeLoadPrepGhost;
    t[12] = timeLoadSignal;
    t[13] = timeUnloadLoop;
    t[14] = timeUnloadSignal;
    t[15] = timeUpdateGhosts_InterNode_pack;
    t[16] = timeUpdateGhosts_InterNode_comm;
    t[17] = timeUpdateGhosts_InterNode_unpack;
    t[18] = timeUpdateGhosts_IntraNode;
    t[19] = timeExcGhosts_InterNode_sizes;
    t[20] = timeExcGhosts_InterNode_pack;
    t[21] = timeExcGhosts_InterNode_comm;
    t[22] = timeExcGhosts_InterNode_unpack;
    t[23] = timeExcGhosts_IntraNode;
}

static python::object wrapGetTimers(class FullDomainDecomposition* obj)
{
    real tms[24];
    obj->loadTimers(tms);
    return boost::python::make_tuple(tms[0], tms[1], tms[2], tms[3], tms[4], tms[5], tms[6], tms[7],
                                     tms[8], tms[9], tms[10], tms[11], tms[12], tms[13], tms[14]);
}

static python::object wrapGetTimers2(class FullDomainDecomposition* obj)
{
    real tms[24];
    obj->loadTimers(tms);
    return boost::python::make_tuple(tms[15], tms[16], tms[17], tms[18], tms[19], tms[20], tms[21],
                                     tms[22], tms[23]);
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Provides interace to testing functions
std::vector<FVDD::CommCellIdx> FullDomainDecomposition::getCommCellsAsIdx() const
{
    auto lc0 = localCells.at(0);
    std::vector<FVDD::CommCellIdx> cci;
    cci.resize(6);
    for (size_t i = 0; i < 6; i++)
    {
        auto const& cc = commCells[i];
        cci.push_back({});
        {
            auto& rcells = cci[i].reals;
            rcells.reserve(cc.reals.size());
            for (auto const& cr : cc.reals) rcells.push_back(cr - lc0);
        }
        {
            auto& gcells = cci[i].ghosts;
            gcells.reserve(cc.ghosts.size());
            for (auto const& cg : cc.ghosts) gcells.push_back(cg - lc0);
        }
    }
    return std::move(cci);
}

// replaces tools.decomp.nodeGridSimple but inverted
Int3D FDDnodeGridSimple(int n)
{
    int ijkmax = 3 * n * n + 1;
    int d1 = 1;
    int d2 = 1;
    int d3 = 1;
    for (int i = 1; i < n + 1; i++)
    {
        for (int j = i; j < n + 1; j++)
        {
            for (int k = j; k < n + 1; k++)
            {
                int ijk = i * i + j * j + k * k;
                if ((i * j * k == n) && (ijk < ijkmax))
                {
                    d1 = i;
                    d2 = j;
                    d3 = k;
                    ijkmax = ijk;
                }
            }
        }
    }
    return Int3D(d1, d2, d3);
}

// ensures that the resulting grid divides mainCellGrid c
Int3D FDDnodeGridMultiple(int n, Int3D const& c)
{
    int ijkmax = 3 * n * n + 1;
    int d1 = 1;
    int d2 = 1;
    int d3 = 1;
    for (int i = 1; i < n + 1; i++)
    {
        for (int j = i; j < n + 1; j++)
        {
            for (int k = j; k < n + 1; k++)
            {
                const int ijk = i * i + j * j + k * k;
                const int rem = c[0] % i + c[1] % j + c[2] % k;
                if ((i * j * k == n) && (ijk < ijkmax) && rem == 0)
                {
                    d1 = i;
                    d2 = j;
                    d3 = k;
                    ijkmax = ijk;
                }
            }
        }
    }

    const auto n_res = d1 * d2 * d3;
    HPX4ESPP_ASSERT_EQUAL_MSG(n, n_res, "Input c=(" << c << ") can't be distributed into n=" << n);

    return {d1, d2, d3};
}

void FullDomainDecomposition::registerPython()
{
    using namespace espressopp::python;
    numpy::initialize();

    class_<FullDomainDecomposition, bases<espressopp::storage::DomainDecomposition, StorageHPX>,
           boost::noncopyable>("Fullhpx4espp_storage_DomainDecomposition",
                               init<shared_ptr<System>, const Int3D&, const Int3D&, int,
                                    const Int3D&, int, bool, bool, bool, bool, bool>())
        .def("initChannels", &FullDomainDecomposition::initChannels)
        .def("getChannelIndices", &FullDomainDecomposition::getChannelIndices)
        .def("connectOffload", &FullDomainDecomposition::connectOffload)
        .def("connectedOffload", &FullDomainDecomposition::connectedOffload)
        .def("disconnectOffload", &FullDomainDecomposition::disconnectOffload)
        .def("resetVirtualStorage", &FullDomainDecomposition::resetVirtualStorage)
        .def("resetTimers", &FullDomainDecomposition::resetTimers)
        .def("getTimers", &wrapGetTimers)
        .def("getTimers2", &wrapGetTimers2);
}

void FullDomainDecomposition::decomposeRealParticles_full_intra(size_t pair_index,
                                                                John::NeighbourRelation neighbour)
{
    espressopp::bc::BC const& boundary = *(getSystem()->bc);

    const auto& [ir, ig, shift] = fullSubNodePairsIntra[neighbour][pair_index];
    const auto& ccr = vdd[ir].fullCommCells[neighbour].reals;
    const auto& ccg = vdd[ig].fullCommCells[neighbour].ghosts;
    const auto& lcr = vdd[ir].vLocalCells;  // TOUCH CONTENTION
    const auto& lcg = vdd[ig].vLocalCells;  // TOUCH CONTENTION

    HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size()));

    const size_t numCells = ccr.size();
    for (size_t ic = 0; ic < numCells; ic++)
    {
        ParticleList& reals = lcr[ccr[ic]]->particles;
        ParticleList& ghosts = lcg[ccg[ic]]->particles;
        reals.reserve(reals.size() + ghosts.size());  // To Crash Enable them
        for (Particle& part : ghosts)
        {
            if (shift)
            {
                for (int coord = 0; coord < 3; ++coord)
                {
                    boundary.foldCoordinate(part.position(), part.image(), coord);
                }
            }
            reals.push_back(part);  // To Crash Enable them
        }
    }
}
std::vector<Particle> FullDomainDecomposition::decomposeRealParticles_pack_node(
    John::NeighbourRelation send_to_neighbour,
    const std::vector<longint>& sendRanges,
    const std::vector<longint>& sendSizes)
{
    const auto numCommNodes = fullCommNodesGhost[send_to_neighbour].size();
    const auto totalSend = sendRanges.back();
    std::vector<Particle> sendBuf(sendRanges.back());
    for (size_t idxCommNode = 0; idxCommNode < numCommNodes; ++idxCommNode)
    {
        const auto inode = fullCommNodesGhost[send_to_neighbour][idxCommNode];
        const auto& vd = vdd[inode];
        const auto& lc = vd.vLocalCells;
        const auto& gc = vd.fullCommCells[send_to_neighbour].ghosts;

        const auto nodeStart = fullNodeCommCellRange[send_to_neighbour][idxCommNode];
        const auto nodeEnd = nodeStart + gc.size();

        HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(
            nodeEnd, fullNodeCommCellRange[send_to_neighbour][idxCommNode + 1]))

        for (int ic = 0; ic < gc.size(); ic++)
        {
            const ParticleList& ghosts = lc[gc[ic]]->particles;
            const auto ii = nodeStart + ic;
            const auto start = sendRanges[ii];
            const auto size = sendSizes[ii];

            HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(ghosts.size(), size))

            for (int ig = 0; ig < size; ig++)
            {
                HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_LT(ig + start, totalSend))

                sendBuf[ig + start] = ghosts[ig];
            }
        }
    }

    /// Check send particles here
    if (false)  // ENABLE_CHECK
    {
        /// check that the particle belongs to this cellGrid
        for (const auto& part : sendBuf)
        {
            const real* cellSize = cellGrid.getCellSize();
            const real* invCellSize = cellGrid.getInverseCellSize();
            const int frame = cellGrid.getFrameWidth();
            auto mapPositionToLocalCell = [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
            {
                int cpos[3];

                for (int i = 0; i < 3; ++i)
                {
                    // check that all particles are still within cellGrid
                    if (false)  // ENABLE_CHECK
                    {
                        const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                        const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                        if (pos[i] < myLeft || pos[i] >= myRight)
                            HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,
                                                     "mapPositionToLocalCell",
                                                     "pos[i]=" << pos[i] << " allowed: [" << myLeft
                                                               << "," << myRight << ")");
                    }

                    real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                    cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                    if (false)  // ENABLE_CHECK
                    {
                        const int gridSize = cellGrid.getFrameGridSize(i);
                        if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                            HPX4ESPP_THROW_EXCEPTION(hpx::assertion_failure,
                                                     "mapPositionToLocalCell",
                                                     "cpos[i]=" << cpos[i] << " allowed: [" << 0
                                                                << "," << gridSize << ")");
                    }
                }
                return localCells[cellGrid.mapPositionToIndex(cpos)];
            };
            const auto myCell = mapPositionToLocalCell(part.position());
        }
    }

    return sendBuf;
}

void FullDomainDecomposition::simple_decomposeRealParticles_receive_routine(
    John::NeighbourRelation receive_neighbour)
{
    std::vector<longint> recvRanges;
    espressopp::bc::BC const& boundary = *(getSystem()->bc);
    const auto filling_relation = opposite_neighbour(receive_neighbour);

    auto recvSizes =
        ch_decompose_real_sizes.get_receive_channel(receive_neighbour).get(hpx::launch::sync);
    longint totalRecv = 0;

    for (longint r : recvSizes)
    {
        recvRanges.push_back(totalRecv);
        totalRecv += r;
    }
    totalRecv = recvRanges.back();

    auto recvBuf =
        ch_decompose_real_particles.get_receive_channel(receive_neighbour).get(hpx::launch::sync);

    const auto numCells = fullCommCells[filling_relation].reals.size();
    // int ctr = 0;
    static hpx::mutex m;
    m.lock();
    auto f_unpack = [filling_relation, &recvRanges, &recvSizes, &recvBuf, &boundary, this](int i)
    {
        ParticleList& reals = fullCommCells[filling_relation].reals[i]->particles;
        const int start = recvRanges[i];
        const int size = recvSizes[i];
        const int end = start + size;
        reals.reserve(reals.size() + size);
        const auto direction = enum_to_direction(filling_relation);
        for (int ip = start; ip < end; ip++)
        {
            Particle& part = recvBuf[ip];
            for (auto j = 0; j < 3; ++j)
            {
                if (direction[j] != 0)
                {
                    boundary.foldCoordinate(part.position(), part.image(), j);
                }
            }
            reals.push_back(part);
        }
    };
    /// NOTE: too few particles to parallelize
    // utils::parallelForLoop(0, numCells, f_unpack);
    for (int i = 0; i < numCells; ++i) f_unpack(i);
    m.unlock();
}

void FullDomainDecomposition::simple_decomposeRealParticles_send_routine(
    John::NeighbourRelation send_to_neighbour)
{
    std::vector<longint> sendSizes;
    std::vector<longint> sendRanges;
    sendSizes.reserve(fullCommCells[send_to_neighbour].ghosts.size());
    for (int i = 0, end = fullCommCells[send_to_neighbour].ghosts.size(); i < end; ++i)
    {
        sendSizes.push_back(fullCommCells[send_to_neighbour].ghosts[i]->particles.size());
    }
    auto totalSend = 0;
    for (longint s : sendSizes)
    {
        sendRanges.push_back(totalSend);
        totalSend += s;
    }

    auto send_sizes_fut = ch_decompose_real_sizes.get_send_channel(send_to_neighbour)
                              .set(hpx::launch::async, sendSizes);

    const int numCells = fullCommCells[send_to_neighbour].ghosts.size();
    std::vector<Particle> sendBuf(totalSend);
    auto f_pack = [send_to_neighbour, &sendRanges, &sendSizes, &sendBuf, this](int i)
    {
        ParticleList& ghosts = fullCommCells[send_to_neighbour].ghosts[i]->particles;
        longint const& start = sendRanges[i];
        longint const& size = sendSizes[i];
        for (int ig = 0, ip = start; ig < size; ip++, ig++)
        {
            sendBuf[ip] = ghosts[ig];
        }
        ghosts.clear();
    };
    /// NOTE: too few particles to parallelize
    // utils::parallelForLoop(0, numCells, f_pack);
    for (int i = 0; i < numCells; ++i) f_pack(i);
    ch_decompose_real_particles.get_send_channel(send_to_neighbour).set(std::move(sendBuf));
}

void FullDomainDecomposition::decomposeRealParticles_send_routine(
    John::NeighbourRelation send_to_neighbour)
{
    const auto receive_neighbour = opposite_neighbour(send_to_neighbour);
    std::vector<longint> sendSizes, sendRanges;
    longint totalSend = 0;

    /// NOTE: assuming uniform subNode division
    const auto numCommNodes = fullCommNodesGhost[send_to_neighbour].size();
    const auto numCommCells = vdd[fullCommNodesGhost[send_to_neighbour][0]]
                                  .fullCommCells[send_to_neighbour]
                                  .ghosts.size();
    const int numSendCells = numCommNodes * numCommCells;
    sendSizes.reserve(numSendCells);
    spdlog::stopwatch build_buffer;
    for (auto idxCommNode = 0; idxCommNode < numCommNodes; idxCommNode++)
    {
        {
            const auto inode = fullCommNodesGhost[send_to_neighbour][idxCommNode];
            const auto& vd = vdd[inode];
            const auto& lc = vd.vLocalCells;
            const auto& cc = vd.fullCommCells[send_to_neighbour].ghosts;

            const auto nodeStart = fullNodeCommCellRange[send_to_neighbour][idxCommNode];
            const auto nodeEnd = nodeStart + cc.size();

            HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(nodeStart, sendSizes.size());
                                HPX4ESPP_ASSERT_EQUAL(nodeStart, idxCommNode * cc.size());)

            for (unsigned long ic : cc)
            {
                sendSizes.push_back(lc[ic]->particles.size());
            }

            HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(nodeEnd, sendSizes.size());)
        }
    }

    HPX4ESPP_DDPF_DEBUG(
        HPX4ESPP_ASSERT_EQUAL(sendSizes.size(), numSendCells);
        HPX4ESPP_ASSERT_EQUAL(nodeCommCellRange[send_to_neighbour].back(), numSendCells);)

    sendRanges = prepare_ranges_from_sizes(sendSizes);
    totalSend = sendRanges.back();

    auto send_sizes_fut = ch_decompose_real_sizes.get_send_channel(send_to_neighbour)
                              .set(hpx::launch::async, sendSizes);

    auto sendBuf = decomposeRealParticles_pack_node(send_to_neighbour, sendRanges, sendSizes);
    SPDLOG_TRACE("Sending Buffers! buff {} size {}", John::relation_to_string(send_to_neighbour),
                 sendBuf.size());
    spdlog::stopwatch buffer_send;

    ch_decompose_real_particles.get_send_channel(send_to_neighbour).set(std::move(sendBuf));
}
void FullDomainDecomposition::decomposeRealParticles_unpack_node(
    John::NeighbourRelation filling_neighbour,
    const std::vector<longint>& recvRanges,
    const std::vector<longint>& recvSizes,
    std::vector<Particle>& received_particles)
{
    espressopp::bc::BC const& boundary = *(getSystem()->bc);
    const auto numCommNodes = fullCommNodesReal[filling_neighbour].size();

    for (size_t idxCommNode = 0; idxCommNode < numCommNodes; ++idxCommNode)
    {
        const auto inode = fullCommNodesReal[filling_neighbour][idxCommNode];
        const auto& vd = vdd[inode];
        const auto& lc = vd.vLocalCells;
        const auto& rc = vd.fullCommCells[filling_neighbour].reals;

        const auto nodeStart = fullNodeCommCellRange[filling_neighbour][idxCommNode];
        const auto nodeEnd = nodeStart + rc.size();

        HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(
            nodeEnd, fullNodeCommCellRange[filling_neighbour][idxCommNode + 1]))

        for (int ic = 0; ic < rc.size(); ic++)
        {
            ParticleList& reals = lc[rc[ic]]->particles;
            const auto ii = nodeStart + ic;
            const auto start = recvRanges[ii];
            const auto size = recvSizes[ii];
            const auto end = start + size;
            reals.reserve(reals.size() + size);
            for (int ip = start; ip < end; ip++)
            {
                HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_LT(ip, totalRecv));
                Particle& part = received_particles[ip];
                auto direction = enum_to_direction(filling_neighbour);
                for (int coord = 0; coord < 3; ++coord)
                {
                    if (direction[coord] != 0)
                    {
                        boundary.foldCoordinate(part.position(), part.image(), coord);
                    }
                }

                /// check that the particle belongs to this cellGrid
                if (false)  // ENABLE_CHECK
                {
                    const real* cellSize = cellGrid.getCellSize();
                    const real* invCellSize = cellGrid.getInverseCellSize();
                    const int frame = cellGrid.getFrameWidth();
                    auto mapPositionToLocalCell =
                        [this, &cellSize, &invCellSize, &frame](const Real3D& pos)
                    {
                        int cpos[3];

                        for (int i = 0; i < 3; ++i)
                        {
                            // check that all particles are still within
                            // cellGrid
                            if (false)  // ENABLE_CHECK
                            {
                                const real myLeft = cellGrid.getMyLeft(i) - cellSize[i];
                                const real myRight = cellGrid.getMyRight(i) + cellSize[i];
                                if (pos[i] < myLeft || pos[i] >= myRight)
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "pos[i]=" << pos[i] << " allowed: [" << myLeft << ","
                                                  << myRight << ")");
                            }

                            real lpos = pos[i] + cellSize[i] - cellGrid.getMyLeft(i);
                            cpos[i] = static_cast<int>(lpos * invCellSize[i]);

                            if (false)  // ENABLE_CHECK
                            {
                                const int gridSize = cellGrid.getFrameGridSize(i);
                                if ((cpos[i] < 0) || (cpos[i] >= gridSize))
                                    HPX4ESPP_THROW_EXCEPTION(
                                        hpx::assertion_failure, "mapPositionToLocalCell",
                                        "cpos[i]=" << cpos[i] << " allowed: [" << 0 << ","
                                                   << gridSize << ")");
                            }
                        }
                        return localCells[cellGrid.mapPositionToIndex(cpos)];
                    };
                    const auto myCell = mapPositionToLocalCell(part.position());
                }

                reals.push_back(part);
            }
        }
    }
}
void FullDomainDecomposition::decomposeRealParticles_receive_routine(
    John::NeighbourRelation filling_relation)
{
    static hpx::mutex m;

    auto other_neighbour = opposite_neighbour(filling_relation);
    auto recvSizes =
        ch_decompose_real_sizes.get_receive_channel(other_neighbour).get(hpx::launch::sync);
    longint totalRecv = 0;

    auto recvRanges = prepare_ranges_from_sizes(recvSizes);
    totalRecv = recvRanges.back();

    auto received_particles =
        ch_decompose_real_particles.get_receive_channel(other_neighbour).get(hpx::launch::sync);
    HPX4ESPP_DDPF_DEBUG(HPX4ESPP_ASSERT_EQUAL(received_particles.size(), totalRecv))
    m.lock();

    SPDLOG_TRACE("Received_buffers from {} to fill {} size {}",
                 John::relation_to_string(other_neighbour),
                 John::relation_to_string(filling_relation), received_particles.size());

    decomposeRealParticles_unpack_node(filling_relation, recvRanges, recvSizes, received_particles);
    m.unlock();
}
void FullDomainDecomposition::exchangeGhosts_intra(const John::NeighbourRelation neighbour,
                                                   const Real3D shift,
                                                   const int extradata)
{
    auto f_copyRealsToGhosts = [this, &shift, &extradata, &neighbour](size_t i)
    {
        copyRealsToGhosts(*fullCommCells[neighbour].reals[i], *fullCommCells[neighbour].ghosts[i],
                          extradata, shift);
    };
    size_t const numCommCells = fullCommCells[neighbour].ghosts.size();
    utils::parallelForLoop(0, numCommCells, f_copyRealsToGhosts);
}

}  // namespace espressopp::hpx4espp::storage
/// WARNING: Storage::localParticles is unused in this implementation
