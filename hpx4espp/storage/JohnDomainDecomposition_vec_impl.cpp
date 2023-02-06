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
#include <hpx/future.hpp>
#include "hpx4espp/storage/JohnDomainDecomposition.hpp"
#include "hpx4espp/utils/algorithms/for_loop.hpp"
#include "hpx4espp/utils/assert.hpp"
#include "bc/BC.hpp"
#include <hpx/format.hpp>
#include "../../john_examples/global_data.hpp"
#include <fstream>
const int DD_COMM_TAG = 0xab;

/// manually enable by adding { STATEMENT }
#define HPX4ESPP_DDVI_DEBUG(STATEMENT)
#define HPX4ESPP_DDVI_CHANNELS_DEBUG(STATEMENT)
HPX_REGISTER_CHANNEL(int);
typedef std::vector<espressopp::real> RealVector;
typedef espressopp::vec::AlignedVector<espressopp::real> AlignedVector;
HPX_REGISTER_CHANNEL(AlignedVector);
HPX_REGISTER_CHANNEL(RealVector);
typedef std::vector<espressopp::longint> IntVec;
HPX_REGISTER_CHANNEL(IntVec);
typedef std::vector<espressopp::Particle> ParticleVector;
HPX_REGISTER_CHANNEL(ParticleVector);
typedef hpx::serialization::serialize_buffer<char> serializer_particiles;

HPX_REGISTER_CHANNEL(serializer_particiles);
namespace espressopp::hpx4espp::storage
{
template <FullDomainDecomposition::AddShift DO_SHIFT>
void FullDomainDecomposition::fullCopyRealsToGhostsIntra(
    John::NeighbourRelation neighbour, size_t ir, size_t ig, size_t is, const Real3D& shift)
{
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        const auto& ccr = vdd[ir].fullCommCellsOwned[neighbour].reals;
        const auto& ccg = vdd[ig].fullCommCellsOwned[neighbour].ghosts;

        if (neighbour == 0)
        {
            HPX4ESPP_ASSERT_EQUAL(vdd[ir].fullCommCellsOwned[0].ghosts.size(),
                                  vdd[ir].commCellsOwned[0].ghosts.size())
            HPX4ESPP_ASSERT_EQUAL(vdd[ir].fullCommCellsOwned[1].ghosts.size(),
                                  vdd[ir].commCellsOwned[1].ghosts.size())
            for (size_t i = 0; i < vdd[ir].fullCommCellsOwned[0].ghosts.size(); ++i)
            {
                auto& mine = vdd[ir].fullCommCellsOwned[0].ghosts[i];
                auto& orig = vdd[ir].commCellsOwned[0].ghosts[i];
                HPX4ESPP_ASSERT_EQUAL(mine, orig);
            }

            for (size_t i = 0; i < vdd[ir].fullCommCellsOwned[1].ghosts.size(); ++i)
            {
                auto& mine2 = vdd[ir].fullCommCellsOwned[1].ghosts[i];

                auto& orig2 = vdd[ir].commCellsOwned[1].ghosts[i];
                HPX4ESPP_ASSERT_EQUAL(mine2, orig2);
            }
        }
        HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size());)
        const size_t numCells = ccr.size();

        const auto& pr = virtualStorage[ir].particles;
        auto& pg = virtualStorage[ig].particles;
        const auto& crr = pr.cellRange();
        const auto& crg = pg.cellRange();

        {
            auto f_dim = [&](const real* __restrict ptr_r, real* __restrict ptr_g, real shift_v)
            {
                for (size_t ic = 0; ic < numCells; ic++)
                {
                    const size_t icr = ccr[ic];
                    const size_t icg = ccg[ic];

                    const size_t numPart = crr[icr + 1] - crr[icr];
                    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(numPart, (crg[icg + 1] - crg[icg]));)

                    const real* __restrict ptr_s_r = ptr_r + crr[icr];
                    real* __restrict ptr_s_g = ptr_g + crg[icg];

                    if (DO_SHIFT)
                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_g[ip] = ptr_s_r[ip] + shift_v;
                        }
                    }
                    else
                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_g[ip] = ptr_s_r[ip];
                        }
                    }
                }
            };
            f_dim(pr.p_x.data(), pg.p_x.data(), shift[0]);
            f_dim(pr.p_y.data(), pg.p_y.data(), shift[1]);
            f_dim(pr.p_z.data(), pg.p_z.data(), shift[2]);
        }
    }
    else
    {
        throw std::runtime_error("ONLY HPX OptimiseComm is enabled");
    }
}
void FullDomainDecomposition::fullAddGhostForcesToRealsIntra(John::NeighbourRelation neighbour,
                                                             size_t ir,
                                                             size_t ig,
                                                             size_t is)
{
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        const auto& ccr = vdd[ir].fullCommCellsOwned[neighbour].reals;
        const auto& ccg = vdd[ig].fullCommCellsOwned[neighbour].ghosts;

        HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size());)

        const size_t numCells = ccr.size();

        auto& pr = virtualStorage[ir].particles;
        const auto& pg = virtualStorage[ig].particles;
        const auto& crr = pr.cellRange();
        const auto& crg = pg.cellRange();

        {
            auto f_dim = [&](real* __restrict ptr_r, const real* __restrict ptr_g)
            {
                for (size_t ic = 0; ic < numCells; ic++)
                {
                    const size_t icr = ccr[ic];
                    const size_t icg = ccg[ic];

                    const size_t numPart = crr[icr + 1] - crr[icr];

                    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(numPart, (crg[icg + 1] - crg[icg]));)

                    real* __restrict ptr_s_r = ptr_r + crr[icr];
                    const real* __restrict ptr_s_g = ptr_g + crg[icg];

                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_r[ip] += ptr_s_g[ip];
                        }
                    }
                }
            };
            f_dim(pr.f_x.data(), pg.f_x.data());
            f_dim(pr.f_y.data(), pg.f_y.data());
            f_dim(pr.f_z.data(), pg.f_z.data());
        }
    }
    else
    {
        throw std::runtime_error("ONLY HPX OptimiseComm is enabled");
    }
}
template <FullDomainDecomposition::PackedData PACKED_DATA,
          FullDomainDecomposition::AddShift DO_SHIFT>
void FullDomainDecomposition::packCells(vec::AlignedVector<real>& sendBuf,
                                        bool commReal,
                                        size_t dir,
                                        size_t idxCommNode,
                                        Real3D const& shift)
{
    const size_t nodeStart =
        commReal ? nodeRangeReal[dir][idxCommNode] : nodeRangeGhost[dir][idxCommNode];
    const size_t nodeEnd =
        commReal ? nodeRangeReal[dir][idxCommNode + 1] : nodeRangeGhost[dir][idxCommNode + 1];
    const size_t numPart = nodeEnd - nodeStart;
    const size_t inode =
        commReal ? commNodesReal[dir][idxCommNode] : commNodesGhost[dir][idxCommNode];
    const auto& vs = virtualStorage[inode];
    const auto& cr = vs.particles.cellRange();
    const auto& vd = vdd[inode];
    const auto& cc = commReal ? vd.commCells[dir].reals : vd.commCells[dir].ghosts;

    {
        /// loop over dimensions (x,y,z)
        auto f_pack_dim = [dir, nodeStart, nodeEnd, numPart, inode, &cc, &cr, &sendBuf, &vd, this](
                              size_t dim, const real* __restrict p_ptr, real shift_v)
        {
            const size_t b_start = (nodeStart * 3) + (numPart * dim);
            real* __restrict b_ptr = sendBuf.data() + b_start;

            /// loop over cells
            size_t b_off = 0;
            for (const auto& ic : cc)
            {
                real* __restrict b_ptr_c = b_ptr + b_off;
                const real* __restrict p_ptr_c = p_ptr + cr[ic];
                const size_t npart = cr[ic + 1] - cr[ic];
                b_off += npart;

                if (HPX4ESPP_OPTIMIZE_COMM)
                {
                    const bool icOwn = (vd.cellGridInfo[ic].subNode == inode);
                    if (!icOwn) continue;
                }

                /// loop over particles
                ESPP_VEC_PRAGMAS
                for (size_t ip = 0; ip < npart; ip++)
                {
                    if (DO_SHIFT == ADD_SHIFT)
                    {
                        b_ptr_c[ip] = p_ptr_c[ip] + shift_v;
                    }
                    else
                    {
                        b_ptr_c[ip] = p_ptr_c[ip];
                    }
                }
            }
        };

        if (PACKED_DATA == PACKED_POSITIONS)
        {
            f_pack_dim(0, vs.particles.p_x.data(), shift[0]);
            f_pack_dim(1, vs.particles.p_y.data(), shift[1]);
            f_pack_dim(2, vs.particles.p_z.data(), shift[2]);
        }
        else
        {
            f_pack_dim(0, vs.particles.f_x.data(), shift[0]);
            f_pack_dim(1, vs.particles.f_y.data(), shift[1]);
            f_pack_dim(2, vs.particles.f_z.data(), shift[2]);
        }
    }
}
template void FullDomainDecomposition::packCells<FullDomainDecomposition::PACKED_POSITIONS,
                                                 FullDomainDecomposition::ADD_SHIFT>(
    vec::AlignedVector<real>& sendBuf,
    bool commReal,
    size_t dir,
    size_t idxCommNode,
    Real3D const& shift);

template void FullDomainDecomposition::packCells<FullDomainDecomposition::PACKED_FORCES,
                                                 FullDomainDecomposition::NO_SHIFT>(
    vec::AlignedVector<real>& sendBuf,
    bool commReal,
    size_t dir,
    size_t idxCommNode,
    Real3D const& shift);

template <FullDomainDecomposition::PackedData PACKED_DATA,
          FullDomainDecomposition::DataMode DATA_MODE>
void FullDomainDecomposition::unpackCells(vec::AlignedVector<real> const& recvBuf,
                                          bool commReal,
                                          size_t dir,
                                          size_t idxCommNode)
{
    const size_t nodeStart =
        commReal ? nodeRangeReal[dir][idxCommNode] : nodeRangeGhost[dir][idxCommNode];
    const size_t nodeEnd =
        commReal ? nodeRangeReal[dir][idxCommNode + 1] : nodeRangeGhost[dir][idxCommNode + 1];
    const size_t numPart = nodeEnd - nodeStart;
    const size_t inode =
        commReal ? commNodesReal[dir][idxCommNode] : commNodesGhost[dir][idxCommNode];
    auto& vs = virtualStorage[inode];
    const auto& cr = vs.particles.cellRange();
    const auto& vd = vdd[inode];
    const auto& cc = commReal ? vd.commCells[dir].reals : vd.commCells[dir].ghosts;

    {
        /// loop over dimensions (x,y,z)
        auto f_pack_dim = [dir, nodeStart, nodeEnd, numPart, inode, &cc, &cr, &recvBuf, &vd, this](
                              size_t dim, real* __restrict p_ptr)
        {
            const size_t b_start = (nodeStart * 3) + (numPart * dim);
            const real* __restrict b_ptr = recvBuf.data() + b_start;

            /// loop over cells
            size_t b_off = 0;
            for (const auto& ic : cc)
            {
                const real* __restrict b_ptr_c = b_ptr + b_off;
                real* __restrict p_ptr_c = p_ptr + cr[ic];
                const size_t npart = cr[ic + 1] - cr[ic];
                b_off += npart;

                if (HPX4ESPP_OPTIMIZE_COMM)
                {
                    const bool icOwn = (vd.cellGridInfo[ic].subNode == inode);
                    if (!icOwn) continue;
                }

                /// loop over particles
                ESPP_VEC_PRAGMAS
                for (size_t ip = 0; ip < npart; ip++)
                {
                    if (DATA_MODE == DATA_ADD)
                    {
                        p_ptr_c[ip] += b_ptr_c[ip];
                    }
                    else
                    {
                        p_ptr_c[ip] = b_ptr_c[ip];
                    }
                }
            }
        };

        if (PACKED_DATA == PACKED_POSITIONS)
        {
            f_pack_dim(0, vs.particles.p_x.data());
            f_pack_dim(1, vs.particles.p_y.data());
            f_pack_dim(2, vs.particles.p_z.data());
        }
        else
        {
            f_pack_dim(0, vs.particles.f_x.data());
            f_pack_dim(1, vs.particles.f_y.data());
            f_pack_dim(2, vs.particles.f_z.data());
        }
    }
}

template void FullDomainDecomposition::unpackCells<FullDomainDecomposition::PACKED_POSITIONS,
                                                   FullDomainDecomposition::DATA_INSERT>(
    vec::AlignedVector<real> const& recvBuf, bool commReal, size_t dir, size_t idxCommNode);

template void FullDomainDecomposition::unpackCells<FullDomainDecomposition::PACKED_FORCES,
                                                   FullDomainDecomposition::DATA_ADD>(
    vec::AlignedVector<real> const& recvBuf, bool commReal, size_t dir, size_t idxCommNode);

void FullDomainDecomposition::addGhostForcesToRealsIntra(size_t dir,
                                                         size_t ir,
                                                         size_t ig,
                                                         size_t is)
{
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        const auto& ccr = vdd[ir].commCellsOwned[dir].reals;
        const auto& ccg = vdd[ig].commCellsOwned[dir].ghosts;

        HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size());)

        const size_t numCells = ccr.size();

        auto& pr = virtualStorage[ir].particles;
        const auto& pg = virtualStorage[ig].particles;
        const auto& crr = pr.cellRange();
        const auto& crg = pg.cellRange();

        {
            auto f_dim = [&](real* __restrict ptr_r, const real* __restrict ptr_g)
            {
                for (size_t ic = 0; ic < numCells; ic++)
                {
                    const size_t icr = ccr[ic];
                    const size_t icg = ccg[ic];

                    const size_t numPart = crr[icr + 1] - crr[icr];

                    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(numPart, (crg[icg + 1] - crg[icg]));)

                    real* __restrict ptr_s_r = ptr_r + crr[icr];
                    const real* __restrict ptr_s_g = ptr_g + crg[icg];

                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_r[ip] += ptr_s_g[ip];
                        }
                    }
                }
            };
            f_dim(pr.f_x.data(), pg.f_x.data());
            f_dim(pr.f_y.data(), pg.f_y.data());
            f_dim(pr.f_z.data(), pg.f_z.data());
        }
    }
    else
    {
        throw std::runtime_error("addGhostForcesToRealsIntra ");
    }
}

template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
void FullDomainDecomposition::ghostCommunication_impl()
{
    if (ghostCommunication_impl_john)
    {
        ghostCommunication_full_impl<SIZES_FIRST, REAL_TO_GHOSTS, EXTRA_DATA>();
    }
    else
    {
        ghostCommunication_six_impl<SIZES_FIRST, REAL_TO_GHOSTS, EXTRA_DATA>();
    }
}

template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
void FullDomainDecomposition::ghostCommunication_six_impl()
{
    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(SIZES_FIRST, false);
                        HPX4ESPP_ASSERT_EQUAL(EXTRA_DATA, 0);)

    auto const& comm = *(getSystem()->comm);
    auto const rank = comm.rank();
    static std::size_t counter = 0;

    for (size_t _coord = 0; _coord < 3; ++_coord)
    {
        int coord = REAL_TO_GHOSTS ? _coord : (2 - _coord);
        const real curCoordBoxL = getSystem()->bc->getBoxL()[coord];
        const bool doPeriodic = (nodeGrid.getGridSize(coord) == 1);
        for (size_t lr = 0; lr < 2; ++lr)
        {
            size_t const dir = 2 * coord + lr;
            size_t const oppDir = 2 * coord + (1 - lr);

            Real3D shift(0, 0, 0);
            if (REAL_TO_GHOSTS)
            {
                shift[coord] = nodeGrid.getBoundary(dir) * curCoordBoxL;
            }

            auto f_interNode_pack = [this, &shift, dir]()
            {
                const real time = wallTimer.getElapsedTime();
                auto packNode = [this, dir, &shift](size_t idxCommNode)
                {
                    if (REAL_TO_GHOSTS)
                    {
                        packCells<PACKED_POSITIONS, ADD_SHIFT>(buffReal, true, dir, idxCommNode,
                                                               shift);
                    }
                    else
                    {
                        packCells<PACKED_FORCES, NO_SHIFT>(buffGhost, false, dir, idxCommNode,
                                                           SHIFT_ZERO);
                    }
                };
                size_t const numCommNodes =
                    REAL_TO_GHOSTS ? commNodesReal[dir].size() : commNodesGhost[dir].size();
                utils::parallelForLoop(0, numCommNodes, packNode);

                timeUpdateGhosts_InterNode_pack += wallTimer.getElapsedTime() - time;
            };

            auto f_interNode_comm = [this, &comm, coord, dir, oppDir, &rank]()
            {
                longint recver, sender, countRecv, countSend;
                real *buffSend, *buffRecv;
                if (REAL_TO_GHOSTS)
                {
                    recver = nodeGrid.getNodeNeighborIndex(dir);
                    sender = nodeGrid.getNodeNeighborIndex(oppDir);
                    SPDLOG_TRACE("REAL to Ghosts Sender {} receiver {}", sender, recver);
                    buffRecv = buffGhost.data();
                    buffSend = buffReal.data();
                    countRecv = nodeRangeGhost[dir].back() * vecModeFactor;
                    countSend = nodeRangeReal[dir].back() * vecModeFactor;
                }
                else
                {
                    recver = nodeGrid.getNodeNeighborIndex(oppDir);
                    sender = nodeGrid.getNodeNeighborIndex(dir);
                    SPDLOG_TRACE("NOt REAL to Ghosts Sender {} receiver {}", sender, recver);
                    buffRecv = buffReal.data();
                    buffSend = buffGhost.data();
                    countRecv = nodeRangeReal[dir].back() * vecModeFactor;
                    countSend = nodeRangeGhost[dir].back() * vecModeFactor;
                }

                SPDLOG_DEBUG("Start ghost_impl Exchange");
                spdlog::stopwatch sw;
                auto timing_logger = spdlog::get("timings");
                const real time = wallTimer.getElapsedTime();
                if (nodeGrid.getNodePosition(coord) % 2 == 0)
                {
//                    timing_logger->info("Iter {} Sending {} r2g{} from {} to {}", counter,REAL_TO_GHOSTS, dir, rank,
//                                        recver);
                    comm.send(recver, DD_COMM_TAG, buffSend, countSend);
//                    timing_logger->info("Iter {} Receiving {} r2g{} from {} to {}", counter,REAL_TO_GHOSTS, dir, recver,
//                                        rank);
                    comm.recv(sender, DD_COMM_TAG, buffRecv, countRecv);
                }
                else
                {
//                    timing_logger->info("Iter {} Receiving {} r2g{} from {} to {}", counter,REAL_TO_GHOSTS, dir, recver,
//                                        rank);
                    comm.recv(sender, DD_COMM_TAG, buffRecv, countRecv);
//                    timing_logger->info("Iter {} Sending {} r2g{} from {} to {}", counter,REAL_TO_GHOSTS, dir, rank,
//                                        recver);
                    comm.send(recver, DD_COMM_TAG, buffSend, countSend);
                }

                SPDLOG_DEBUG("End ghost_impl Exchange");
                timeUpdateGhosts_InterNode_comm += wallTimer.getElapsedTime() - time;
            };

            auto f_interNode_unpack = [this, dir]()
            {
                const real time = wallTimer.getElapsedTime();

                auto unpackNode = [this, dir](size_t idxCommNode)
                {
                    if (REAL_TO_GHOSTS)
                    {
                        unpackCells<PACKED_POSITIONS, DATA_INSERT>(buffGhost, false, dir,
                                                                   idxCommNode);
                    }
                    else
                    {
                        unpackCells<PACKED_FORCES, DATA_ADD>(buffReal, true, dir, idxCommNode);
                    }
                };
                size_t const numCommNodes =
                    REAL_TO_GHOSTS ? commNodesGhost[dir].size() : commNodesReal[dir].size();
                utils::parallelForLoop(0, numCommNodes, unpackNode);

                timeUpdateGhosts_InterNode_unpack += wallTimer.getElapsedTime() - time;
            };

            /// Inter-node block
            auto f_interNode = [this, &shift, &f_interNode_pack, &f_interNode_comm,
                                &f_interNode_unpack, coord, dir, oppDir]()
            {
                f_interNode_pack();
                f_interNode_comm();
                f_interNode_unpack();
            };

            /// Intra-node block (including shifted/periodic)
            auto f_intraNode = [this, &shift, doPeriodic, dir]()
            {
                const real time = wallTimer.getElapsedTime();

                if (HPX4ESPP_OPTIMIZE_COMM)
                {
                    auto f_pair = [this, dir, &shift](size_t i)
                    {
                        const size_t ip = i / numCommSubs;
                        const size_t is = i % numCommSubs;

                        const auto& pair = subNodePairsIntraPeriodic[dir][ip];

                        {
                            if (REAL_TO_GHOSTS)
                            {
                                // Only one MPI rank across dimension
                                copyRealsToGhostsIntra<ADD_SHIFT>(dir, std::get<0>(pair),
                                                                  std::get<1>(pair), is, shift);
                            }
                            else
                            {
                                addGhostForcesToRealsIntra(dir, std::get<0>(pair),
                                                           std::get<1>(pair), is);
                            }
                        }
                        return;
                    };
                    size_t const numPairs = subNodePairsIntraPeriodic[dir].size();
                    utils::parallelForLoop(0, numPairs * numCommSubs, f_pair);
                }
                else
                {
                    auto f_pair = [this, dir, &shift](size_t i)
                    {
                        const size_t ip = i / numCommSubs;
                        const size_t is = i % numCommSubs;

                        const auto& pair = subNodePairsIntra[dir][ip];

                        if (REAL_TO_GHOSTS)
                        {
                            if (std::get<2>(pair))
                                copyRealsToGhostsIntra<ADD_SHIFT>(dir, std::get<0>(pair),
                                                                  std::get<1>(pair), is, shift);
                            else
                            {
                                copyRealsToGhostsIntra<NO_SHIFT>(dir, std::get<0>(pair),
                                                                 std::get<1>(pair), is, SHIFT_ZERO);
                            }
                        }
                        else
                        {
                            addGhostForcesToRealsIntra(dir, std::get<0>(pair), std::get<1>(pair),
                                                       is);
                        }
                    };
                    size_t const numPairs = subNodePairsIntra[dir].size();
                    utils::parallelForLoop(0, numPairs * numCommSubs, f_pair);
                }

                /// NOTE: Time only if comm is involved
                // if (!doPeriodic)
                timeUpdateGhosts_IntraNode += wallTimer.getElapsedTime() - time;
            };

            if ((hpx::threads::get_self_ptr() != nullptr) && !doPeriodic && commAsync)
            {
                {
                    f_interNode_pack();
                    std::array<hpx::future<void>, 2> ret = {hpx::async(f_interNode_comm),
                                                            hpx::async(f_intraNode)};
                    hpx::wait_all(ret);
                    f_interNode_unpack();
                }
            }
            else
            {
                if (!doPeriodic) f_interNode();
                f_intraNode();
            }
        }
    }
    counter++;

}

template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
void FullDomainDecomposition::ghostCommunication_full_impl()
{
    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(SIZES_FIRST, false);
                        HPX4ESPP_ASSERT_EQUAL(EXTRA_DATA, 0);)

    auto const& comm = *(getSystem()->comm);

    std::array<hpx::future<void>, 26> futures;
    std::array<hpx::future<void>, 26> send_futures;

    static int counter = 0;

    //    if (counter < 2)
    //    {
    //        using json = nlohmann::json;
    //        json before_exchange;
    //        for (size_t particle_list = 0; particle_list < getLocalCells().size();
    //        ++particle_list)
    //        {
    //            for (size_t particle_num = 0;
    //                 particle_num < getLocalCells()[particle_list]->particles.size();
    //                 ++particle_num)
    //            {
    //                const Particle& particle =
    //                getLocalCells()[particle_list]->particles[particle_num];
    //
    //                before_exchange[particle_list][particle_num] = json::array(
    //                    {{"id", particle.id()},
    //                     {"velocity", particle.velocity()[0], particle.velocity()[1],
    //                      particle.velocity()[2]},
    //                     {"position", particle.position()[0], particle.position()[1],
    //                      particle.position()[2]},
    //                     {"force", particle.force()[0], particle.force()[1],
    //                     particle.force()[2]}});
    //            }
    //        }
    //        if (REAL_TO_GHOSTS)
    //        {
    //            counter++;
    //        }
    //        std::ofstream particle_status(
    //            spdlog::fmt_lib::format("BeforeBuffSend iteration {} Real2Ghost {} rank {}.json",
    //                                    counter, REAL_TO_GHOSTS, hpx::get_locality_id()));
    //        particle_status << before_exchange;
    //    }

    for (auto neighbour : John::NeighbourEnumIterable)
    {
        Real3D shift(0, 0, 0);
        neighbour = REAL_TO_GHOSTS ? neighbour : opposite_neighbour(neighbour);
        const auto direction = enum_to_direction(opposite_neighbour(neighbour));
        const bool doPeriodic = full_grid.is_periodic_neighbour(neighbour);

        if (REAL_TO_GHOSTS && doPeriodic)
        {
            for (int coord = 0; coord < 3; ++coord)
            {
                const real curCoordBoxL = getSystem()->bc->getBoxL()[coord];
                if (direction[coord] != 0)
                {
                    shift[coord] = direction[coord] * curCoordBoxL;
                }
            }
        }

        if (full_grid.neighbour_is_self(neighbour))
        {
            ghostCommunication_impl_full_intra_node(shift, true, neighbour, REAL_TO_GHOSTS);
            continue;
        }


        send_futures[neighbour] = hpx::async(
            [this, shift, neighbour]() {
                ghostCommunication_impl_send_routine<REAL_TO_GHOSTS>(shift, neighbour,
                                                                     iteration_number);
            });


        futures[neighbour] = hpx::async(
            [this, shift, neighbour]() {
                ghostCommunication_impl_receive_routine<REAL_TO_GHOSTS>(neighbour,
                                                                        iteration_number);
            });
        // Doing an intra comm
    }
    hpx::wait_all(futures);
    hpx::wait_all(send_futures);
    SPDLOG_TRACE("ghostCommunication_impl function complete");
}

template void FullDomainDecomposition::ghostCommunication_impl<false, true, 0>();
template void FullDomainDecomposition::ghostCommunication_impl<false, false, 0>();

template void FullDomainDecomposition::ghostCommunication_six_impl<false, true, 0>();
template void FullDomainDecomposition::ghostCommunication_six_impl<false, false, 0>();
template void FullDomainDecomposition::ghostCommunication_full_impl<false, true, 0>();
template void FullDomainDecomposition::ghostCommunication_full_impl<false, false, 0>();
template void
FullDomainDecomposition::fullCopyRealsToGhostsIntra<FullDomainDecomposition::ADD_SHIFT>(
    John::NeighbourRelation neighbour, size_t ir, size_t ig, size_t is, Real3D const& shift);

template void
FullDomainDecomposition::fullCopyRealsToGhostsIntra<FullDomainDecomposition::NO_SHIFT>(
    John::NeighbourRelation neighbour, size_t ir, size_t ig, size_t is, Real3D const& shift);

template <FullDomainDecomposition::PackedData PACKED_DATA,
          FullDomainDecomposition::AddShift DO_SHIFT>
void FullDomainDecomposition::packCells_full(vec::AlignedVector<real>& sendBuf,
                                             bool commReal,
                                             John::NeighbourRelation neighbour,
                                             size_t idxCommNode,
                                             Real3D const& shift)
{
    const size_t nodeStart = commReal ? fullNodeRangeReal[neighbour][idxCommNode]
                                      : fullNodeRangeGhost[neighbour][idxCommNode];
    const size_t nodeEnd = commReal ? fullNodeRangeReal[neighbour][idxCommNode + 1]
                                    : fullNodeRangeGhost[neighbour][idxCommNode + 1];
    const size_t numPart = nodeEnd - nodeStart;
    const size_t inode = commReal ? fullCommNodesReal[neighbour][idxCommNode]
                                  : fullCommNodesGhost[neighbour][idxCommNode];
    const auto& vs = virtualStorage[inode];
    const auto& cr = vs.particles.cellRange();
    const auto& vd = vdd[inode];
    const auto& cc =
        commReal ? vd.fullCommCells[neighbour].reals : vd.fullCommCells[neighbour].ghosts;

    using json = nlohmann::json;

    using json = nlohmann::json;

    json particle_status;

    if (neighbour == 0 || neighbour == 1)
    {
        const size_t six_nodeStart = commReal ? nodeRangeReal[neighbour][idxCommNode]
                                              : nodeRangeGhost[neighbour][idxCommNode];
        const size_t six_nodeEnd = commReal ? nodeRangeReal[neighbour][idxCommNode + 1]
                                            : nodeRangeGhost[neighbour][idxCommNode + 1];
        const size_t six_numPart = nodeEnd - nodeStart;
        const size_t six_inode = commReal ? commNodesReal[neighbour][idxCommNode]
                                          : commNodesGhost[neighbour][idxCommNode];
        const auto& six_vs = virtualStorage[inode];
        const auto& six_cr = vs.particles.cellRange();
        const auto& six_vd = vdd[inode];
        const auto& six_cc =
            commReal ? vd.commCells[neighbour].reals : vd.commCells[neighbour].ghosts;

        //        HPX4ESPP_ASSERT_EQUAL(nodeStart, six_nodeStart);
        //        HPX4ESPP_ASSERT_EQUAL(nodeEnd, six_nodeEnd);
        //        HPX4ESPP_ASSERT_EQUAL(numPart, six_numPart);
        //        HPX4ESPP_ASSERT_EQUAL(inode, six_inode);
    }
    {
        /// loop over dimensions (x,y,z)
        auto f_pack_dim = [nodeStart, numPart, inode, &cc, &cr, &sendBuf, &particle_status, &vd,
                           this](size_t dim, const real* __restrict p_ptr, real shift_v)
        {
            const size_t b_start = (nodeStart * 3) + (numPart * dim);
            real* __restrict b_ptr = sendBuf.data() + b_start;

            /// loop over cells
            size_t b_off = 0;
            for (const auto& ic : cc)
            {
                real* __restrict b_ptr_c = b_ptr + b_off;
                const real* __restrict p_ptr_c = p_ptr + cr[ic];
                const size_t npart = cr[ic + 1] - cr[ic];
                b_off += npart;

                if (HPX4ESPP_OPTIMIZE_COMM)
                {
                    const bool icOwn = (vd.cellGridInfo[ic].subNode == inode);
                    HPX4ESPP_ASSERT_EQUAL(icOwn, true);
                    if (!icOwn) continue;  // TODO: Maybe Ownership ?
                }

                /// loop over particles
                ESPP_VEC_PRAGMAS
                for (size_t ip = 0; ip < npart; ip++)
                {
                    if (DO_SHIFT == ADD_SHIFT)
                    {
                        b_ptr_c[ip] = p_ptr_c[ip] + shift_v;
                    }
                    else
                    {
                        b_ptr_c[ip] = p_ptr_c[ip];
                    }
                }
            }
        };

        if (PACKED_DATA == PACKED_POSITIONS)
        {
            f_pack_dim(0, vs.particles.p_x.data(), shift[0]);
            f_pack_dim(1, vs.particles.p_y.data(), shift[1]);
            f_pack_dim(2, vs.particles.p_z.data(), shift[2]);
        }
        else
        {
            f_pack_dim(0, vs.particles.f_x.data(), shift[0]);
            f_pack_dim(1, vs.particles.f_y.data(), shift[1]);
            f_pack_dim(2, vs.particles.f_z.data(), shift[2]);
        }
    }
}

template void FullDomainDecomposition::packCells_full<FullDomainDecomposition::PACKED_POSITIONS,
                                                      FullDomainDecomposition::ADD_SHIFT>(
    vec::AlignedVector<real>& sendBuf,
    bool commReal,
    John::NeighbourRelation neighbour,
    size_t idxCommNode,
    Real3D const& shift);

template void FullDomainDecomposition::packCells_full<FullDomainDecomposition::PACKED_FORCES,
                                                      FullDomainDecomposition::NO_SHIFT>(
    vec::AlignedVector<real>& sendBuf,
    bool commReal,
    John::NeighbourRelation neighbour,
    size_t idxCommNode,
    Real3D const& shift);

template hpx::future<void> FullDomainDecomposition::ghostCommunication_impl_send_routine<true>(
    Real3D shift, John::NeighbourRelation neighbour_to_send_to, std::size_t current_iteration);
template hpx::future<void> FullDomainDecomposition::ghostCommunication_impl_send_routine<false>(
    Real3D shift, John::NeighbourRelation neighbour_to_send_to, std::size_t current_iteration);

template <bool REAL_TO_GHOSTS>
hpx::future<void> FullDomainDecomposition::ghostCommunication_impl_send_routine(
    Real3D shift, John::NeighbourRelation neighbour_to_send_to, std::size_t current_iteration)
{
    const real time = wallTimer.getElapsedTime();
    auto sendBuffer = vec::AlignedVector<real>();
    sendBuffer.resize(REAL_TO_GHOSTS ? fullPreallocReal : fullPreallocGhost);
    size_t const numCommNodes = REAL_TO_GHOSTS ? fullCommNodesReal[neighbour_to_send_to].size()
                                               : fullCommNodesGhost[neighbour_to_send_to].size();
    for (size_t idxCommNode = 0; idxCommNode < numCommNodes; ++idxCommNode)
    {
        // TODO: Lamda doesn't like wrapping this
        if (REAL_TO_GHOSTS)
        {
            packCells_full<PACKED_POSITIONS, ADD_SHIFT>(sendBuffer, true, neighbour_to_send_to,
                                                        idxCommNode, shift);
        }
        else
        {
            packCells_full<PACKED_FORCES, NO_SHIFT>(sendBuffer, false, neighbour_to_send_to,
                                                    idxCommNode, SHIFT_ZERO);
        }
    }
    timeUpdateGhosts_InterNode_pack += wallTimer.getElapsedTime() - time;

    SPDLOG_TRACE("Sending side {} size {}", John::relation_to_string(neighbour_to_send_to),
                 sendBuffer.size());

    //    using json = nlohmann::json;
    //    json particle_status;
    //
    //    static int counter = 0;
    //    static bool alternate = false;
    //    if ((counter == 0 || counter == 1) && neighbour_to_send_to<2 )
    //    {
    //        std::ofstream file(spdlog::fmt_lib::format(
    //                               "sending_buff neighbour {} counter {} rank {}
    //                               real2ghost{}.json",
    //                               John::relation_to_string(neighbour_to_send_to), counter,
    //                               hpx::get_locality_id(), REAL_TO_GHOSTS),
    //                           std::ios::trunc);
    //        particle_status = sendBuffer;
    //        file << particle_status;
    //    }
    //    if (neighbour_to_send_to == 1 && alternate)
    //    {
    //        counter++;
    //    }
    //    alternate = !alternate;

    std::size_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

//    sending_json.do_action(
//        [now, current_iteration](json& json_data)
//        {
//            auto iteration_key = fmt::format("Send {}", current_iteration);
//            if (!json_data.contains(iteration_key))
//            {
//                json_data[iteration_key] = now;
//            }
//            else
//            {
//                json_data[iteration_key] =
//                    std::min(json_data[iteration_key].get<std::size_t>(), now);
//            }
//        });

    return ch_ghostCommunication_impl.get_send_channel(neighbour_to_send_to)
        .set(hpx::launch::async, std::move(sendBuffer));
}
template <bool REAL_TO_GHOSTS>
void FullDomainDecomposition::ghostCommunication_impl_receive_routine(
    John::NeighbourRelation filling_relation, std::size_t current_iteration)
{
    auto receive_from = opposite_neighbour(filling_relation);
    auto received_data =
        ch_ghostCommunication_impl.get_receive_channel(receive_from).get(hpx::launch::sync);
    std::size_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//    sending_json.do_action(
//        [now, current_iteration](json& json_data)
//        {
//            auto iteration_key = fmt::format("Receive {}", current_iteration);
//            if (!json_data.contains(iteration_key))
//            {
//                json_data[iteration_key] = now;
//            }
//            else
//            {
//                json_data[iteration_key] =
//                    std::max(json_data[iteration_key].get<std::size_t>(), now);
//            }
//        });

    SPDLOG_TRACE("Received side {} from {} size {}", John::relation_to_string(filling_relation),
                 John::relation_to_string(receive_from), received_data.size());

    auto f_interNode_unpack = [this, received_data = std::move(received_data), filling_relation]()
    {
        const real time = wallTimer.getElapsedTime();

        auto unpackNode = [this, &received_data, filling_relation](size_t idxCommNode)
        {
            if (REAL_TO_GHOSTS)
            {
                unpackCells_full<PACKED_POSITIONS, DATA_INSERT>(received_data, false,
                                                                filling_relation, idxCommNode);
            }
            else
            {
                unpackCells_full<PACKED_FORCES, DATA_ADD>(received_data, true, filling_relation,
                                                          idxCommNode);
            }
        };
        size_t const numCommNodes = REAL_TO_GHOSTS ? fullCommNodesGhost[filling_relation].size()
                                                   : fullCommNodesReal[filling_relation].size();
        utils::parallelForLoop(0, numCommNodes, unpackNode);
        timeUpdateGhosts_InterNode_unpack += wallTimer.getElapsedTime() - time;
    };
}
template void FullDomainDecomposition::ghostCommunication_impl_receive_routine<true>(
    John::NeighbourRelation filling_relation, std::size_t current_iteration);
template void FullDomainDecomposition::ghostCommunication_impl_receive_routine<false>(
    John::NeighbourRelation filling_relation, std::size_t current_iteration);
static hpx::mutex unpack_mtx;

void FullDomainDecomposition::ghostCommunication_impl_full_intra_node(
    Real3D shift, bool doPeriodic, John::NeighbourRelation neighbour, bool real_to_ghosts)
{
    unpack_mtx.lock();
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        size_t const numPairs = fullSubNodePairsIntraPeriodic[neighbour].size();
        for (size_t i = 0; i < numPairs * numCommSubs; ++i)
        {
            const size_t ip = i / numCommSubs;
            const size_t is = i % numCommSubs;

            const auto& pair = fullSubNodePairsIntraPeriodic[neighbour][ip];

            if (real_to_ghosts)
            {
                // Only one MPI rank across dimension
                fullCopyRealsToGhostsIntra<ADD_SHIFT>(neighbour, std::get<0>(pair),
                                                      std::get<1>(pair), is, shift);
            }
            else
            {
                fullAddGhostForcesToRealsIntra(neighbour, std::get<0>(pair), std::get<1>(pair), is);
            }
        }
    }
    unpack_mtx.unlock();
}

template <FullDomainDecomposition::PackedData PACKED_DATA,
          FullDomainDecomposition::DataMode DATA_MODE>
void FullDomainDecomposition::unpackCells_full(vec::AlignedVector<real> const& recvBuf,
                                               bool commReal,
                                               John::NeighbourRelation neighbour,
                                               size_t idxCommNode)
{
    const size_t nodeStart = commReal ? fullNodeRangeReal[neighbour][idxCommNode]
                                      : fullNodeRangeGhost[neighbour][idxCommNode];
    const size_t nodeEnd = commReal ? fullNodeRangeReal[neighbour][idxCommNode + 1]
                                    : fullNodeRangeGhost[neighbour][idxCommNode + 1];
    const size_t numPart = nodeEnd - nodeStart;
    const size_t inode = commReal ? fullCommNodesReal[neighbour][idxCommNode]
                                  : fullCommNodesGhost[neighbour][idxCommNode];
    auto& vs = virtualStorage[inode];
    const auto& cr = vs.particles.cellRange();
    const auto& vd = vdd[inode];
    const auto& cc =
        commReal ? vd.fullCommCells[neighbour].reals : vd.fullCommCells[neighbour].ghosts;

    if (neighbour == 0 || neighbour == 1)
    {
        const size_t six_nodeStart = commReal ? nodeRangeReal[neighbour][idxCommNode]
                                              : nodeRangeGhost[neighbour][idxCommNode];
        const size_t six_nodeEnd = commReal ? nodeRangeReal[neighbour][idxCommNode + 1]
                                            : nodeRangeGhost[neighbour][idxCommNode + 1];
        const size_t six_numPart = six_nodeEnd - six_nodeStart;
        const size_t six_inode = commReal ? commNodesReal[neighbour][idxCommNode]
                                          : commNodesGhost[neighbour][idxCommNode];

        auto& six_vs = virtualStorage[inode];
        const auto& six_cr = vs.particles.cellRange();
        const auto& six_vd = vdd[inode];
        const auto& six_cc =
            commReal ? vd.commCells[neighbour].reals : vd.commCells[neighbour].ghosts;

        HPX4ESPP_ASSERT_EQUAL(six_inode, inode);
        HPX4ESPP_ASSERT_EQUAL(six_numPart, numPart);
        HPX_ASSERT(six_cc == cc);
    }

    unpack_mtx.lock();
    {
        /// loop over dimensions (x,y,z)
        auto f_pack_dim = [nodeStart, numPart, inode, &cc, &cr, &recvBuf, &vd, this](
                              size_t dim, real* __restrict p_ptr)
        {
            const size_t b_start = (nodeStart * 3) + (numPart * dim);
            const real* __restrict b_ptr = recvBuf.data() + b_start;

            /// loop over cells
            size_t b_off = 0;
            for (const auto& ic : cc)
            {
                const real* __restrict b_ptr_c = b_ptr + b_off;
                real* __restrict p_ptr_c = p_ptr + cr[ic];
                const size_t npart = cr[ic + 1] - cr[ic];
                b_off += npart;

                if (HPX4ESPP_OPTIMIZE_COMM)
                {
                    const bool icOwn = (vd.cellGridInfo[ic].subNode == inode);
                    HPX4ESPP_ASSERT_EQUAL(icOwn, true);
                    if (!icOwn) continue;  // TODO: Maybe ownership ?
                }

                /// loop over particles
                ESPP_VEC_PRAGMAS
                for (size_t ip = 0; ip < npart; ip++)
                {
                    if (DATA_MODE == DATA_ADD)
                    {
                        p_ptr_c[ip] += b_ptr_c[ip];
                    }
                    else
                    {
                        p_ptr_c[ip] = b_ptr_c[ip];
                    }
                }
            }
        };

        if (PACKED_DATA == PACKED_POSITIONS)
        {
            f_pack_dim(0, vs.particles.p_x.data());
            f_pack_dim(1, vs.particles.p_y.data());
            f_pack_dim(2, vs.particles.p_z.data());
        }
        else
        {
            f_pack_dim(0, vs.particles.f_x.data());
            f_pack_dim(1, vs.particles.f_y.data());
            f_pack_dim(2, vs.particles.f_z.data());
        }
    }
    unpack_mtx.unlock();
}

template void FullDomainDecomposition::unpackCells_full<FullDomainDecomposition::PACKED_POSITIONS,
                                                        FullDomainDecomposition::DATA_INSERT>(
    vec::AlignedVector<real> const& recvBuf,
    bool commReal,
    John::NeighbourRelation neighbour,
    size_t idxCommNode);

template void FullDomainDecomposition::unpackCells_full<FullDomainDecomposition::PACKED_FORCES,
                                                        FullDomainDecomposition::DATA_ADD>(
    vec::AlignedVector<real> const& recvBuf,
    bool commReal,
    John::NeighbourRelation neighbour,
    size_t idxCommNode);

/// Partitions a vector of sizes to its corresponding bitset offsets
/// \tparam ALIGNED Align to cache line
/// \param sizes input sizes
/// \param sizeTotal Total Size
/// \return List where each pair forms a range, the last value is used to indicate the actual
/// total size
template <bool ALIGNED>
auto sizes_to_ranges(std::vector<longint> const& sizes, std::size_t sizeTotal)
{
    const std::size_t nsize = sizes.size();
    std::vector<longint> ranges;
    ranges.resize(nsize + 1);
    longint total = 0;
    for (int i = 0; i < nsize; i++)
    {
        ranges[i] = total;
        if (ALIGNED)
        {
            total += ROUND_TO_CACHE_LINE(sizeTotal * sizes[i]);
        }
        else
        {
            total += sizeTotal * sizes[i];
        }
    }
    ranges[nsize] = total;
    return ranges;
}

template <FullDomainDecomposition::AddShift DO_SHIFT>
void FullDomainDecomposition::copyRealsToGhostsIntra(
    size_t dir, size_t ir, size_t ig, size_t is, Real3D const& shift)
{
    if (HPX4ESPP_OPTIMIZE_COMM)
    {
        const auto& ccr = vdd[ir].commCellsOwned[dir].reals;
        const auto& ccg = vdd[ig].commCellsOwned[dir].ghosts;
        HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(ccr.size(), ccg.size());)
        const size_t numCells = ccr.size();

        const auto& pr = virtualStorage[ir].particles;
        auto& pg = virtualStorage[ig].particles;
        const auto& crr = pr.cellRange();
        const auto& crg = pg.cellRange();

        {
            auto f_dim = [&](const real* __restrict ptr_r, real* __restrict ptr_g, real shift_v)
            {
                for (size_t ic = 0; ic < numCells; ic++)
                {
                    const size_t icr = ccr[ic];
                    const size_t icg = ccg[ic];

                    const size_t numPart = crr[icr + 1] - crr[icr];
                    HPX4ESPP_DDVI_DEBUG(HPX4ESPP_ASSERT_EQUAL(numPart, (crg[icg + 1] - crg[icg]));)

                    const real* __restrict ptr_s_r = ptr_r + crr[icr];
                    real* __restrict ptr_s_g = ptr_g + crg[icg];

                    if (DO_SHIFT)
                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_g[ip] = ptr_s_r[ip] + shift_v;
                        }
                    }
                    else
                    {
                        ESPP_VEC_PRAGMAS
                        for (size_t ip = 0; ip < numPart; ip++)
                        {
                            ptr_s_g[ip] = ptr_s_r[ip];
                        }
                    }
                }
            };
            f_dim(pr.p_x.data(), pg.p_x.data(), shift[0]);
            f_dim(pr.p_y.data(), pg.p_y.data(), shift[1]);
            f_dim(pr.p_z.data(), pg.p_z.data(), shift[2]);
        }
    }
    else
    {
        //
        throw std::runtime_error("ONLY optimse comm");
    }
}

template void FullDomainDecomposition::copyRealsToGhostsIntra<FullDomainDecomposition::ADD_SHIFT>(
    size_t dir, size_t ir, size_t ig, size_t is, Real3D const& shift);

template void FullDomainDecomposition::copyRealsToGhostsIntra<FullDomainDecomposition::NO_SHIFT>(
    size_t dir, size_t ir, size_t ig, size_t is, Real3D const& shift);

template <bool ALIGNED>
void FullDomainDecomposition::exchangeGhosts_receive_routine(
    const John::NeighbourRelation neighbour_to_fill, const int extradata)
{
    constexpr size_t sizePos = sizeof(ParticlePosition);
    constexpr size_t sizePrp = sizeof(ParticleProperties);
    constexpr size_t sizeTotal = sizePos + sizePrp;
    auto f_fillBufView = [](auto& bufview, const auto& buffixed, const auto& ranges)
    {
        const size_t size = ranges.size() - 1;
        if (bufview.size() < size) bufview.resize(size);
        for (size_t i = 0; i < size; i++)
        {
            bufview[i].view(buffixed, ranges[i], ranges[i + 1]);
        }
    };

    auto neighbour_to_request = opposite_neighbour(neighbour_to_fill);

    auto receive_sizes_fut = ch_exchangeGhosts_impl_sizes.get_receive_channel(neighbour_to_request)
                                 .get(hpx::launch::async);

    auto receive_buffer_fut =
        ch_exchangeGhosts_impl_particles.get_receive_channel(neighbour_to_request)
            .get(hpx::launch::async);

    auto receive_sizes = receive_sizes_fut.get();
    SPDLOG_TRACE("receive_sizes {} from {}", receive_sizes.size(),
                 John::relation_to_string(neighbour_to_fill));
    auto receive_ranges = sizes_to_ranges<ALIGNED>(receive_sizes, sizeTotal);
    auto f_resize = [this, &receive_sizes, &neighbour_to_fill](size_t i)
    { fullCommCells[neighbour_to_fill].ghosts[i]->particles.resize(receive_sizes[i]); };
    auto receive_buffer = receive_buffer_fut.get();
    size_t const numCommCells = fullCommCells[neighbour_to_fill].ghosts.size();
    utils::parallelForLoop(0, numCommCells, f_resize);
    std::vector<InBufferView> local_in_buffer;
    auto in_buffer = BufferFixed((*(getSystem()->comm)));

    in_buffer.set_buffer(receive_buffer.data(), receive_buffer.size(), false);

    f_fillBufView(local_in_buffer, in_buffer, receive_ranges);
    static hpx::mutex m;
    auto f_unpack = [](Cell& _ghosts, InBufferView& buf, int extradata)
    {
        ParticleList& ghosts = _ghosts.particles;
        for (auto& ghost : ghosts)
        {
            buf.read(ghost, extradata);
            /// NOTE: multithreaded particle map not implemented
            // if (extradata & DATA_PROPERTIES) {
            //   updateInLocalParticles(&(*dst), true);
            // }
            ghost.ghost() = true;
        }
    };
    const size_t size = fullCommCells[neighbour_to_fill].ghosts.size();
    m.lock();

    utils::parallelForLoop(
        0, size,
        [this, &f_unpack, &extradata, neighbour_to_fill, &local_in_buffer](size_t i)
        { f_unpack(*fullCommCells[neighbour_to_fill].ghosts[i], local_in_buffer[i], extradata); });
    m.unlock();
}

template <bool ALIGNED>
void FullDomainDecomposition::exchangeGhosts_send_routine(
    const John::NeighbourRelation neighbour_to_send_to, const Real3D shift, const int extradata)
{
    constexpr size_t sizePos = sizeof(ParticlePosition);
    constexpr size_t sizePrp = sizeof(ParticleProperties);
    constexpr size_t sizeTotal = sizePos + sizePrp;
    auto f_fillBufView = [](auto& bufview, const auto& buffixed, const auto& ranges)
    {
        const size_t size = ranges.size() - 1;
        if (bufview.size() < size) bufview.resize(size);
        for (size_t i = 0; i < size; i++)
        {
            bufview[i].view(buffixed, ranges[i], ranges[i + 1]);
        }
    };

    std::vector<OutBufferView> current_outBufView;

    std::vector<longint> sendSizes, sendRanges;
    sendSizes.reserve(fullCommCells[neighbour_to_send_to].reals.size());
    for (auto& real : fullCommCells[neighbour_to_send_to].reals)
    {
        sendSizes.push_back(real->particles.size());
    }
    sendRanges = sizes_to_ranges<ALIGNED>(sendSizes, sizeTotal);

    SPDLOG_TRACE("SendSizes {} direction publishing {}", sendSizes.size(),
                 John::relation_to_string(neighbour_to_send_to));
    ch_exchangeGhosts_impl_sizes.get_send_channel(neighbour_to_send_to).set(std::move(sendSizes));
    const auto sendTotalBytes = sendRanges.back();
    outBuffers_futs[neighbour_to_send_to].wait();
    auto& neighbour_outBuf = outBuffers[neighbour_to_send_to];
    neighbour_outBuf.allocate(sendTotalBytes);
    f_fillBufView(current_outBufView, neighbour_outBuf, sendRanges);
    auto f_pack = [](OutBufferView& buf, Cell& _reals, int extradata, const Real3D& shift)
    {
        ParticleList& reals = _reals.particles;
        for (auto& real : reals)
        {
            buf.write(real, extradata, shift);
        }
    };
    const size_t size = fullCommCells[neighbour_to_send_to].reals.size();
    utils::parallelForLoop(
        0, size,
        [this, &f_pack, &extradata, &shift, &current_outBufView, neighbour_to_send_to](size_t i)
        {
            f_pack(current_outBufView[i], *fullCommCells[neighbour_to_send_to].reals[i], extradata,
                   shift);
        });

    hpx::serialization::serialize_buffer<char> sending_buffer(
        neighbour_outBuf.getBuf(), neighbour_outBuf.getUsedSize(),
        hpx::serialization::serialize_buffer<char>::reference);

    outBuffers_futs[neighbour_to_send_to] =
        ch_exchangeGhosts_impl_particles.get_send_channel(neighbour_to_send_to)
            .set(hpx::launch::async, sending_buffer);
}

template <bool ALIGNED>
void FullDomainDecomposition::exchangeGhosts_six_impl()
{
    // Decompose HPX
    auto const& comm = *(getSystem()->comm);
    const int rank = comm.rank();

    constexpr bool sizesFirst = true;
    constexpr bool realToGhosts = true;
    constexpr int extradata = DATA_PROPERTIES;

    constexpr size_t sizePos = sizeof(ParticlePosition);
    constexpr size_t sizePrp = sizeof(ParticleProperties);
    constexpr size_t sizeTotal = sizePos + sizePrp;
    static int iteration = 0;
    iteration++;
    auto f_fillBufView = [](auto& bufview, const auto& buffixed, const auto& ranges)
    {
        const size_t size = ranges.size() - 1;
        if (bufview.size() < size) bufview.resize(size);
        for (size_t i = 0; i < size; i++)
        {
            bufview[i].view(buffixed, ranges[i], ranges[i + 1]);
        }
    };

    // LOG4ESPP_DEBUG(logger, "do ghost communication " << (sizesFirst ? "with sizes " : "")
    //       << (realToGhosts ? "reals to ghosts " : "ghosts to reals ") << extradata);

    /* direction loop: x, y, z.
      Here we could in principle build in a one sided ghost
      communication, simply by taking the lr loop only over one
      value. */
    for (int _coord = 0; _coord < 3; ++_coord)
    {
        /* inverted processing order for ghost force communication,
          since the corner ghosts have to be collected via several
          nodes. We now add back the corner ghost forces first again
          to ghost forces, which only eventually go back to the real
          particle.
        */
        int coord = realToGhosts ? _coord : (2 - _coord);
        real curCoordBoxL = getSystem()->bc->getBoxL()[coord];

        // lr loop: left right
        for (int lr = 0; lr < 2; ++lr)
        {
            int dir = 2 * coord + lr;
            int oppositeDir = 2 * coord + (1 - lr);

            Real3D shift(0, 0, 0);

            shift[coord] = nodeGrid.getBoundary(dir) * curCoordBoxL;

            // LOG4ESPP_DEBUG(logger, "direction " << dir);

            if (nodeGrid.getGridSize(coord) == 1)
            {
                const real time = wallTimer.getElapsedTime();

                // LOG4ESPP_DEBUG(logger, "local communication");
                //                std::cout<<"nodeGrid.getGridSize(coord) == 1" << std::endl;
                SPDLOG_TRACE("nodeGrid.getGridSize(coord) was 1 doing local comm");

                // copy operation, we have to receive as many cells as we send
                if (commCells[dir].ghosts.size() != commCells[dir].reals.size())
                {
                    throw std::runtime_error(
                        "DomainDecomposition::doGhostCommunication: send/recv cell structure "
                        "mismatch during local copy");
                }

                auto f_copyRealsToGhosts = [this, &shift, &extradata, &dir](size_t i) {
                    copyRealsToGhosts(*commCells[dir].reals[i], *commCells[dir].ghosts[i],
                                      extradata, shift);
                };
                size_t const numCommCells = commCells[dir].ghosts.size();
                utils::parallelForLoop(0, numCommCells, f_copyRealsToGhosts);

                timeExcGhosts_IntraNode += wallTimer.getElapsedTime() - time;
            }
            else
            {
                std::vector<longint> sendSizes, recvSizes, sendRanges, recvRanges;

                longint sendTotalBytes, recvTotalBytes;

                // exchange size information, if necessary
                const auto channel_name =
                    hpx::util::format("exchange/ghosts/impl/{}/{}/{}", lr, coord, iteration);

                //                std::cout << hpx::util::format("{} to {} : Sending Information
                //                lr={} coord={}",
                //                                               rank,
                //                                               nodeGrid.getNodeNeighborIndex(oppositeDir),
                //                                               lr, coord)
                //                          << std::endl;
                hpx::distributed::channel<std::vector<int>> my_send_channel(hpx::find_here());
                hpx::register_with_basename(channel_name, my_send_channel, rank);
                auto my_recv_channel =
                    hpx::find_from_basename<hpx::distributed::channel<std::vector<int>>>(
                        channel_name, nodeGrid.getNodeNeighborIndex(oppositeDir));
                {
                    const real time = wallTimer.getElapsedTime();

                    // prepare buffers
                    sendSizes.reserve(commCells[dir].reals.size());
                    for (auto& real : commCells[dir].reals)
                    {
                        sendSizes.push_back(real->particles.size());
                    }
                    recvSizes.resize(commCells[dir].ghosts.size());
                    SPDLOG_DEBUG("Start exchangeGhost Sizes");
                    SPDLOG_TRACE("GHOSTExchangeimpl,Sender {} Reciever {}",
                                 nodeGrid.getNodeNeighborIndex(dir),
                                 nodeGrid.getNodeNeighborIndex(oppositeDir));
                    spdlog::stopwatch sw;
                    auto send_fut = my_send_channel.set(hpx::launch::async, sendSizes);
                    //                    auto timing_logger = spdlog::get("timings");

                    //                    timing_logger->info("Sending {} from {} to {}",dir,rank,
                    //                    nodeGrid.getNodeNeighborIndex(oppositeDir));
                    hpx::future<std::vector<longint>> send_ranges_future = hpx::async(
                        hpx::launch::async, sizes_to_ranges<ALIGNED>, sendSizes, sizeTotal);

                    //                    timing_logger->info("Receiving {} from {} to {}",dir,
                    //                    nodeGrid.getNodeNeighborIndex(dir),rank);
                    recvSizes = my_recv_channel.get(hpx::launch::sync);

                    SPDLOG_DEBUG("END exchangeGhost Sizes {}", sw);

                    // resize according to received information
                    auto f_resize = [this, &recvSizes, &dir](size_t i)
                    { commCells[dir].ghosts[i]->particles.resize(recvSizes[i]); };
                    size_t const numCommCells = commCells[dir].ghosts.size();
                    utils::parallelForLoop(0, numCommCells, f_resize);

                    sendRanges = send_ranges_future.get();
                    recvRanges = sizes_to_ranges<ALIGNED>(recvSizes, sizeTotal);
                    sendTotalBytes = sendRanges.back();
                    recvTotalBytes = recvRanges.back();

                    timeExcGhosts_InterNode_sizes += wallTimer.getElapsedTime() - time;
                }

                // prepare send and receive buffers
                longint receiver, sender;
                {
                    const real time = wallTimer.getElapsedTime();

                    outBuf.allocate(sendTotalBytes);
                    f_fillBufView(outBufView, outBuf, sendRanges);
                    inBuf.allocate(recvTotalBytes);
                    f_fillBufView(inBufView, inBuf, recvRanges);

                    {
                        receiver = nodeGrid.getNodeNeighborIndex(dir);
                        sender = nodeGrid.getNodeNeighborIndex(oppositeDir);
                        auto f_pack =
                            [](OutBufferView& buf, Cell& _reals, int extradata, const Real3D& shift)
                        {
                            ParticleList& reals = _reals.particles;
                            for (auto& real : reals)
                            {
                                buf.write(real, extradata, shift);
                            }
                        };
                        const size_t size = commCells[dir].reals.size();
                        utils::parallelForLoop(
                            0, size,
                            [this, &f_pack, &extradata, &shift, dir](size_t i)
                            { f_pack(outBufView[i], *commCells[dir].reals[i], extradata, shift); });
                    }

                    timeExcGhosts_InterNode_pack += wallTimer.getElapsedTime() - time;
                }

                {
                    const real time = wallTimer.getElapsedTime();

                    // exchange particles, odd-even rule
                    if (nodeGrid.getNodePosition(coord) % 2 == 0)
                    {
                        outBuf.send(receiver, DD_COMM_TAG);
                        inBuf.recv(sender, DD_COMM_TAG);
                    }
                    else
                    {
                        inBuf.recv(sender, DD_COMM_TAG);
                        outBuf.send(receiver, DD_COMM_TAG);
                    }

                    timeExcGhosts_InterNode_comm += wallTimer.getElapsedTime() - time;
                }

                {
                    const real time = wallTimer.getElapsedTime();

                    {
                        auto f_unpack = [](Cell& _ghosts, InBufferView& buf, int extradata)
                        {
                            ParticleList& ghosts = _ghosts.particles;
                            for (auto& ghost : ghosts)
                            {
                                buf.read(ghost, extradata);
                                /// NOTE: multithreaded particle map not implemented
                                // if (extradata & DATA_PROPERTIES) {
                                //   updateInLocalParticles(&(*dst), true);
                                // }
                                ghost.ghost() = true;
                            }
                        };
                        const size_t size = commCells[dir].ghosts.size();
                        utils::parallelForLoop(
                            0, size,
                            [this, &f_unpack, &extradata, dir](size_t i)
                            { f_unpack(*commCells[dir].ghosts[i], inBufView[i], extradata); });
                    }

                    timeExcGhosts_InterNode_unpack += wallTimer.getElapsedTime() - time;
                }
            }
        }
    }
    LOG4ESPP_DEBUG(logger, "ghost communication finished");
}

template <bool ALIGNED>
void FullDomainDecomposition::exchangeGhosts_impl()
{
    if (exchangeGhosts_impl_john)
    {
        exchangeGhosts_full_impl<ALIGNED>();
    }
    else
    {
        exchangeGhosts_six_impl<ALIGNED>();
    }
}

template <bool ALIGNED>
void FullDomainDecomposition::exchangeGhosts_full_impl()
{
    // Is called while Decompose HPX
    auto const& comm = *(getSystem()->comm);

    constexpr int extradata = DATA_PROPERTIES;

    std::array<hpx::future<void>, 26> ghost_exchanges;
    std::vector<Cell*> ghosts;

    /// Checks Intra that there is no overlapping between ghosts and reals
    if (true)
    {
        std::set<Cell*> cells;
        for (const auto& neighbour : John::NeighbourEnumIterable)
        {
            for (size_t i = 0; i < fullCommCells[neighbour].ghosts.size(); ++i)
            {
                Cell*& cell = fullCommCells[neighbour].ghosts[i];
                if (cells.count(cell) == 0)
                {
                    cells.insert(cell);
                }
                else
                {
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "CHECKING",
                        spdlog::fmt_lib::format("Cell num {} in neighbour {} already exists", i,
                                                neighbour));
                }
            }
            std::set<Cell*> same_as_dest;
            for (size_t i = 0; i < fullCommCells[neighbour].reals.size(); ++i)
            {
                Cell*& cell_r = fullCommCells[neighbour].reals[i];
                Cell*& cell_g = fullCommCells[neighbour].ghosts[i];

                if (same_as_dest.count(cell_r) == 0 && same_as_dest.count(cell_g) == 0)
                {
                    same_as_dest.insert(cell_r);
                    same_as_dest.insert(cell_g);
                }
                else
                {
                    HPX4ESPP_THROW_EXCEPTION(
                        hpx::assertion_failure, "CHECKING",
                        spdlog::fmt_lib::format("OVERLAPPING !! {} in neighbour {} already exists",
                                                i, neighbour));
                }
            }
        }
    }

    for (const auto& neighbour : John::NeighbourEnumIterable)
    {
        // LOG4ESPP_DEBUG(logger, "local communication");
        //                std::cout<<"nodeGrid.getGridSize(coord) == 1" << std::endl;
        Real3D shift(0, 0, 0);
        const auto direction = enum_to_direction(opposite_neighbour(neighbour));
        const bool doPeriodic = full_grid.is_periodic_neighbour(neighbour);

        if (doPeriodic)
        {
            for (int coord = 0; coord < 3; ++coord)
            {
                const real curCoordBoxL = getSystem()->bc->getBoxL()[coord];
                if (direction[coord] != 0)
                {
                    shift[coord] = direction[coord] * curCoordBoxL;
                }
            }
        }

        SPDLOG_TRACE("Shifting by {},{},{} neighbour {} ", shift[0], shift[1], shift[2],
                     John::relation_to_string(neighbour));

        // copy operation, we have to receive as many cells as we send
        if (fullCommCells[neighbour].ghosts.size() != fullCommCells[neighbour].reals.size())
        {
            throw std::runtime_error(
                "FullDomainDecomposition::doGhostCommunication: send/recv cell structure "
                "mismatch during local copy");
        }

        if (full_grid.neighbour_is_self(neighbour))
        {
            ghost_exchanges[neighbour] = hpx::async(
                [this, neighbour, shift]()
                {
                    const real time = wallTimer.getElapsedTime();
                    SPDLOG_TRACE("Doing neighbour ghost_impl {}", neighbour);
                    exchangeGhosts_intra(neighbour, shift, extradata);
                    timeExcGhosts_IntraNode += wallTimer.getElapsedTime() - time;
                });
            continue;  // Don't do inter
        }
        auto send_future =
            hpx::async([this, neighbour, shift]()
                       { exchangeGhosts_send_routine<ALIGNED>(neighbour, shift, extradata); });
        ghost_exchanges[neighbour] =
            hpx::async([this, neighbour, shift]()
                       { exchangeGhosts_receive_routine<ALIGNED>(neighbour, extradata); });
    }

    hpx::wait_all(ghost_exchanges);
    SPDLOG_TRACE("Done INTRA");
    // LOG4ESPP_DEBUG(logger, "do ghost communication " << (sizesFirst ? "with sizes " : "")
    //       << (realToGhosts ? "reals to ghosts " : "ghosts to reals ") << extradata);

    /* direction loop: x, y, z.
      Here we could in principle build in a one sided ghost
      communication, simply by taking the lr loop only over one
      value. */
}
template void FullDomainDecomposition::exchangeGhosts_impl<true>();
template void FullDomainDecomposition::exchangeGhosts_impl<false>();

template void FullDomainDecomposition::exchangeGhosts_six_impl<true>();
template void FullDomainDecomposition::exchangeGhosts_six_impl<false>();

template void FullDomainDecomposition::exchangeGhosts_full_impl<true>();
template void FullDomainDecomposition::exchangeGhosts_full_impl<false>();

}  // namespace espressopp::hpx4espp::storage
