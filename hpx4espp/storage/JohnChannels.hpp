/*
  Copyright (C) 2021-2022
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

#ifndef HPX4ESPP_STORAGE_JohnCHANNELS_HPP
#define HPX4ESPP_STORAGE_JohnCHANNELS_HPP

#include "hpx4espp/include/hpx_version.hpp"
#include <spdlog/spdlog.h>

#if (HPX4ESPP_HPX_VERSION_FULL >= 10500)

#include <hpx/channel.hpp>
#include <hpx/lcos_local/channel.hpp>

#else
#include <hpx/lcos/channel.hpp>
#include <hpx/local_lcos/channel.hpp>
#endif

#include <hpx/include/lcos.hpp>

#include "vec/include/simdconfig.hpp"
#include "storage/NodeGrid.hpp"
#include "python.hpp"

#define CHIDX(COORD, LR) ((LR) + 3*(COORD))

namespace espressopp::hpx4espp::storage {


    template<typename T,int TOPIC,typename ALLOCATOR=std::allocator<T>>
    class JohnChannels {
        using ChannelType = hpx::distributed::channel<T>;
        using IndexType = std::tuple<int, int, int>;
    public:

        [[nodiscard]] constexpr static std::size_t
        linear_index(const std::size_t coord, const std::size_t lr) noexcept {
            return lr + coord * 3;
        }


        JohnChannels(espressopp::storage::NodeGrid const &nodeGrid,
                     std::array<size_t, 3> const &subNodeGrid) {
            const int rank = mpiWorld->rank();
            size_t totSubNodes = 0;
            for (int coord = 0; coord < 3; ++coord) {
                /// correctly skip self-interactions
                if (nodeGrid.getGridSize(coord) == 1) {
                    numSubNodes[coord] = 0;
                    shiftSubNodes[coord] = totSubNodes;
                    totSubNodes += numSubNodes[coord];
                } else {
                    auto subCommGrid = subNodeGrid;
                    subCommGrid[coord] = 1;
                    numSubNodes[coord] = subNodeGrid[0] * subNodeGrid[1] * subNodeGrid[2];
                    shiftSubNodes[coord] = totSubNodes;
                    totSubNodes += numSubNodes[coord];

                    const int dir = 2 * coord + 0;
                    const int oppDir = 2 * coord + 1;
                    const int selfid = rank;
                    int lr;

                    /// senders
                    {
                        lr = 0;
                        const int recver0 = nodeGrid.getNodeNeighborIndex(dir);
                        sendIdxs.emplace_back(lr, selfid, recver0);

                        lr = 1;
                        const int recver1 = nodeGrid.getNodeNeighborIndex(oppDir);
                        sendIdxs.emplace_back(lr, selfid, recver1);
                    }


                    /// receivers
                    {
                        lr = 0;
                        const int sender0 = nodeGrid.getNodeNeighborIndex(oppDir);
                        recvIdxs.emplace_back(lr, sender0, selfid);

                        lr = 1;
                        const int sender1 = nodeGrid.getNodeNeighborIndex(dir);
                        recvIdxs.emplace_back(lr, sender1, selfid);
                    }
                }
            }

            /// setup receiver channel
            {
                for (auto const &c: recvIdxs) {
                    std::string name = getChannelName(c);
                    recvChannels.emplace_back(hpx::find_here());
                    hpx::register_with_basename(name, recvChannels.back(), 0);
                    spdlog::info("Channel receive {}",name);
                }
            }

            /// setup sender channel
            {
                for (auto const &c: sendIdxs) {
                    std::string name = getChannelName(c);
                    sendChannels.push_back(hpx::find_from_basename<ChannelType>(name, 0));
                    spdlog::info("Channel send {}",name);

                }
            }

        }

        [[nodiscard]] ChannelType &sendChannel(size_t coord, size_t lr) noexcept{
            return sendChannels[linear_index(coord, lr)];
        }

        [[nodiscard]] ChannelType &recvChannel(size_t coord, size_t lr) noexcept {
            return recvChannels[linear_index(coord, lr)];
        }

        IndexType  sendIdx(size_t coord, size_t lr) const {
            return sendIdxs[linear_index(coord, lr)];
        }

        IndexType recvIdx(size_t coord, size_t lr) const {
            return recvIdxs[linear_index(coord, lr)];
        }
        python::object getChannelIndices() const {
            throw std::runtime_error("JohnChannelGetIndicesPy not implemented");
            auto l = python::list();
            return l;
        }

    protected:
        std::vector<ChannelType> sendChannels;
        std::vector<ChannelType> recvChannels;

        std::vector<T> sendBufs;

        std::vector<IndexType> sendIdxs;
        std::vector<IndexType> recvIdxs;

        static std::string getChannelName(int lr, int src, int dst) {
            return hpx::util::format("/hpx4espp_channel/{}/{}/{}/{}", TOPIC, lr, src, dst);
        }

        static std::string getChannelName(IndexType const &p) {
            return std::move(getChannelName(std::get<0>(p), std::get<1>(p), std::get<2>(p)));
        }

        std::array<size_t, 3> numSubNodes;
        std::array<size_t, 3> shiftSubNodes;
    };

}  // namespace espressopp



#endif  // HPX4ESPP_STORAGE_CHANNELS_HPP
