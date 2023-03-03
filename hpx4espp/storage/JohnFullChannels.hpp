//
// Created by jn98zk on 7/17/22.
//

#ifndef ESPRESSOPP_JOHNFULLCHANNELS_HPP
#define ESPRESSOPP_JOHNFULLCHANNELS_HPP
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



#include <spdlog/spdlog.h>

#include <hpx/channel.hpp>
#include <hpx/lcos_local/channel.hpp>
#include <hpx/include/lcos.hpp>
#include "storage/FullNeighbourNodeGrid.hpp"

namespace espressopp::hpx4espp::channels
{

template <typename T, int TOPIC, typename ALLOCATOR = std::allocator<T>>
class JohnFullChannels
{
    using ChannelType = hpx::distributed::channel<T>;

public:

    JohnFullChannels() = default;

    JohnFullChannels(John::FullNeighbourNodeGrid const &nodeGrid,
                     std::size_t rank)
    {
        for (const auto &neighbour : John::NeighbourEnumIterable)
        {
            // Skip setting up channels for yourself
            const auto neighbour_rank = nodeGrid.getNodeNeighborIndex(neighbour);
            // Skip self comm setup
            if (neighbour_rank == rank)
            {
                continue;
            }
            const auto opposite_neighbour = John::periodic_neighbour[neighbour];

            // Setup Sender in that direction
            auto const send_channel_name = getChannelName(neighbour);
            auto const recv_channel_name = getChannelName(opposite_neighbour);

            sendChannels[neighbour] = ChannelType(hpx::find_here());
            hpx::register_with_basename(send_channel_name, sendChannels[neighbour], rank);

            recvChannels[opposite_neighbour] =
                hpx::find_from_basename<ChannelType>(recv_channel_name, neighbour_rank);
        }
    }


    ChannelType& get_send_channel(John::NeighbourRelation neighbour){
        return sendChannels[neighbour];
    }

    ChannelType& get_receive_channel(John::NeighbourRelation neighbour){
        return recvChannels[neighbour];
    }


    protected:
        std::array<ChannelType, 26> sendChannels;
        std::array<ChannelType, 26> recvChannels;

        static std::string getChannelName(John::NeighbourRelation relation)
        {
            return hpx::util::format("/hpx4espp_channel/{}/{}", TOPIC, relation);
        }
    };

}  // namespace espressopp


#endif  // ESPRESSOPP_JOHNFULLCHANNELS_HPP
