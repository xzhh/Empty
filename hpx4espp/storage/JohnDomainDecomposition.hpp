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

#ifndef HPX4ESPP_STORAGE_FULLDOMAINDECOMPOSITION_HPP
#define HPX4ESPP_STORAGE_FULLDOMAINDECOMPOSITION_HPP

#include <hpx/iostream.hpp>
//#include "hpx4espp/hpx_serialization.hpp"
#include <iostream>
#include "log4espp.hpp"
#include "storage/DomainDecomposition.hpp"
#include "StorageHPX.hpp"
#include "JohnChannels.hpp"
#include "JohnFullChannels.hpp"
#include "esutil/Timer.hpp"
#include "hpx4espp/BufferView.hpp"

// If false, revert to original domain decomp
// #define HPX4ESPP_OPTIMIZE_COMM (1)

/// Forward declaration



namespace espressopp::integrator
{
class MDIntegrator;
}  // namespace espressopp::integrator

namespace espressopp::hpx4espp::storage
{
struct FullVirtualDomainDecomposition
{
    espressopp::storage::NodeGrid nodeGrid;       /// index wrt entire system domain
    espressopp::storage::NodeGrid nodeGridLocal;  /// index wrt locality subdomain

    John::FullNeighbourNodeGrid fullNodeGridLocal;

    espressopp::CellGrid cellGrid;

    struct CellInfo
    {
        enum CellType
        {
            CELL_REAL,
            CELL_GHOST_EXTERNAL,
            CELL_GHOST_INTERNAL
        };
        CellType cellType;
        size_t subNode;
        size_t subNodeCell;  // Unused, so purge it
    };

    std::vector<CellInfo> cellGridInfo;

    struct CommCellIdx
    {
        std::vector<size_t> reals;
        std::vector<size_t> ghosts;
    };

    std::array<CommCellIdx, 6> commCells;
    /// Buffer to SubCellgrid  mapping
    std::array<CommCellIdx, 6> commCellsOwned;

    std::array<CommCellIdx, 26> fullCommCells;
    /// Buffer to SubCellgrid  mapping
    std::array<CommCellIdx, 26> fullCommCellsOwned;

    /// Not needed
    //    std::array<std::vector<size_t>, 6> numCommSubCells;
    //    std::array<std::vector<std::pair<size_t, size_t>>, 6> commSubCellRanges;
    //    std::array<std::vector<size_t>, 6> subCommRangeReal;
    //    std::array<std::vector<size_t>, 6> subCommRangeGhost;

    // Don't Touch
    CellVec vGhostCells;
    CellPtrVec vLocalCells;
    CellPtrVec vRealCells;
};

typedef FullVirtualDomainDecomposition FVDD;

class FullDomainDecomposition : public espressopp::storage::DomainDecomposition, public StorageHPX
{
public:
    typedef espressopp::storage::DomainDecomposition baseClass;
    bool HPX4ESPP_OPTIMIZE_COMM;

    FullDomainDecomposition(shared_ptr<System> system,
                            const Int3D& nodeGrid,
                            const Int3D& cellGrid,
                            int halfCellInt,
                            const Int3D& subCellGrid,
                            int numCommSubs,
                            bool commAsync,
                            bool excgAligned,
                            bool commUseChannels,
                            bool decompUseParFor,
                            bool _HPX4ESPP_OPTIMIZE_COMM);

    ~FullDomainDecomposition();



    void initChannels();

    /** Copy particles to packed form. To be called at the start of integrator.run */
    void loadCells();

    /** Copy particles back from packed form. To be called at the end of integrator.run */
    void unloadCells();

    void prepareGhostBuffers();

    void updateGhostsBlocking();

    void collectGhostForcesBlocking();

    void connect();

    void disconnect();

    void resetVirtualStorage();

    void exchangeGhosts_intra(John::NeighbourRelation neighbour,const Real3D shift,const int extradata);
    template <bool ALIGNED>
    void exchangeGhosts_send_routine(John::NeighbourRelation neighbour_to_send_to,const Real3D shift,const int extradata);


    void decomposeHPX();

    static void registerPython();

protected:
    std::array<size_t, 3> subCellGrid;

    std::array<size_t, 3> subNodeGrid;

    bool exchangeGhosts_impl_john{false};
    bool ghostCommunication_impl_john{true};
    bool decomposeRealParticlesHPXParFor_john{false};
    size_t numSubNodes;

    size_t numSubCells;

    std::vector<std::vector<size_t>> resortWaves;

    std::vector<FVDD> vdd;

    // Flattened sub-nodegrid that is communicating with other nodes
    std::array<std::vector<size_t>, 6> commNodesReal;
    std::array<std::vector<size_t>, 6> commNodesGhost;
    std::array<std::vector<size_t>, 6> nodeRangeReal;
    std::array<std::vector<size_t>, 6> nodeRangeGhost;
    std::array<std::vector<std::tuple<size_t, size_t, bool>>, 6> subNodePairsIntra;
    std::array<std::vector<std::tuple<size_t, size_t, bool>>, 6> subNodePairsIntraPeriodic;

    // Full 26 Neighbours
    std::array<std::vector<size_t>, 26> fullCommNodesReal;
    std::array<std::vector<size_t>, 26> fullCommNodesGhost;
    std::array<std::vector<size_t>, 26> fullNodeRangeReal;
    std::array<std::vector<size_t>, 26> fullNodeRangeGhost;
    std::array<std::vector<std::tuple<size_t, size_t, bool>>, 26> fullSubNodePairsIntra;
    std::array<std::vector<std::tuple<size_t, size_t, bool>>, 26> fullSubNodePairsIntraPeriodic;

    size_t maxReal, maxGhost;
    size_t fullMaxReal{}, fullMaxGhost{};
    vec::AlignedVector<real> buffReal, buffGhost;

    // We don't have buffer for async as they are created dynamically
//    vec::AlignedVector<real> fullBuffReal, fullBuffGhost;

    /// range of cells for each node
    std::array<std::vector<size_t>, 6> nodeCommCellRange;
    std::array<std::vector<size_t>, 26> fullNodeCommCellRange;
    ExposedParticleList decompSendBuf, decompRecvBuf;

    template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
    void ghostCommunication_impl();
    template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
    void ghostCommunication_full_impl();
    template <bool SIZES_FIRST, bool REAL_TO_GHOSTS, int EXTRA_DATA>
    void ghostCommunication_six_impl();
    bool commAsync;



    typedef espressopp::vec::AlignedVector<char> AlignedVectorChar;

    std::vector<std::tuple<int, int>> nsubComm;
    std::array<std::vector<AlignedVectorChar>, 6> sendBufReal;
    std::array<std::vector<AlignedVectorChar>, 6> sendBufGhost;
    std::array<std::vector<size_t>, 6> sendBufSizeReal;
    std::array<std::vector<size_t>, 6> sendBufSizeGhost;

    enum PackedData
    {
        PACKED_POSITIONS = 0,
        PACKED_FORCES = 1
    };
    enum DataMode
    {
        DATA_INSERT = 0,
        DATA_ADD = 1
    };
    enum AddShift
    {
        NO_SHIFT = 0,
        ADD_SHIFT = 1
    };
    static const espressopp::Real3D SHIFT_ZERO;



    const size_t numCommSubs;

    void ghostCommunication_impl_full_intra_node(Real3D shift,
                                                 bool doPeriodic,
                                                 John::NeighbourRelation neighbour,
                                                 bool real_to_ghosts);

public:
    python::object getChannelIndices() const;

protected:
    bool commUseChannels;
    bool channelsInit = false;
    JohnChannels<std::vector<int>, 3> exchange_ghost_channels;
    //    JohnChannels<AlignedRealVecDD,4> ghostCommunication_impl_channels;

    espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::longint>, 10>
        ch_decompose_real_sizes;

    espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::Particle>, 11>
        ch_decompose_real_particles;

    espressopp::hpx4espp::channels::JohnFullChannels<vec::AlignedVector<real>, 20>
        ch_ghostCommunication_impl;

    espressopp::hpx4espp::channels::JohnFullChannels<std::vector<espressopp::longint>, 30>
        ch_exchangeGhosts_impl_sizes;

    espressopp::hpx4espp::channels::JohnFullChannels<hpx::serialization::serialize_buffer<char>, 31>
        ch_exchangeGhosts_impl_particles;


    template <AddShift DO_SHIFT>
    void fullCopyRealsToGhostsIntra(
        John::NeighbourRelation neighbour, size_t ir, size_t ig, size_t is, Real3D const& shift);

    void fullAddGhostForcesToRealsIntra(John::NeighbourRelation neighbour,
                                        size_t ir,
                                        size_t ig,
                                        size_t is);




    template <PackedData PACKED_DATA, AddShift DO_SHIFT>
    void packCells_full(vec::AlignedVector<real>& sendBuf,
                   bool commReal,
                   John::NeighbourRelation neighbour,
                   size_t idxCommNode,
                   Real3D const& shift);
    template <AddShift DO_SHIFT>
    void copyRealsToGhostsIntra(size_t dir, size_t ir, size_t ig, size_t is, Real3D const& shift);


    void addGhostForcesToRealsIntra(size_t dir, size_t ir, size_t ig, size_t is);


    template <PackedData PACKED_DATA, DataMode DATA_MODE>
    void unpackCells_full(vec::AlignedVector<real> const& recvBuf,
                     bool commReal,
                      John::NeighbourRelation neighbour,
                     size_t idxCommNode);

    static constexpr size_t vecModeFactor = 3;

    /** Members extending base DomainDecomposition class */
protected:
    BufferFixed inBuf, outBuf;
    std::vector<BufferFixed> outBuffers;
    std::vector<hpx::future<void>> outBuffers_futs;
    std::vector<InBufferView> inBufView;
    std::vector<OutBufferView> outBufView;
    // std::vector<BufferView> inBufView, outBufView;

    std::size_t fullPreallocReal;
    std::size_t fullPreallocGhost;

    template <bool ALIGNED>
    void exchangeGhosts_impl();

    template <bool ALIGNED>
    void exchangeGhosts_full_impl();

    template <bool ALIGNED>
    void exchangeGhosts_six_impl();

    bool excgAligned = false;


    bool decompUseParFor;

    template <bool REAL_TO_GHOSTS>
    hpx::future<void> ghostCommunication_impl_send_routine(
        Real3D shift, John::NeighbourRelation neighbour_to_send_to, std::size_t current_iteration);

    template <bool ALIGNED>
    void exchangeGhosts_receive_routine(John::NeighbourRelation neighbour,
                                        int extradata);

    template <bool REAL_TO_GHOSTS>
    void ghostCommunication_impl_receive_routine(John::NeighbourRelation filling_relation,
                                                 std::size_t current_iteration);

    void decomposeRealParticlesHPXCellTask_MultiNode_NonPeriodic();

    void decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic();



    void decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_full();

    void decomposeRealParticlesHPXParFor_MultiNode_NonPeriodic_six();

    void decomposeRealParticles_full_intra(size_t pair_index, John::NeighbourRelation neighbour);

    /** Utility functions for connecting to integrator to do stepwise offload */
public:
    void connectOffload(boost::shared_ptr<espressopp::integrator::MDIntegrator> mdintegrator);

    void disconnectOffload();

    bool connectedOffload() const { return offload; }

protected:
    void resetParticles();

    void resetCells();

    void befCalcForces();

    void updateForces();

    void simple_decomposeRealParticles_send_routine(
        John::NeighbourRelation send_to_neighbour);

    std::vector<Particle> decomposeRealParticles_pack_node(
        John::NeighbourRelation send_to_neighbour,
        const std::vector<longint>& sendRanges,
        const std::vector<longint>& sendSizes);

    void decomposeRealParticles_send_routine(John::NeighbourRelation send_to_neighbour);

    void decomposeRealParticles_unpack_node(John::NeighbourRelation filling_neighbour,
                                            const std::vector<longint>& recvRanges,
                                            const std::vector<longint>& recvSizes,
                                            std::vector<Particle>& received_particles);

    void decomposeRealParticles_receive_routine(John::NeighbourRelation filling_relation);

    void simple_decomposeRealParticles_receive_routine(
        John::NeighbourRelation receive_neighbour);

    template <PackedData PACKED_DATA, AddShift DO_SHIFT>
    void packCells(vec::AlignedVector<real>& sendBuf,
                   bool commReal,
                   size_t dir,
                   size_t idxCommNode,
                   Real3D const& shift);

    template <PackedData PACKED_DATA, DataMode DATA_MODE>
    void unpackCells(vec::AlignedVector<real> const& recvBuf,
                     bool commReal,
                     size_t dir,
                     size_t idxCommNode);

    // signals that connect to integrator
    boost::signals2::connection sigBefCalcForces;
    boost::signals2::connection sigUpdateForces;

    // signals that connect to system
    boost::signals2::connection sigResetParticles;
    boost::signals2::connection sigResetCells;

    bool offload = false;

    /** Timer-related  */
public:
    void loadTimers(real t[2]);

    void resetTimers();

protected:
    real timeUpdateGhostsBlocking_comm;
    real timeCollectGhostForcesBlocking_comm;
    real timeParticlesCopyFrom;
    real timeParticlesUpdateTo;
    real timeDecomposeInvGhosts;
    real timeDecomposeReal;
    real timeDecomposeExcGhosts;
    real timeDecomposeSignal;
    real timeDecomposeRealResort;
    real timeDecomposeRealComm;
    real timeLoadLoop;
    real timeLoadPrepGhost;
    real timeLoadSignal;
    real timeUnloadLoop;
    real timeUnloadSignal;
    real timeUpdateGhosts_InterNode_pack;
    real timeUpdateGhosts_InterNode_comm;
    real timeUpdateGhosts_InterNode_unpack;
    real timeUpdateGhosts_IntraNode;
    real timeExcGhosts_InterNode_sizes;
    real timeExcGhosts_InterNode_pack;
    real timeExcGhosts_InterNode_comm;
    real timeExcGhosts_InterNode_unpack;
    real timeExcGhosts_IntraNode;

    espressopp::esutil::WallTimer wallTimer;

    /** Provides interace to testing functions */
public:
    std::vector<FVDD> getVDD() const { return vdd; }
    std::vector<FVDD::CommCellIdx> getCommCellsAsIdx() const;

    static LOG4ESPP_DECL_LOGGER(logger);
    template <typename VECTOR>
    static std::vector<longint> prepare_ranges_from_sizes(VECTOR&& sizes) noexcept
    {
        longint total_count = 0;
        std::vector<longint> ranges;
        ranges.reserve(sizes.size() + 1);
        for (longint sub_count : sizes)
        {
            ranges.push_back(total_count);
            total_count += sub_count;
        }
        ranges.push_back(total_count);
        return ranges;
    }
};

}  // namespace espressopp::hpx4espp::storage

#endif  // HPX4ESPP_STORAGE_FULLDOMAINDECOMPOSITION_HPP
