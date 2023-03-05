//
// Created by jn98zk on 8/7/22.
//

#ifndef ESPRESSOPP_GLOBAL_DATA_HPP
#define ESPRESSOPP_GLOBAL_DATA_HPP
//#include "thead_safe_json.hpp"

//extern HPXThreadsafeJson sending_json;

extern std::size_t iteration_number;

extern bool exchangeGhosts_impl_john_opt;
extern bool ghostCommunication_impl_john_opt;
extern bool decomposeRealParticlesHPXParFor_john_opt;
extern bool enable_resort;

#endif  // ESPRESSOPP_GLOBAL_DATA_HPP
