## Download fmt::fmt-header-only release 8.1.1
#include(FetchContent)
#
#FetchContent_Declare(fmt::fmt-header-only URL https://github.com/fmt::fmt-header-onlylib/fmt::fmt-header-only/releases/download/8.1.1/fmt::fmt-header-only-8.1.1.zip EXCLUDE_FROM_ALL)
## Make fmt::fmt-header-only available as link target
#FetchContent_MakeAvailable(fmt::fmt-header-only)

find_package(fmt REQUIRED)

add_executable(hpx_new_main decomp.hpp hpx_new_main.cpp)
target_link_libraries(hpx_new_main fmt::fmt-header-only)


target_link_libraries(hpx_new_main _espressopp HPX::hpx _hpx4espp_component HPX::wrap_main Boost::mpi)
set_property(TARGET hpx_new_main PROPERTY CXX_STANDARD 17)

add_executable(node_grid_calcs decomp.hpp node_grid_calcs.cpp)


target_link_libraries(node_grid_calcs _espressopp HPX::hpx _hpx4espp_component HPX::wrap_main Boost::mpi fmt::fmt-header-only)
set_property(TARGET node_grid_calcs PROPERTY CXX_STANDARD 17)


add_executable(more_main more_main.cpp)

execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
target_compile_definitions(more_main PRIVATE
        "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")


target_link_libraries(more_main _espressopp HPX::hpx _hpx4espp_component HPX::wrap_main Boost::mpi fmt::fmt-header-only)
set_property(TARGET more_main PROPERTY CXX_STANDARD 17)
target_compile_options(more_main PUBLIC -fPIC)


add_executable(cell_grid_calc cell_grid_calc.cpp)
target_link_libraries(cell_grid_calc _espressopp HPX::hpx _hpx4espp_component HPX::wrap_main Boost::mpi fmt::fmt-header-only)

#add_executable(read_json read_json.cpp)
#add_executable(read_send_buff read_send_buff.cpp)
add_executable(compare_before_after compare_before_after.cpp)
add_executable(compare_stuff.cpp compare_stuff.cpp)
add_executable(bin_diff.cpp bin_diff.cpp)
add_executable(hpx_ping_pong hpx_ping_pong.cpp)
target_link_libraries(hpx_ping_pong HPX::hpx HPX::wrap_main)
target_link_libraries(bin_diff.cpp fmt::fmt-header-only)
