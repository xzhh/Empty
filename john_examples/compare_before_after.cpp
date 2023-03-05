//
// Created by jn98zk on 7/29/22.
//
//#include "json_lib.hpp"
#include <fstream>
#include <iostream>
using json = nlohmann::json;
#include <string>

int main(){

    std::ifstream john_file("/home/jn98zk/CLionProjects/espressopp/cmake-build-debug/john_examples/BeforeBuffSend iteration 0 Real2Ghost false rank 1.json");
    json john_impl = json::parse(john_file);
    std::ifstream orig_file("/home/jn98zk/CLionProjects/bachelor_copy/cmake-build-debug/john_examples/BeforeBuffSend iteration 0 Real2Ghost false rank 1.json");
    json orig_impl = json::parse(orig_file);


    auto diff12 = json::diff(john_impl,orig_impl);
    auto diff21 = json::diff(orig_impl,john_impl);

    json my_diff;
    
    auto max_diff = std::numeric_limits<double>::min();

    for (size_t i = 0; i < diff12.size(); ++i) {
        auto val1 = diff12[i].at("value").get<double>();
        auto val2 = diff21[i].at("value").get<double>();
        auto where = diff12[i].at("path").get<std::string>();
        const auto the_diff = val1 - val2;
        my_diff[i] = {where,the_diff};
        max_diff = std::max(std::abs(the_diff), max_diff);
    }


    std::ofstream abs_diff_file("/home/jn98zk/CLionProjects/espressopp/john_examples/before_abs_diff.json");
    abs_diff_file << std::setw(4) << my_diff << std::endl;
    std::ofstream diff_patch("/home/jn98zk/CLionProjects/espressopp/john_examples/before_abs_patch.json");
    diff_patch << std::setw(4) << diff21 << std::endl;

    std::cout<<"max diff " << max_diff << " size "<<diff12.size();
}
