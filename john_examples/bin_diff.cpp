//
// Created by jn98zk on 8/2/22.
//

//#include "json_lib.hpp"
#include <fstream>
#include <iostream>
#include <fmt/format.h>
using json = nlohmann::json;
#include <string>
std::size_t split_string(const char delimiter,const std::string& input,std::size_t break_after){
    std::size_t seen_count=0;

    for (size_t i = 0; i < input.size(); ++i)
    {
        seen_count += input[i]==delimiter;
        if(seen_count==break_after){
            return i;
        }
    }
    return 0;
}
int main(int argc, char* argv[]){


    std::ifstream abs_diff_file("/home/jn98zk/CLionProjects/espressopp/john_examples/before_update_abs_send.json");
    std::ifstream orig_file("/home/jn98zk/CLionProjects/espressopp/cmake-build-debug/john_examples/after_collect_forces iteration 9 rank 0.json");
    json orig = json::parse(orig_file);
    json diff = json::parse(abs_diff_file);

    json bin_counter;


    for (const auto& item : diff)
    {

        auto path = item[0].get<std::string>();
        auto delimiter_pos= split_string('/',path,3);
        auto full_cell = path.substr(0,delimiter_pos);
        json::json_pointer ptr(full_cell);

        auto orig_d = orig.at(ptr);
        auto grid_x  = orig_d[2][1].get<int>();
        auto grid_y  = orig_d[2][2].get<int>();
        auto grid_z = orig_d[2][3].get<int>();
        std::string key = fmt::format("{},{},{}", grid_x, grid_y, grid_z);
        if(bin_counter.contains(key)){
            bin_counter[key] = bin_counter[key].get<int>()+1;

        }else{
            bin_counter[key]  = 1;
        }
//        break;
    }


    std::ofstream bin_out("/home/jn98zk/CLionProjects/espressopp/john_examples/binnner.json");
    bin_out << bin_counter;

}
