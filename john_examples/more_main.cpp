//
// Created by jn98zk on 5/11/22.
//

#include "decomp.hpp"
#include <hpx/hpx_init.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "spdlog/sinks/basic_file_sink.h"
#include <spdlog/fmt/chrono.h>

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif

bool exchangeGhosts_impl_john_opt = false;
bool ghostCommunication_impl_john_opt = true;
bool decomposeRealParticlesHPXParFor_john_opt = false;
bool enable_resort = true;

//#include "json_lib.hpp"
#include <fstream>
//using json = nlohmann::json;

std::stringstream log_timer(const espressopp::real *timers)
{
    std::stringstream stream;
    stream << " Run = " << timers[0] << std::endl;
    stream << " Pair = " << timers[1] << std::endl;
    stream << " FENE = " << timers[2] << std::endl;
    stream << " Angle = " << timers[3] << std::endl;
    stream << " Comm1 = " << timers[4] << std::endl;
    stream << " Comm2 = " << timers[5] << std::endl;
    stream << " Int1 = " << timers[6] << std::endl;
    stream << " Int2 = " << timers[7] << std::endl;
    stream << " Resort = " << timers[8] << std::endl;
    stream << " Other = " << timers[9] << std::endl;
    return stream;
}

int hpx_main(hpx::program_options::variables_map &vm)
{
    initMPIEnv();
    std::string run_id;
    run_id = vm["id"].as<std::string>();
    auto isteps = vm["isteps"].as<std::size_t>();
    exchangeGhosts_impl_john_opt = vm["exchangeGhosts_impl_john_opt"].as<bool>();
    ghostCommunication_impl_john_opt = vm["ghostCommunication_impl_john_opt"].as<bool>();
    decomposeRealParticlesHPXParFor_john_opt =
        vm["decomposeRealParticlesHPXParFor_john_opt"].as<bool>();

    enable_resort = vm["resort"].as<bool>();
    auto rc = 2.5;
    auto skin = 0.3;
    auto timestep = 0.005;
    auto rho = 0.8442;
    auto numSubs = vm["numSubs"].as<std::size_t>();
    auto temperature =  vm["temp"].as<int>();
    const int num_threads = (int)hpx::get_num_worker_threads();
    const auto num_ranks = hpx::get_num_localities(hpx::launch::sync);
    const auto my_rank = hpx::get_locality_id();
    const auto file_path =
        spdlog::fmt_lib::format("logs/git hash {}/{} num_ranks {} threads {} my_rank {}  isteps {}",
                                GIT_COMMIT_HASH, run_id, num_ranks, num_threads, my_rank, isteps);

    const auto timing_path =
        spdlog::fmt_lib::format("logs/git hash {}/timings/{} num_ranks {} threads {} my_rank {}  isteps {}",
                                GIT_COMMIT_HASH, run_id, num_ranks, num_threads, my_rank, isteps);
    const auto main_pattern = "[%H:%M:%S:%f][%^%l%$] %#:%! %n %v";
    if (my_rank==0){
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_path, true);
        //    auto timing_file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(timing_path, true);
        //    auto timing_logger = std::make_shared<spdlog::logger>(spdlog::logger("timings",timing_file_sink));
        //    timing_logger->set_pattern("%T:%F %v");
        file_sink->set_level(spdlog::level::trace);
        //    spdlog::register_logger(timing_logger);
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);

        auto default_logger = my_rank==0?
                                           std::make_shared<spdlog::logger>(spdlog::logger("", {file_sink, console_sink})):std::make_shared<spdlog::logger>(spdlog::logger(""));
        spdlog::set_default_logger(default_logger);
        default_logger->set_level(spdlog::level::trace);
        spdlog::logger quiet_logger("params", {file_sink, console_sink});

        console_sink->set_pattern(spdlog::fmt_lib::format("[Rank {}]{}", my_rank, main_pattern));
        file_sink->set_pattern(main_pattern);
    }


    spdlog::info("\nisteps {}\nrc {}\nskin {}\ntimestep {}\nrho {}\nnumSubs {}\ntemperature {}",
                 isteps, rc, skin, timestep, rho, numSubs, temperature);
    auto particles_per_direction = 32;
    auto num_particles =
        particles_per_direction * particles_per_direction * particles_per_direction;

    auto cubicData = createCubic(num_particles, rho, true);
    spdlog::info("num particles before replication {}", num_particles);
    auto cubicData_replicated = replicate(cubicData, 2, 2, 2);
    num_particles = (int)cubicData_replicated.x.size();
    spdlog::flush_every(std::chrono::seconds(5));

    auto density = num_particles /
                   (cubicData_replicated.Lx * cubicData_replicated.Ly * cubicData_replicated.Lz);
    spdlog::info(
        "particles_per_direction {}\n"
        "num_particles {}\n"
        "density {}\n",
        particles_per_direction, num_particles, density);


    auto [system, integrator, storage] =
        make_hpx_system(cubicData_replicated.make_box(), numSubs, 1, rc, skin, timestep, 1,
                        temperature, num_threads);

    std::ifstream f("../../src/tools/particle_velocity.json");
//    json velocities = json::parse(f);
    spdlog::info("System Online :)");
//    auto max_size = velocities[0].size();
    auto skip_every = my_rank%2==0;
    for (int i = 0; i < num_particles; ++i)
    {
        // check distance to origin of box.
        // r of sphere should be less than half of the box.
//        //
//        if(skip_every && i%2==0){
//            continue ;
//        }
        auto particle = system->storage->addParticle(
            i + 1, espressopp::Real3D(cubicData_replicated.x[i], cubicData_replicated.y[i],
                                      cubicData_replicated.z[i]));

        if (particle != nullptr)
        {
            particle->setMass(1);
            particle->setType(0);
//            particle->setV(espressopp::Real3D(velocities[0][i % max_size].get<espressopp::real>(),
//                                              velocities[1][i % max_size].get<espressopp::real>(),
//                                              velocities[2][i % max_size].get<espressopp::real>()));
        }
        if (i % 10000 == 0)
        {
            storage->decomposeHPX();
            SPDLOG_DEBUG("Decomposing {}", i);
        }
    }

    storage->decomposeHPX();
    spdlog::info(" Decomposing DONE ");

    auto vl = std::make_shared<espressopp::hpx4espp::VerletList>(system, storage, rc, true, true);
    auto potLJ = espressopp::hpx4espp::interaction::LennardJones();
    ////Include this to get the error very large index
    potLJ.setEpsilon(1.0);
    potLJ.setSigma(1.0);
    potLJ.setCutoff(rc);
    potLJ.setShift(0.0);

    auto interLJ = std::make_shared<espressopp::hpx4espp::interaction::VerletListLennardJones>(vl);
    interLJ->setPotential(0, 0, potLJ);
    system->addInteraction(interLJ);

    spdlog::info(info(system, integrator, false).str());
    integrator->resetTimers();
    auto logging_size = 100;
//    for (size_t i = 0; i < isteps; i+=logging_size)
//    {
        integrator->run_(isteps);
        spdlog::info("\n{}", info(system, integrator, false).str());
        spdlog::info("\n{}", final_info(system, integrator, vl).str());
//    }

    espressopp::real timers[10];
    integrator->loadTimers(timers);

    spdlog::info("\n{}", log_timer(timers).str());

    spdlog::info("\n{}", info(system, integrator, false).str());

    spdlog::info("\n{}", final_info(system, integrator, vl).str());

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
    std::time_t t = std::time(nullptr);

    // clang-format off

    desc_commandline.add_options()
            ("id", value<std::string>()->default_value(fmt::format("{:%d-%m %H:%M:%S}", spdlog::fmt_lib::localtime(t))),
             "ID to be used for the run");
    desc_commandline.add_options()
            ("isteps", value<std::size_t>()->default_value(100),
             "Number of steps to be performed");

desc_commandline.add_options()
            ("temp", value<int>()->default_value(1),
             "Equilibrium temperature of the system");

desc_commandline.add_options()
            ("numSubs", value<std::size_t>()->default_value(32),
             "Number of steps to be performed");

desc_commandline.add_options()
            ("exchangeGhosts_impl_john_opt", value<bool>()->default_value(false),
             "Number of steps to be performed");

desc_commandline.add_options()
            ("ghostCommunication_impl_john_opt", value<bool>()->default_value(false),
             "Number of steps to be performed");

desc_commandline.add_options()
            ("decomposeRealParticlesHPXParFor_john_opt", value<bool>()->default_value(false),
             "Number of steps to be performed");

desc_commandline.add_options()
            ("resort", value<bool>()->default_value(true),
             "Number of steps to be performed");

    std::vector <std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
