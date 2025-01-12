#include"./include/n_body_cpu_class.cuh"
#include"./include/n_body_gpu_class.cuh"
#include<string>
#include<chrono>
#include<iostream>

int main(){
    //auto sim = SimulatorCPU::NBodySimulatorCPU_PM();
    //auto sim = SimulatorCPU::NBodySimulatorCPU_PP();
    //auto sim = SimulatorCPU::NBodySimulatorCPU_PP_OpenBC();
    auto sim = SimulatorGPU::NBodySimulatorGPU_PP_OpenBC();
    const unsigned int N = 1024;
    const unsigned int mesh_num= 64;
    const float box_size = std::cbrt((float)N);
    const float softening_epsilon = box_size / 100.0;
    const float dt = 0.0001;
    const int total_steps = 10000;
    const std::string output_file_name = "data/history.txt";

    std::cout << "N=" << N << std::endl;
    std::cout << "box size=" << box_size << std::endl;
    std::cout << "mesh num=" << mesh_num << std::endl;
    //sim.initialize(N, box_size, mesh_num, dt, output_file_name); // initialize of PM
    sim.initialize(N, box_size, softening_epsilon, dt, output_file_name); // initialize of PP

    sim.dump_coordinate(-1, "first");
    const auto st = std::chrono::system_clock::now();
    for(unsigned int step = 0; step < total_steps; step++) {
        sim.evolve_single_step();
        if(step % 1000 == 0) { 
            //sim.dump_coordinate(step);
            //sim.show_total_energy();
            //sim.show_CoM();
        }
    }
    const auto end = std::chrono::system_clock::now();
    const auto time_milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout << time_milli.count() << " [ms]/ " << total_steps << "steps" << std::endl;
    
    sim.dump_coordinate(-2, "last");
    sim.ending();
    return 0;
}