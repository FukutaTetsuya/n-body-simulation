#include"n_body_cpu_class.cuh"
#include<string>

int main(){
    auto sim = SimulatorCPU::NBodySimulatorCPU();
    const unsigned int N = 32;
    const float dt = 0.01;
    float t = 0.0;
    const std::string output_file_name = "data/history.txt";
    sim.initialize(N, dt, output_file_name);

    sim.dump_coordinate(0, "first");
    for(unsigned int step = 0; step < 10; step++) {
        t += dt;
        sim.evolve_single_step();
    }
    sim.dump_coordinate(0, "last");
    sim.ending();
    return 0;
}