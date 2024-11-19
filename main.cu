#include"n_body_cpu_class.cuh"
#include<string>

int main(){
    auto sim = SimulatorCPU::NBodySimulatorCPU();
    const unsigned int N = 32;
    const float dt = 0.01;
    float t = 0.0;
    const std::string output_file_name = "history.txt";
    sim.initialize(N, dt, output_file_name);

    sim.dump_coordinate();
    for(unsigned int step = 0; step < 10; step++) {
        t += dt;
        sim.evolve_single_step();
    }
    sim.ending();
    return 0;
}