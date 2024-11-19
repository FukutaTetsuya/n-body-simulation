#include<stdio.h>
#include<random>
#include <memory>
#include<string>
#include "n_body_class.cuh"

namespace SimulatorCPU{
class NBodySimulatorCPU : public SimulatorBase::NBodySimulator {
public:
    void initialize(int passed_N, float passed_dt, std::string file_name) override {
        printf("cpu class initialize\n");
        coordinate_file_name = file_name;
        N = passed_N;
        dt = passed_dt;
        mass = std::make_unique<float[]>(N);
        for(int i = 0; i < 3; i++) {
            r[i] = std::make_unique<float[]>(N);
            v[i] = std::make_unique<float[]>(N);
            a[i] = std::make_unique<float[]>(N);
        }
        std::random_device seed;
        std::mt19937 engine(seed());
        const int cbrt_N = (int)(std::cbrt((float)N)) + 1;
        const float init_distance = 2.0;
        const float init_rand_range = 0.1;
        std::uniform_real_distribution<float> rand(-init_rand_range, +init_rand_range);
        double CoM_x = 0.0;
        double CoM_y = 0.0;
        double CoM_z = 0.0;
        for(int i = 0; i < cbrt_N; i++) {
            const int two_third_N = cbrt_N * cbrt_N;
            for(int j = 0; j < cbrt_N; j++) {
                for(int k = 0; k < cbrt_N; k++) {
                    const int n = i * two_third_N + j * cbrt_N + k;
                    if(n>=N) {
                        continue;
                    }
                    const float x = (float)i * init_distance + rand(engine);
                    const float y = (float)j * init_distance + rand(engine);
                    const float z = (float)k * init_distance + rand(engine);
                    mass[n] = 1.0;
                    r[0][n] = x;
                    r[1][n] = y;
                    r[2][n] = z;
                    CoM_x += x;
                    CoM_y += y;
                    CoM_z += z;
                }
            }
        }
        CoM_x /= (double)N;
        CoM_y /= (double)N;
        CoM_z /= (double)N;
        for(int i = 0; i < N; i++) {
            r[0][i] -= CoM_x;
            r[1][i] -= CoM_y;
            r[2][i] -= CoM_z;
        }
        return;
    }

    void evolve_single_step(void) override{
        printf("cpu evolve single step\n");
        return;
    }
    void show_total_energy(void) const override{
        printf("cpu shot total energy\n");
        return;
    }
    void dump_coordinate(void) const override{
        printf("cpu dump coorinate\n");
        return;
    }
    void ending(void) override{
        printf("cpu ending\n");
        return;
    }
};
}