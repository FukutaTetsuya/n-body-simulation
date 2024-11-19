#include<stdio.h>
#include<fstream>
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
    void dump_coordinate(int step, std::string first_or_last_item = "neither") const override {
        printf("cpu dump coorinate\n");
        std::string output_data = "";
        // append
        auto open_mode = std::ios::app;
        if(first_or_last_item == "first") {
            // discard existing file
            open_mode = std::ios::out;
            output_data = "[{";
        } else{
            output_data = "{";
        }
        output_data += "\"t\":" + std::to_string(step) + ",";

        output_data += "\"x\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(r[0][i]) + ",";
        }
        output_data += std::to_string(r[0][N - 1]) + "],";

        output_data += "\"y\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(r[1][i]) + ",";
        }
        output_data += std::to_string(r[1][N - 1]) + "],";

        output_data += "\"z\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(r[1][i]) + ",";
        }
        output_data += std::to_string(r[1][N - 1]) + "]";

        if(first_or_last_item == "last") {
            output_data += "}]\n";
        } else {
            output_data += "},\n";
        }

        std::ofstream file(coordinate_file_name, open_mode);
        file << output_data;
        file.close();
        return;
    }
    void ending(void) override{
        printf("cpu ending\n");
        return;
    }
};
}