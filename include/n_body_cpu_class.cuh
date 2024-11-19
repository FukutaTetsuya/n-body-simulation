#include<stdio.h>
#include<fstream>
#include<random>
#include <memory>
#include<string>
#include "n_body_class.cuh"

namespace SimulatorCPU{
class NBodySimulatorCPU : public SimulatorBase::NBodySimulator {
private:
    std::unique_ptr<float[]> mass_dt;
public:
    void initialize(int passed_N, float passed_dt, std::string file_name) override {
        coordinate_file_name = file_name;
        N = passed_N;
        dt = passed_dt;
        mass_dt = std::make_unique<float[]>(N);
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
                    mass_dt[n] = 1.0 * dt;
                    r[0][n] = x;
                    r[1][n] = y;
                    r[2][n] = z;
                    v[0][n] = 0.0;
                    v[1][n] = 0.0;
                    v[2][n] = 0.0;
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
        update_accelaration();
        return;
    }

    void evolve_single_step(void) override{
        //update x += v*dt
        update_coordinate();
        //update v' += (a/2)*dt
        update_velocity_half_step();
        //update a = f(x)/m
        update_accelaration();
        //update v += (a/2)*dt
        update_velocity_half_step();
        return;
    }

    void show_total_energy(void) const override{
        printf("cpu shot total energy\n");
        return;
    }
     
    void dump_coordinate(int step, std::string first_or_last_item = "neither") const override {
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

private:
    void update_coordinate() {
        for(int i = 0; i < N; i++) {
            r[0][i] += v[0][i] * dt;
            r[1][i] += v[1][i] * dt;
            r[2][i] += v[2][i] * dt;
        }
        return;
    }

    void update_velocity_half_step() {
        const float dt_half = dt * 0.5;
        for(int i = 0; i < N; i++) {
            v[0][i] += a[0][i] * dt_half;
            v[1][i] += a[1][i] * dt_half;
            v[2][i] += a[2][i] * dt_half;
        }
        return;
    }

    void update_accelaration() {
        //reset a[0,1,2][0,...,N-1] = 0.0
        for(int i = 0; i < N; i++) {
            a[0][i] = 0.0;
            a[1][i] = 0.0;
            a[2][i] = 0.0;
        }
        //see all particle pair
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float mass_i_dt = mass_dt[i];
            for(int j = 0; j < i; j++) {
                const float mass_j_dt = mass_dt[j];
                const float xij = r[0][j] - x;
                const float yij = r[1][j] - y;
                const float zij = r[2][j] - z;
                const float dr_square = xij*xij + yij*yij + zij*zij;
                const float dr_three_two = dr_square * std::sqrt(dr_square);
                float dUdx = - xij / dr_three_two;
                float dUdy = - yij / dr_three_two;
                float dUdz = - zij / dr_three_two;
                a[0][i] -= mass_j_dt * dUdx;
                a[1][i] -= mass_j_dt * dUdy;
                a[2][i] -= mass_j_dt * dUdz;
                a[0][j] += mass_i_dt * dUdx;
                a[1][j] += mass_i_dt * dUdy;
                a[2][j] += mass_i_dt * dUdz;
            }
        }
        return;
    }


};
}