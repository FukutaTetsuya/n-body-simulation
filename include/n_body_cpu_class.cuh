#include<iostream>
#include<fstream>
#include<random>
#include <memory>
#include<string>
#include "n_body_class.cuh"

namespace SimulatorCPU{
class NBodySimulatorCPU_PP : public SimulatorBase::NBodySimulator {
    // N体シミュレーションのCPU実装
    // 手法は、
    // - 時間発展は速度Verlet法
    // - 相互作用はparticle-particle法。全粒子ペアの相互作用を直接計算する
    // - 空間は周期境界条件を課すが、相互作用は最近接のコピーとのみ計算する
    // - 0割り回避のためPlummer modelのカットオフを採用 ポテンシャルを-1/sqrt(dr^2 + epsilon)とする
    // 初期位置は格子点周りに少し乱数振っている
    // 質量は1.0で固定している
private:
    float softening_epsilon;

public:
    void initialize(int given_N, float given_L, float given_epsilon,float given_dt, std::string file_name) {
        coordinate_file_name = file_name;
        N = given_N;
        L = given_L;
        softening_epsilon = given_epsilon;
        dt = given_dt;
        mass = std::make_unique<float[]>(N);
        for(int i = 0; i < 3; i++) {
            r[i] = std::make_unique<float[]>(N);
            v[i] = std::make_unique<float[]>(N);
            a[i] = std::make_unique<float[]>(N);
        }
        std::random_device seed;
        std::mt19937 engine(seed());
        const int cbrt_N = (int)(std::cbrt((float)N)) + 1;
        const float init_distance = L / (float)cbrt_N;
        const float init_rand_range = init_distance * 0.1;
        std::uniform_real_distribution<float> rand(-init_rand_range, +init_rand_range);
        for(int i = 0; i < cbrt_N; i++) {
            const int two_third_N = cbrt_N * cbrt_N;
            for(int j = 0; j < cbrt_N; j++) {
                for(int k = 0; k < cbrt_N; k++) {
                    const int n = i * two_third_N + j * cbrt_N + k;
                    if(n>=N) {
                        continue;
                    }
                    float x = (float)i * init_distance + rand(engine);
                    if(x < 0) { x += L; }
                    if(x >= L) { x -= L; }
                    float y = (float)j * init_distance + rand(engine);
                    if(y < 0) { y += L; }
                    if(y >= L) { y -= L; }
                    float z = (float)k * init_distance + rand(engine);
                    if(z < 0) { z += L; }
                    if(z >= L) { z -= L; }
                    mass[n] = 1.0;
                    r[0][n] = x;
                    r[1][n] = y;
                    r[2][n] = z;
                    v[0][n] = 0.0;
                    v[1][n] = 0.0;
                    v[2][n] = 0.0;
                }
            }
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
        //see all particle pair
        const float half_L = L / 2.0;
        const float _half_L = -half_L;
        float total_U = 0.0;
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float mass_i = mass[i];
            for(int j = 0; j < N; j++) {
                if(j==i) {continue;}
                const float mass_j = mass[j];
                float xij = r[0][j] - x;
                if(xij > half_L) {xij -= L;}
                if(xij <= _half_L) {xij += L;}
                float yij = r[1][j] - y;
                if(yij > half_L) {yij -= L;}
                if(yij <= _half_L) {yij += L;}
                float zij = r[2][j] - z;
                if(zij > half_L) {zij -= L;}
                if(zij <= _half_L) {zij += L;}
                const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
                const float dr = std::sqrt(dr_square);
                total_U -= 1.0 / dr;
            }
        }
        float total_K = 0.0;
        for(int i = 0; i < N; i++) {
            const float vx = v[0][i];
            const float vy = v[1][i];
            const float vz = v[2][i];
            const float half_mass = 0.5 * mass[i];
            const float single_K = half_mass * (vx*vx + vy*vy + vz*vz);
            total_K += single_K;
        }
        const float total_energy = total_U + total_K;
        std::cout << "E,K,U," << total_energy << "," << total_U << "," << total_K << std::endl;
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
        return;
    }

private:
    void update_coordinate() {
        const float dt_square = dt * dt;
        for(int i = 0; i < N; i++) {
            r[0][i] += v[0][i] * dt;
            r[0][i] += a[0][i] * dt_square;
            if(r[0][i] < 0) { r[0][i] += L; }
            if(r[0][i] >= L) { r[0][i] -= L; }
            r[1][i] += v[1][i] * dt;
            r[1][i] += a[1][i] * dt_square;
            if(r[1][i] < 0) { r[1][i] += L; }
            if(r[1][i] >= L) { r[1][i] -= L; }
            r[2][i] += v[2][i] * dt;
            r[2][i] += a[2][i] * dt_square;
            if(r[2][i] < 0) { r[2][i] += L; }
            if(r[2][i] >= L) { r[2][i] -= L; }
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
        const float half_L = L / 2.0;
        const float _half_L = -half_L;
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float mass_i = mass[i];
            for(int j = 0; j < i; j++) {
                const float mass_j = mass[j];
                float xij = r[0][j] - x;
                if(xij > half_L) {xij -= L;}
                if(xij <= _half_L) {xij += L;}
                float yij = r[1][j] - y;
                if(yij > half_L) {yij -= L;}
                if(yij <= _half_L) {yij += L;}
                float zij = r[2][j] - z;
                if(zij > half_L) {zij -= L;}
                if(zij <= _half_L) {zij += L;}
                const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
                const float dr_three_two = dr_square * std::sqrt(dr_square);
                // partial differential of potential U
                float dUdx = - xij / dr_three_two;
                float dUdy = - yij / dr_three_two;
                float dUdz = - zij / dr_three_two;
                // gravity force is -1 * self mass * \nabla U
                a[0][i] -= mass_j * dUdx;
                a[1][i] -= mass_j * dUdy;
                a[2][i] -= mass_j * dUdz;
                a[0][j] += mass_i * dUdx;
                a[1][j] += mass_i * dUdy;
                a[2][j] += mass_i * dUdz;
            }
        }
        return;
    }
};

class NBodySimulatorCPU_PM : public SimulatorBase::NBodySimulator {
    // N体シミュレーションのCPU実装
    // 手法は、
    // - 時間発展は速度Verlet法
    // - 相互作用はparticle-mesh法。粒子配置を格子点上の密度場近似して格子点上の重力場を求める
    //   粒子の質量は近傍8個の格子点に割り振る
    // - 空間は周期境界条件を課す。相互作用計算にフーリエ変換を使うため、無限個のコピーとの相互作用が計算される
private:
    std::unique_ptr<float[]> mass_dt;
    unsigned int M; // 1辺当たりの格子点の数 FFTの都合で2のべき乗であってほしい
    float M_size; // 格子点の間隔
    std::unique_ptr<float[]> density[3];
public:
    virtual void initialize(int passed_N, float passed_L, float passed_dt, std::string file_name) override {
        std::cout << "init in PP class" << std::endl;
        coordinate_file_name = file_name;
        N = passed_N;
        L = passed_L;
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
        const float init_distance = L / (float)cbrt_N;
        const float init_rand_range = init_distance * 0.1;
        std::uniform_real_distribution<float> rand(-init_rand_range, +init_rand_range);
        for(int i = 0; i < cbrt_N; i++) {
            const int two_third_N = cbrt_N * cbrt_N;
            for(int j = 0; j < cbrt_N; j++) {
                for(int k = 0; k < cbrt_N; k++) {
                    const int n = i * two_third_N + j * cbrt_N + k;
                    if(n>=N) {
                        continue;
                    }
                    float x = (float)i * init_distance + rand(engine);
                    if(x < 0) { x += L; }
                    if(x >= L) { x -= L; }
                    float y = (float)j * init_distance + rand(engine);
                    if(y < 0) { y += L; }
                    if(y >= L) { y -= L; }
                    float z = (float)k * init_distance + rand(engine);
                    if(z < 0) { z += L; }
                    if(z >= L) { z -= L; }
                    mass_dt[n] = 1.0 * dt;
                    r[0][n] = x;
                    r[1][n] = y;
                    r[2][n] = z;
                    v[0][n] = 0.0;
                    v[1][n] = 0.0;
                    v[2][n] = 0.0;
                }
            }
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
        return;
    }

protected:
    void update_coordinate() {
        for(int i = 0; i < N; i++) {
            r[0][i] += v[0][i] * dt;
            if(r[0][i] < 0) { r[0][i] += L; }
            if(r[0][i] >= L) { r[0][i] -= L; }
            r[1][i] += v[1][i] * dt;
            if(r[1][i] < 0) { r[1][i] += L; }
            if(r[1][i] >= L) { r[1][i] -= L; }
            r[2][i] += v[2][i] * dt;
            if(r[2][i] < 0) { r[2][i] += L; }
            if(r[2][i] >= L) { r[2][i] -= L; }
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
        const float half_L = L / 2.0;
        const float _half_L = -half_L;
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float mass_i_dt = mass_dt[i];
            for(int j = 0; j < i; j++) {
                const float mass_j_dt = mass_dt[j];
                float xij = r[0][j] - x;
                if(xij > half_L) {xij -= L;}
                if(xij <= _half_L) {xij += L;}
                float yij = r[1][j] - y;
                if(yij > half_L) {yij -= L;}
                if(yij <= _half_L) {yij += L;}
                float zij = r[2][j] - z;
                if(zij > half_L) {zij -= L;}
                if(zij <= _half_L) {zij += L;}
                const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
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