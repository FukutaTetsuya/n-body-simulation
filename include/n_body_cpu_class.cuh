#include<iostream>
#include<fstream>
#include<random>
#include<memory>
#include<vector>
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

class NBodySimulatorCPU_PP_OpenBC : public SimulatorBase::NBodySimulator {
    // N体シミュレーションのCPU実装
    // 手法は、
    // - 時間発展は速度Verlet法
    // - 相互作用はparticle-particle法。全粒子ペアの相互作用を直接計算する
    // - 空間は開放境界条件 どこかに飛んで行った粒子はそのまま
    // - 0割り回避のためPlummer modelのカットオフを採用 ポテンシャルを-1/sqrt(dr^2 + epsilon)とする
    // 初期位置は乱数
    // 質量は1.0で固定している
private:
    float softening_epsilon;

public:
    void initialize(int given_N, float given_L, float given_epsilon, float given_dt, std::string file_name) {
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
        std::uniform_real_distribution<float> rand(-L*0.5, +L*0.5);
        for(int i = 0; i < N; i++) {
            r[0][i] = rand(engine);
            r[1][i] = rand(engine);
            r[2][i] = rand(engine);
            mass[i] = 1.0;
            v[0][i] = 0.0;
            v[1][i] = 0.0;
            v[2][i] = 0.0;
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
        float total_U = 0.0;
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float mass_i = mass[i];
            for(int j = 0; j < i; j++) {
                //if(j==i) {continue;}
                const float mass_j = mass[j];
                float xij = r[0][j] - x;
                float yij = r[1][j] - y;
                float zij = r[2][j] - z;
                const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
                const float dr = std::sqrt(dr_square);
                total_U -= mass_i * mass_j / dr;
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

    void show_CoM(void) const {
        double CoM[3] = {0.0, 0.0, 0.0};
        for(int i = 0; i < N; i++) {
            const float m = mass[i];
            CoM[0] += (double)(m * r[0][i]);
            CoM[1] += (double)(m * r[1][i]);
            CoM[2] += (double)(m * r[2][i]);
        }
        std::cout << "CoM x,y,z = " << CoM[0] << "," << CoM[1] << "," << CoM[2] << std::endl;
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
            r[1][i] += v[1][i] * dt;
            r[1][i] += a[1][i] * dt_square;
            r[2][i] += v[2][i] * dt;
            r[2][i] += a[2][i] * dt_square;
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
            const float mass_i = mass[i];
            for(int j = 0; j < i; j++) {
                const float mass_j = mass[j];
                float xij = r[0][j] - x;
                float yij = r[1][j] - y;
                float zij = r[2][j] - z;
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
    // 質量も配列で持ってはいるけど使わず、全部1とみなしてハードコーディングしてある
private:
    // TODO: M, M_sizeよりましな命名を考える
    unsigned int M; // 1辺当たりの格子点の数 そのうちFFT実装するかもしれないので2のべき乗であってほしい DFTなら違ってもかまわない
    float M_size; // 格子点の間隔
    std::unique_ptr<float[]> density; // 要素数M*M*Mのスカラ場 インデックスはx + y*M + z*M*M
    std::unique_ptr<float[]> potential_DFT_Re; // 要素数M*M*Mのスカラ場 インデックスはx + y*M + z*M*M
    std::unique_ptr<float[]> potential_DFT_Im; // 要素数M*M*Mのスカラ場 インデックスはx + y*M + z*M*M
    std::unique_ptr<float[]> potential_field; // 要素数M*M*Mのスカラ場 インデックスはx + y*M + z*M*M
    std::unique_ptr<float[]> force_field[3]; // 各成分の要素数M*M*Mのベクトル場 i=0,1,2がそれぞれx,y,z成分 インデックスはx + y*M + z*M*M
public:
    void initialize(int given_N, float given_L, unsigned int given_M, float given_dt, std::string file_name) {
        std::cout << "init in PP class" << std::endl;
        coordinate_file_name = file_name;
        N = given_N;
        L = given_L;
        M = given_M;
        dt = given_dt;
        M_size = L / (float)given_M;
        mass = std::make_unique<float[]>(N);
        density = std::make_unique<float[]>(M*M*M);
        potential_DFT_Re = std::make_unique<float[]>(M*M*M);
        potential_DFT_Im = std::make_unique<float[]>(M*M*M);
        potential_field = std::make_unique<float[]>(M*M*M);
        for(int i = 0; i < 3; i++) {
            r[i] = std::make_unique<float[]>(N);
            v[i] = std::make_unique<float[]>(N);
            a[i] = std::make_unique<float[]>(N);
            force_field[i] = std::make_unique<float[]>(M*M*M);
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

    void update_density(void) {
        // 近似的な密度場を求める
        // 近傍8個の格子点に質量を割り振る方式
        auto lattice_point_lengths = std::vector<float>(M);
        for(int i = 0; i < M; i++) {
            lattice_point_lengths[i] = (float)i * M_size;
        }
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            // x,y,z_index = 0..M-1 は原点に近い格子点のインデックス
            const int x_index = (int)(x / M_size);
            const int y_index = (int)(y / M_size);
            const int z_index = (int)(z / M_size);
            // 近傍8格子点が定める立方体を、粒子の位置を通るxy,yz,zx平面で分割
            // その体積の逆比で密度を割り振る
            const float x_diff_ratio = (x - lattice_point_lengths[x_index]) / M_size;
            const float y_diff_ratio = (y - lattice_point_lengths[y_index]) / M_size;
            const float z_diff_ratio = (z - lattice_point_lengths[z_index]) / M_size;
            const float x_diff_ratio_1 = 1.0 - x_diff_ratio;
            const float y_diff_ratio_1 = 1.0 - y_diff_ratio;
            const float z_diff_ratio_1 = 1.0 - z_diff_ratio;
            // 体積の逆比で密度を割り振るといったものの、
            // PM法が割り振るのが質量なのか密度なのか確信がない
            // M_size^3で割っているのは密度の立場
            // 粒子の移動が明らかにトロければ質量が正しかったということにしちゃえばいいか
            const float density_i = mass[i] / (M_size * M_size * M_size);
            density[x_index       + y_index*M         + z_index*M*M]         += density_i * x_diff_ratio_1 * y_diff_ratio_1 * z_diff_ratio_1;
            density[(x_index+1)%M + y_index*M         + z_index*M*M]         += density_i * x_diff_ratio   * y_diff_ratio_1 * z_diff_ratio_1;
            density[x_index       + ((y_index+1)%M)*M + z_index*M*M]         += density_i * x_diff_ratio_1 * y_diff_ratio   * z_diff_ratio_1;
            density[x_index       + y_index*M         + ((z_index+1)%M)*M*M] += density_i * x_diff_ratio_1 * y_diff_ratio_1 * z_diff_ratio;
            density[(x_index+1)%M + ((y_index+1)%M)*M + z_index*M*M]         += density_i * x_diff_ratio   * y_diff_ratio   * z_diff_ratio_1;
            density[(x_index+1)%M + y_index*M         + ((z_index+1)%M)*M*M] += density_i * x_diff_ratio   * y_diff_ratio_1 * z_diff_ratio;
            density[x_index       + ((y_index+1)%M)*M + ((z_index+1)%M)*M*M] += density_i * x_diff_ratio_1 * y_diff_ratio   * z_diff_ratio;
            density[(x_index+1)%M + ((y_index+1)%M)*M + ((z_index+1)%M)*M*M] += density_i * x_diff_ratio   * y_diff_ratio   * z_diff_ratio;
        }
        return;
    }

    void update_potential_DFT(void) {
        // Poisson eqを満たすポテンシャル、のDFTを求める
        // densityのDFTを波数の2乗で割って係数を掛けたものがそれである 多分
        const float coefficient = -(L * L) / (12.0 * M_PI * M_PI);

        // k{x,y,z} = 0..M : 波数
        // n{x,y,z} = 0..M : 実空間の格子点のインデックス
        for(int kx = 0; kx < M; kx++) {
            for(int ky = 0; ky < M; ky++) {
                for(int kz = 0; kz < M; kz++) {
                    const int k_index = kx + ky*M + kz*M*M;
                    const float k_square = (float)(kx*kx + ky*ky + kz*kz);
                    if(k_square < 0.5) {
                        // NOTE: 波数0は格子点における密度の総和になるが、
                        // 逆変換の際はすべての格子点のポテンシャルに係数1(cos0)で足しこまれるため、
                        // 離散微分して力に変換すると相殺する
                        // どんな値でも影響がないので0を入れておく
                        // …本来はゼロ割りされる部分なので数学的に正当なのかは知らん
                        potential_DFT_Re[k_index] = 0.0;
                        potential_DFT_Im[k_index] = 0.0;
                        continue;
                    }
                    float density_DFT_Re = 0.0;
                    float density_DFT_Im = 0.0;
                    for(int nx = 0; nx < M; nx++) {
                        for(int ny = 0; ny < M; ny++) {
                            for(int nz = 0; nz < M; nz++) {
                                const float density_n = density[nx + ny*M + nz*M*M];
                                const float theta = 2.0 * M_PI * (nx*kx + ny*ky + nz*kz) / (float)M;
                                density_DFT_Re += density_n * std::cos(theta);
                                density_DFT_Im += density_n * std::sin(theta);
                            }
                        }
                    }
                    potential_DFT_Re[k_index] = coefficient * density_DFT_Re / k_square;
                    potential_DFT_Im[k_index] = coefficient * density_DFT_Im / k_square;
                }
            }
        }
        return;
    }

    void update_potential_field(void) {
        // ポテンシャルのDFTを逆DFTして実空間のポテンシャル場を求める
        const float theta_coefficient = -2.0 * M_PI / (float)M;
        for(int nx = 0; nx < M; nx++) {
            for(int ny = 0; ny < M; ny++) {
                for(int nz = 0; nz < M; nz++) {
                    const int n_index = nx + ny*M + nz*M*M;
                    float potential_Re = 0.0;
                    float potential_Im = 0.0;
                    for(int kx = 0; kx < M; kx++) {
                        for(int ky = 0; ky < M; ky++) {
                            for(int kz = 0; kz < M; kz++) {
                                const int k_index = kx + ky*M + kz*M*M;
                                const float theta = theta_coefficient * (nx*kx + ny*ky + nz*kz);
                                potential_Re += potential_DFT_Re[k_index] * std::cos(theta) - potential_DFT_Im[k_index] * std::sin(theta);
                                potential_Im += potential_DFT_Re[k_index] * std::sin(theta) + potential_DFT_Im[k_index] * std::cos(theta);
                            }
                        }
                    }
                    potential_field[n_index] = potential_Re;
                    // potential_Imは大体ゼロのはず
                    if(potential_Im > 1.0e-6) {
                        std::cerr << "Imaginary part of potential is too large" << std::endl;
                    }
                }
            }
        }
        return;
    }

    void update_force_field(void) {
        // 実空間の格子点の重力場を求める
        // 周辺6点のポテンシャルの差分で定める
        const float gradient_coefficient = -1.0 / (2.0 * M_size);
        for(int nx = 0; nx < M; nx++) {
            for(int ny = 0; ny < M; ny++) {
                for(int nz = 0; nz < M; nz++) {
                    const int n_index = nx + ny*M + nz*M*M;
                    force_field[0][n_index] = gradient_coefficient * (potential_field[(nx+1)%M + ny*M + nz*M*M] - potential_field[(nx-1+M)%M + ny*M + nz*M*M]);
                    force_field[1][n_index] = gradient_coefficient * (potential_field[nx + ((ny+1)%M)*M + nz*M*M] - potential_field[nx + ((ny-1+M)%M)*M + nz*M*M]);
                    force_field[2][n_index] = gradient_coefficient * (potential_field[nx + ny*M + ((nz+1)%M)*M*M] - potential_field[nx + ny*M + ((nz-1+M)%M)*M*M]);
                }
            }
        }
        return;
    }

    void update_accelaration_from_force_field(void) {
        // 粒子の加速度を求める
        // 近傍8点の重力場からの補完 密度場への割り当てと同じ係数
        auto lattice_point_lengths = std::vector<float>(M_size);
        for(int i = 0; i < M; i++) {
            lattice_point_lengths[i] = i * M_size;
        }
        for(int i = 0; i < N; i++) {
            const float x = r[0][i];
            const float y = r[1][i];
            const float z = r[2][i];
            const float x_ratio = x / L;
            const float y_ratio = y / L;
            const float z_ratio = z / L;
            // x,y,z_index = 0..M-1 は原点に近い格子点のインデックス
            const int x_index = (int)(x / M_size);
            const int y_index = (int)(y / M_size);
            const int z_index = (int)(z / M_size);
            // 近傍8格子点が定める立方体を、粒子の位置を通るxy,yz,zx平面で分割
            // その体積の逆比を粒子の位置での重力への寄与とする
            // これは線形補完ってやつなのか？
            const float x_diff_ratio = (x - lattice_point_lengths[x_index]) / M_size;
            const float y_diff_ratio = (y - lattice_point_lengths[y_index]) / M_size;
            const float z_diff_ratio = (z - lattice_point_lengths[z_index]) / M_size;
            const float x_diff_ratio_1 = 1.0 - x_diff_ratio;
            const float y_diff_ratio_1 = 1.0 - y_diff_ratio;
            const float z_diff_ratio_1 = 1.0 - z_diff_ratio;

            // 重力は粒子の質量に比例して加速度は逆比例なので相殺する 質量は登場しない
            for(int axis = 0; axis < 3; axis++) {
                float temp_a = 0.0;
                temp_a += force_field[axis][x_index       + y_index*M         + z_index*M*M]         * x_diff_ratio_1 * y_diff_ratio_1 * z_diff_ratio_1;
                temp_a += force_field[axis][(x_index+1)%M + y_index*M         + z_index*M*M]         * x_diff_ratio   * y_diff_ratio_1 * z_diff_ratio_1;
                temp_a += force_field[axis][x_index       + ((y_index+1)%M)*M + z_index*M*M]         * x_diff_ratio_1 * y_diff_ratio   * z_diff_ratio_1;
                temp_a += force_field[axis][x_index       + y_index*M         + ((z_index+1)%M)*M*M] * x_diff_ratio_1 * y_diff_ratio_1 * z_diff_ratio;
                temp_a += force_field[axis][(x_index+1)%M + ((y_index+1)%M)*M + z_index*M*M]         * x_diff_ratio   * y_diff_ratio   * z_diff_ratio_1;
                temp_a += force_field[axis][(x_index+1)%M + y_index*M         + ((z_index+1)%M)*M*M] * x_diff_ratio   * y_diff_ratio_1 * z_diff_ratio;
                temp_a += force_field[axis][x_index       + ((y_index+1)%M)*M + ((z_index+1)%M)*M*M] * x_diff_ratio_1 * y_diff_ratio   * z_diff_ratio;
                temp_a += force_field[axis][(x_index+1)%M + ((y_index+1)%M)*M + ((z_index+1)%M)*M*M] * x_diff_ratio   * y_diff_ratio   * z_diff_ratio;
                a[axis][i] = temp_a;
            }
        }
        return;
    }

    void update_accelaration() {
        std::cout << "update accelaration" << std::endl;
        // 粒子座標からdensityを作成する
        std::cout << "update density" << std::endl;
        update_density();
        // Poisson eqを満たすポテンシャルをDFTしたものを求める
        std::cout << "update potential DFT" << std::endl;
        update_potential_DFT();
        // ポテンシャルの逆DFTを行って実空間のポテンシャル場を求める
        std::cout << "update potential field" << std::endl;
        update_potential_field();
        //各点の重力場を求める
        std::cout << "update force field" << std::endl;
        update_force_field();
        // 各粒子の受ける力を求めて、加速度を更新する
        std::cout << "update acceleration from force field" << std::endl;
        update_accelaration_from_force_field();
        return;
    }
};
}