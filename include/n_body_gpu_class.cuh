#include<cstdint>
#include<cuda_runtime.h>
#include<curand.h>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<string>

#define CUDA_CALL(x) do {if((x) != cudaSuccess){\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);\
    }} while(0)

namespace SimulatorGPU {
using uint = std::uint32_t;
namespace Kernels{
    __global__ void expand_coordinate(const int N, const float L, float* r) {
        const int index = blockIdx.x;
        if(index >= N*3) {return;}
        r[index] = r[index] * L - L * 0.5;
        return;
    }

    __global__ void set_initial_velocity(const int N, float* v) {
        const int index = blockIdx.x;
        if(index >= N*3) {return;}
        v[index] = 0.0;
        return;
    }

    __global__ void set_mass(const int N, float* mass) {
        const int index = blockIdx.x;
        if(index >= N) {return;}
        mass[index] = 1.0;
        return;
    }

    __global__ void update_coordinate(const int N, const float dt, float* r, float* v, float* a) {
        const float dt_square = dt * dt;
        const int index = blockIdx.x;
        if(index >= 3*N) {return;}
        r[index] += v[index] * dt;
        r[index] += a[index] * dt_square;
        return;
    }

    __global__ void update_velocity_half_step(const int N, const float dt, float* v, float* a) {
        const float dt_half = dt * 0.5;
        const int index = blockIdx.x;
        if(index >= 3*N) {return;}
        v[index] += a[index] * dt_half;
        return;
    }

    __global__ void update_accelaration(const int N, const float softening_epsilon, float* r, float* mass, float* a) {
        const int index = blockIdx.x;
        if(index >= N) {return;}
        // このアクセスよくないっすねえ
        const int x_index = index;
        const int y_index = index + N;
        const int z_index = index + 2*N;
        const float x = r[x_index];
        const float y = r[y_index];
        const float z = r[z_index];
        float a_x = 0.0;
        float a_y = 0.0;
        float a_z = 0.0;
        for(int j = 0; j < N; j++) {
            if(j==index) {continue;}
            const float mass_j = mass[j];
            float xij = r[j] - x;
            float yij = r[j+N] - y;
            float zij = r[j+2*N] - z;
            const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
            const float inv_dr_three_two = 1.0 / (dr_square * sqrtf(dr_square));
            // ポテンシャルの偏微分に-1を掛けたもの
            float dUdx = mass_j * xij * inv_dr_three_two;
            float dUdy = mass_j * yij * inv_dr_three_two;
            float dUdz = mass_j * zij * inv_dr_three_two;
            // 重力による加速度において自分の質量は相殺する
            a_x += dUdx;
            a_y += dUdy;
            a_z += dUdz;
        }
        a[x_index] = a_x;
        a[y_index] = a_y;
        a[z_index] = a_z;
        return;
    }

    __global__ void update_single_particle_energy(const int N, const float softening_epsilon, float* r, float* mass, float* v, float* single_particle_energy) {
        const int index = blockIdx.x;
        if(index >= N) {return;}

        const float half_mass = 0.5 * mass[index];
        // ポテンシャルエネルギー
        float single_U = 0.0;
        const float x = r[index];
        const float y = r[index + N];
        const float z = r[index + 2*N];
        for(int j = 0; j < N; j++) {
            if(j==index) {continue;}
            const float xij = r[j] - x;
            const float yij = r[j+N] - y;
            const float zij = r[j+2*N] - z;
            const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
            const float dr = sqrtf(dr_square);
            single_U -= mass[j] / dr;
        }
        single_U *= half_mass;

        // 運動エネルギー
        const float vx = v[index];
        const float vy = v[index + N];
        const float vz = v[index + 2*N];
        const float single_K = half_mass * (vx*vx + vy*vy + vz*vz);

        single_particle_energy[index] = single_U + single_K;
 
        return;
    }

    __global__ void reduce_and_show_total_energy(const int N, float* single_particle_energy) {
        const int index = blockIdx.x;
        if(index > 0) {return;}
        float total_energy = 0.0;
        // 粒子数が少ないので単純なループ
        for(int i = 0; i < N; i++) {
            total_energy += single_particle_energy[i];
        }
        printf("total energy = %f\n", total_energy);
        return;
    }

    __global__ void reduce_coordinate_and_show_CoM(const int N, float* r, float* mass) {
        const int index = blockIdx.x;
        if(index > 0) {return;}
        float CoM[3] = {0.0, 0.0, 0.0};
        // 粒子数が少ないので単純なループ
        for(int i = 0; i < N; i++) {
            const float m = mass[i];
            CoM[0] += (double)(m * r[i]);
            CoM[1] += (double)(m * r[N + i]);
            CoM[2] += (double)(m * r[2*N + i]);
        }
        printf("CoM x,y,z = %f,%f,%f\n", CoM[0], CoM[1], CoM[2]);
        return;
    }

};
class NBodySimulatorGPU_PP_OpenBC {
    // N体シミュレーションのGPU実装
    // 手法は、
    // - 時間発展は速度Verlet法
    // - 相互作用はparticle-particle法。全粒子ペアの相互作用を直接計算する
    // - 空間は開放境界条件 どこかに飛んで行った粒子はそのまま
    // - 0割り回避のためPlummer modelのカットオフを採用 ポテンシャルを-1/sqrt(dr^2 + epsilon)とする
    // 初期位置は乱数
    // 質量は1.0で固定している
private:
    float softening_epsilon;
    int N;
    float L; // 初期化時の粒子分布の範囲
    float dt;
    std::string coordinate_file_name;
    float* mass;
    float* r; // 3次元分まとめて格納 x_i = r[i], y_i = r[i+N], z_i = r[i+2N]
    float* v; // 3次元分まとめて格納 インデックスはrと同じ
    float* a; // 3次元分まとめて格納 インデックスはrと同じ
    float* single_particle_energy;
    float* host_r;

public:
    // TODO メモリを扱うのでコンストラクタ・デストラクタに置き換えるべき
    __host__ void initialize(int given_N, float given_L, float given_epsilon, float given_dt, std::string file_name) {
        coordinate_file_name = file_name;
        N = given_N;
        L = given_L;
        softening_epsilon = given_epsilon;
        dt = given_dt;
        CUDA_CALL(cudaMalloc((void **)(&mass), N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&r), 3 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&v), 3 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&a), 3 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&single_particle_energy), N * sizeof(float)));
        host_r = new float[3 * N];
        set_initial_coordinate_velocity_mass();
        Kernels::update_accelaration<<<N,1,0,0>>>(N, softening_epsilon, r, mass, a);
        return;
    }

    __host__ void ending(void) {
        CUDA_CALL(cudaFree((void *)mass));
        CUDA_CALL(cudaFree((void *)r));
        CUDA_CALL(cudaFree((void *)v));
        CUDA_CALL(cudaFree((void *)a));
        CUDA_CALL(cudaFree((void *)single_particle_energy));
        delete[] host_r;
        return;
    }

    __host__ void evolve_single_step(void) {
        //update x += v*dt + (a/2)*dt^2
        Kernels::update_coordinate<<<N*3,1,0,0>>>(N, dt, r, v, a);
        //update v' += (a/2)*dt
        Kernels::update_velocity_half_step<<<N*3,1,0,0>>>(N, dt, v, a);
        //update a = f(x)/m
        Kernels::update_accelaration<<<N,1,0,0>>>(N, softening_epsilon, r, mass, a);
        //update v += (a/2)*dt
        Kernels::update_velocity_half_step<<<N*3,1,0,0>>>(N, dt, v, a);
        return;
    }

    __host__ void show_total_energy(void) {
        // 各粒子のエネルギーを計算
        Kernels::update_single_particle_energy<<<N,1,0,0>>>(N, softening_epsilon, r, mass, v, single_particle_energy);
        // 総和をとる
        Kernels::reduce_and_show_total_energy<<<1,1,0,0>>>(N, single_particle_energy);
        return;
    }
     
    __host__ void dump_coordinate(int step, std::string first_or_last_item = "neither") const {
        // ホストにコピー
        cudaMemcpy((void *)host_r, (void *)r, 3 * N * sizeof(float), cudaMemcpyDeviceToHost);
        std::string output_data = "";
        // append
        auto open_mode = std::ios::app;
        if(first_or_last_item == "first") {
            // 最初の1回は既存のファイルを消すモードで開く
            open_mode = std::ios::out;
            // 最初の1回はlistを開く
            output_data = "[{";
        } else{
            output_data = "{";
        }
        output_data += "\"t\":" + std::to_string(step) + ",";

        output_data += "\"x\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(host_r[i]) + ",";
        }
        output_data += std::to_string(host_r[N - 1]) + "],";

        output_data += "\"y\":[";
        for(int i = N; i < 2*N - 1; i++)
        {
            output_data += std::to_string(host_r[i]) + ",";
        }
        output_data += std::to_string(host_r[2*N - 1]) + "],";

        output_data += "\"z\":[";
        for(int i = 2*N; i < 3*N - 1; i++)
        {
            output_data += std::to_string(host_r[i]) + ",";
        }
        output_data += std::to_string(host_r[3*N - 1]) + "]";

        if(first_or_last_item == "last") {
            // 最後の1回はlistを閉じる
            output_data += "}]\n";
        } else {
            output_data += "},\n";
        }

        std::ofstream file(coordinate_file_name, open_mode);
        file << output_data;
        file.close();
        return;
    }

    __host__ void show_CoM(void) const {
        Kernels::reduce_coordinate_and_show_CoM<<<1,1,0,0>>>(N, r, mass);
        return;
    }
    
private:
    __host__ void set_initial_coordinate_velocity_mass(void) {
        curandGenerator_t gen;
        unsigned long long seed = 707;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen,seed);
        curandGenerateUniform(gen, r, N*3);
        Kernels::expand_coordinate<<<N*3,1,0,0>>>(N, L, r);
        Kernels::set_initial_velocity<<<N*3,1,0,0>>>(N, v);
        Kernels::set_mass<<<N,1,0,0>>>(N, mass);
        curandDestroyGenerator(gen);
        return;
    }
};
}