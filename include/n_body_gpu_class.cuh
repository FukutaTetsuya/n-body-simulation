#include<cstdint>
#include<cuda_runtime.h>
#include<curand.h>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<string>
#include <vector_functions.h>
#include <vector_types.h>

#define CUDA_CALL(x) do {if((x) != cudaSuccess){\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);\
    }} while(0)

namespace SimulatorGPU {
using uint = std::uint32_t;
namespace Kernels{
    __global__ void expand_coordinate(const int N, const float L, float* r) {
        const int index = blockIdx.x;
        if(index >= N*4) {return;}
        r[index] = r[index] * L - L * 0.5;
        return;
    }

    __global__ void set_initial_velocity(const int N, float* v) {
        const int index = blockIdx.x;
        if(index >= N*4) {return;}
        v[index] = 0.0;
        return;
    }

    __global__ void set_initial_acceleration(const int N, float* a) {
        const int index = blockIdx.x;
        if(index >= N*4) {return;}
        a[index] = 0.0;
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
        if(index >= 4*N) {return;}
        r[index] += v[index] * dt;
        r[index] += a[index] * dt_square;
        return;
    }

    __global__ void update_velocity_half_step(const int N, const float dt, float* v, float* a) {
        const float dt_half = dt * 0.5;
        const int index = blockIdx.x;
        if(index >= 4*N) {return;}
        v[index] += a[index] * dt_half;
        return;
    }

    __global__ void update_accelaration(const int N, const float softening_epsilon, void* r, float* mass, void* a) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        // index >= Nでも相手方の座標を拾ってくる任務があるためここではリターンしない
        // 力の計算も行うが虚無である
        const uint index_in_block = threadIdx.x;
        const uint particle_per_block = blockDim.x;
        float4* global_r = (float4*)r;
        float4* global_a = (float4*)a;
        float4 self_r = make_float4(0.0, 0.0, 0.0, 0.0);
        if(index < N) {
            self_r = global_r[index];
        }
        float a_x = 0.0;
        float a_y = 0.0;
        float a_z = 0.0;
        extern __shared__ float4 shared_r[];
        // タイルが相手方のN粒子を取りつくすまでのループ
        for(uint i = 0; i < N; i += particle_per_block) {
            // 座標をsmemにコピー
            const uint fetch_index = i + index_in_block;
            if(fetch_index < N) {
                shared_r[index_in_block] = global_r[i + index_in_block];
                // TODO 座標と質量をまとめて格納しておくとここで合理的だな
                (shared_r[index_in_block]).w = mass[index];
            } else {
                shared_r[index_in_block] = make_float4(0.0, 0.0, 0.0, 0.0);
            }
            __syncthreads();
            // smemにコピーした粒子との相互作用を計算
            // ゼロ割回避のepsilonを入れているので、自分自身との力はゼロになる すべてのインデックスに対して計算してしまって構わない
            // 相手粒子がなくても、質量にゼロを代入しているのでそのまま計算してかまわない
            for(uint j = 0; j < particle_per_block; j++){
                const float xij = shared_r[j].x - self_r.x;
                const float yij = shared_r[j].y - self_r.y;
                const float zij = shared_r[j].z - self_r.z;
                const float mass_j = shared_r[j].w;
                const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
                const float inv_dr_three_two = 1.0 / (dr_square * sqrtf(dr_square));
                // ポテンシャルの偏微分に-1を掛けたもの
                const float dUdx = mass_j * xij * inv_dr_three_two;
                const float dUdy = mass_j * yij * inv_dr_three_two;
                const float dUdz = mass_j * zij * inv_dr_three_two;
                // 重力による加速度において自分の質量は相殺する
                a_x += dUdx;
                a_y += dUdy;
                a_z += dUdz; 
            }
            __syncthreads();
        }
        // 結果をglobalに返す
        if(index < N) {
            global_a[index] = make_float4(a_x, a_y, a_z, 0.0);
        }
        return;
    }

    __global__ void update_single_particle_energy(const int N, const float softening_epsilon, float* r, float* mass, float* v, float* single_particle_energy) {
        const int index = blockIdx.x;
        if(index >= N) {return;}

        const float half_mass = 0.5 * mass[index];
        // ポテンシャルエネルギー
        float single_U = 0.0;
        const float x = r[index*4];
        const float y = r[index*4 + 1];
        const float z = r[index*4 + 2];
        for(int j = 0; j < N; j++) {
            if(j==index) {continue;}
            const float xij = r[j*4] - x;
            const float yij = r[j*4 + 1] - y;
            const float zij = r[j*4 + 2] - z;
            const float dr_square = xij*xij + yij*yij + zij*zij + softening_epsilon;
            const float dr = sqrtf(dr_square);
            single_U -= mass[j] / dr;
        }
        single_U *= half_mass;

        // 運動エネルギー
        const float vx = v[index*4];
        const float vy = v[index*4 + 1];
        const float vz = v[index*4 + 2];
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
            CoM[0] += (double)(m * r[i*4]);
            CoM[1] += (double)(m * r[i*4 + 1]);
            CoM[2] += (double)(m * r[i*4 + 2]);
        }
        printf("CoM x,y,z = %f,%f,%f\n", CoM[0], CoM[1], CoM[2]);
        return;
    }

};
class NBodySimulatorGPU_PP {
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
    float* r; // 3次元分まとめて格納 コアレスアクセスを実現するため1次元余分に確保する x_i = r[i*4], y_i = r[i*4+1], z_i = r[i*4+2], r[i*4+3]には意味のない値が入る
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
        CUDA_CALL(cudaMalloc((void **)(&r), 4 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&v), 4 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&a), 4 * N * sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)(&single_particle_energy), N * sizeof(float)));
        host_r = new float[4 * N];
        set_initial_coordinate_velocity_mass();
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
        const uint particle_per_block = 16;
        const dim3 grid_dim = dim3((N + particle_per_block - 1) / particle_per_block, 1, 1);
        const dim3 block_dim = dim3(particle_per_block, 1, 1);
        const uint shared_memory_size_byte = particle_per_block * 4 * sizeof(float); // float4を使いたいので無駄を承知

        //update x += v*dt + (a/2)*dt^2
        Kernels::update_coordinate<<<N*4,1,0,0>>>(N, dt, r, v, a);
        //update v' += (a/2)*dt
        Kernels::update_velocity_half_step<<<N*4,1,0,0>>>(N, dt, v, a);
        //update a = f(x)/m
        Kernels::update_accelaration<<<grid_dim,block_dim,shared_memory_size_byte,0>>>(N, softening_epsilon, (void *)r, mass, (void *)a);
        //update v += (a/2)*dt
        Kernels::update_velocity_half_step<<<N*4,1,0,0>>>(N, dt, v, a);
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
        cudaMemcpy((void *)host_r, (void *)r, 4 * N * sizeof(float), cudaMemcpyDeviceToHost);
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
            output_data += std::to_string(host_r[i*4]) + ",";
        }
        output_data += std::to_string(host_r[4*N - 4]) + "],";

        output_data += "\"y\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(host_r[i*4 + 1]) + ",";
        }
        output_data += std::to_string(host_r[4*N - 3]) + "],";

        output_data += "\"z\":[";
        for(int i = 0; i < N - 1; i++)
        {
            output_data += std::to_string(host_r[i*4 + 2]) + ",";
        }
        output_data += std::to_string(host_r[4*N - 2]) + "]";

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
        curandGenerateUniform(gen, r, N*4);
        Kernels::expand_coordinate<<<N*4,1,0,0>>>(N, L, r);
        Kernels::set_initial_velocity<<<N*4,1,0,0>>>(N, v);
        Kernels::set_initial_acceleration<<<N*4,1,0,0>>>(N, a);
        Kernels::set_mass<<<N,1,0,0>>>(N, mass);
        curandDestroyGenerator(gen);
        return;
    }
};
}