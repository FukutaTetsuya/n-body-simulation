# GPU Gems 31:fast n body simulation cuda

## 31.1 Introduction

## 31.2 All-pairs N-body simulation

## 31.3 A CUDA implementation of the all-pairs N-body algorithm

i,j成分が`$f_{ij}$`であるようなNxN行列を思い浮かべる
力の計算は、この行列のNxN個の要素を計算することと同じである
1スレッドが1粒子=1行を担当するような並列化はデータ転送がNxN必要で、メモリバンド幅が律速になる
これをうまいこと解消して、演算のピーク性能を引き出すことを目標にする

NxN行列の中のpxpの小領域を指して「タイル」と呼ぶことにする
2p個の粒子の情報だけで1枚のタイルを計算でき、半分は再利用できる
NxN行列のうちp行をpスレッドで計算する、p行をさらにp列ずつに分けて計算する、ということ
2p個の情報はsmemやレジスタにおいておける
入出力のためのメモリアクセスはなるべく連番にする

また、ボトムアップで1対1の力の計算から実装していく

## 31.3.1  Body-body force calculation

ここは素直な実装 わざわざサブルーチンに切り出すんだという感想
複数のfloatをやり取りするためにfloat4型を使っている
`bodyBodyInteraction: (float4 selfPosition, float4 otherPosition, float4 selfOldAcceleration) --> (float4 selfNewAcceleration)`

## 31.3.2 Tile calculation

NxN行列の1つのタイルをp本のスレッドで計算する
p本のスレッドがp回ずつ同じ(SIMDな)計算を行い、p個の加速度を更新する
`tile_calculation: (float4 myPosition, float3 selfOldAcceleration) --> (float3 selfNewAcceleration)`

```c++
__device__ float3
tile_calculation(float4 myPosition, float3 accel)
{
  // assert(blockDim == (p,1,1));
  int i;
  // p other particles' positions maybe copied in otherPositions[] in step
  extern __shared__ float4[] otherPositions;
  for (i = 0; i < blockDim.x; i++) {
    accel = bodyBodyInteraction(myPosition, otherPositions[i], accel);
  }
  return accel;
}
```

## 31.3.3 Clustering Tiles into Thread Blocks

1つのスレッドブロックが1行のタイルを処理していく
タイルのサイズは並列度とデータの使いまわしのバランスをとって決める
並列度は高いほうがいいが、あるていどタイルが大きくないと使いまわす恩恵が小さい
shared memoryにデータを乗せ切れるという制約もある
smemの使い方 https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

```c++
__global__ void
calculate_forces(void *devX, void *devA)
{
  // ホストからfloat4*で受けたいから？ メモリはvoid*型
  extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  myPosition = globalX[gtid];
  for (i = 0, tile = 0; i < N; i += p, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  // Save the result in global memory for the integration step.
   float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;
}
```

## 31.3.4 Defining a grid of thread blocks

gridDimを決めましょうねという話

## 31.4 Performance results

### 31.4.1 Optimization

- ループアンローリング
- ブロックサイズp
- 小さい(CUDA Core数より少ない 4070なら5888)Nに対しては1粒子を複数スレッドで計算する NxNますに縦向きにも切り込みを入れる
