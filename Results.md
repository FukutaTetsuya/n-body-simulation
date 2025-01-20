## シミュレーション対象

重力で相互作用する天体を模した、N体シミュレーションを作成した。
ポテンシャルは本来`$\propto 1/r$`だが、発散を避けるためPlummer model `$\propto 1/\sqrt (r^2+\epsilon)$`を利用している。
相互作用のアルゴリズムは粒子-粒子法を採用し、すべての粒子ペア<i,j>に対して力f_ij,f_jiを計算する。`$O(N^2)$`の計算である。
(粒子-メッシュ法も作りたかったが時間切れ...)

## CPU

### 計測条件

- intel core i7 147000F
- コンパイルオプション: `nvcc main.cu -lcurand -O2 -Xcompiler -Wall`
  - ホストのc++コンパイラ: nvccのデフォルト(gcc)
- `std::chrono::system_clock::now()`でざっくり計測
- 10000ステップ * 3回計測
- 座標等のダンプは無し

### 計測結果

わかりやすく`$O(N^2)$`が見えている。

| N | time[ms] |
| -- | ------- |
| 1024 | 13759,14981,13797 |
| 2048 | 58432,59441,59443 |
| 4096 | 234463,234754,234985 |
| 8192 | |
| 16384 | |
| 32768 | |
## GPU 最適化なし

CPUコードをそのままcudaに置き換え、ループの最外周のみスレッド並列化した。`get_force<<<N,1>>>()`のような感じで、粒子数回のループを粒子数分のスレッドを立ち上げての計算に変えた。
データの格納順序は`x_1, .., x_N, y_1, .., y_N, z_1, .., z_N`のようになっている。

### 情報

- intel core i7 147000F
- nvidia gforce rtx 4070
- コンパイルオプション: `nvcc main.cu -lcurand -O2 -Xcompiler -Wall`
  - ホストのc++コンパイラ: nvccのデフォルト(gcc)
- `std::chrono::system_clock::now()`でざっくり計測
- 10000ステップ * 3回計測
- 座標等のダンプは無し

### 計測結果

なんの工夫もしていないがCPUより速い。やはり`$O(N^2)$`が見えている。

| N | time[ms] |
| -- | ------- |
| 256 | 414,386,386 |
| 512 | 767,749,750 |
| 1024 | 2095,2086,2057 |
| 2048 | 7522,7554,7545 |
| 4096 | 26600,26724,26794 |

## GPU最適化 自己流

除算を減らす
```c++
//const float dr_three_two = 1.0 / (dr_square * sqrtf(dr_square));
const float inv_dr_three_two = 1.0 / (dr_square * sqrtf(dr_square));
//float dUdx = mass_j * xij / dr_three_two;
float dUdx = mass_j * xij * inv_dr_three_two;
// ...
``` 

| N | time[ms] |
| -- | ------- |
| 2048 | 4762,4827,4803 |

## GPU最適化 GPU Gems

### float4を使う準備としてメモリを余分に確保

```c++
cudaMalloc((void **)(&r), 4 * N * sizeof(float));
//...
const float x = r[index*4];
const float y = r[index*4 + 1];
const float z = r[index*4 + 2];
// etc.
```
さすがに遅くなるが猛烈にというわけではない

| N | time[ms] |
| -- | ------- |
| 2048 | 4936,4943,5035 |

### タイルを使う準備として力の計算にthread blockを利用

```c++
__host__ void evolve_single_step(void) {
  //...
  //Kernels::update_accelaration<<<N,1,0,0>>>(N, softening_epsilon, r, mass, a);
  const dim3 grid_dim = dim3((N + particle_per_block - 1) / particle_per_block, 1, 1);
  const dim3 block_dim = dim3(particle_per_block, 1, 1);
  Kernels::update_accelaration<<<grid_dim,block_dim,0,0>>>(N, softening_epsilon, r, mass, a);
  //...
}

__global__ void update_accelaration(const int N, const float softening_epsilon, float* r, float* mass, float* a) {
  //const int index = blockIdx.x;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  //...
}
```

これだけで速くなるのはどういう理屈だったっけ

| N | time[ms] |
| -- | ------- |
| 1024 | 1493,1460,1467 |
| 2048 | 2764,2732,2746 |
| 4096 | 5624,4441,5620 |
| 8192 | 11615,10490,11707 |
| 16384 | 24347,23213,23224 |
| 32768 | 91242,91239,91282 |

### smemを活用した高速化

| N | time[ms] |
| -- | ------- |
| 1024 | 965,955,933 |
| 2048 | 1775,1778,1824 |
| 4096 | 3515,3495,3505 |
| 8192 | 7593,7607,7652 |
| 16384 | 18413,18346,19517 |
| 32768 | 73663,73062,72012 |

## 速度Verlet法の安定性
N = 16384, dt = 0.0001, total_steps = 10000 (total_t ==  10), すべてfloat

| t | energy, center of mass(x,y,z) |
| -- | ----- |
| 0 | -9894197., -1034.191040,-1491.128540,1320.212646 |
| 1000 | -9894293., -1034.276123,-1491.170410,1320.281494 |
| 2000 | -9894523., -1034.371460,-1491.197388,1320.404053 |
| 3000 | -9895005., -1034.321655,-1491.230957,1320.432617 |
| 4000 | -9895768., -1034.341797,-1491.241211,1320.421143 |
| 5000 | -9899049., -1034.346313,-1491.243530,1320.421753 |
| 6000 | -10257775., -1034.346680,-1491.243042,1320.422729 |
| 7000 | -10410358., -1034.346924,-1491.246216,1320.414429 |
| 8000 | -10525942., -1034.368042,-1491.249878,1320.384766 |
| 9000 | -10644761., -1034.307983,-1491.218872,1320.456299 |
| 9999 | -10759201., -1034.352051,-1491.265381,1320.464844 |
