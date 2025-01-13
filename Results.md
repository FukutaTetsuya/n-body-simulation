## CPU

### 情報

- intel core i7 147000F
- コンパイルオプション: `nvcc main.cu -lcurand -O2 -Xcompiler -Wall`
  - ホストのc++コンパイラ: nvccのデフォルト(gcc)

### 計測結果

- `std::chrono::system_clock::now()`でざっくり計測
- 10000ステップ * 3回計測
- 座標等のダンプは無し

| N | 1step time[ms] |
| -- | ------- |
| 256 | |
| 512 | |
| 1024 | |
| 2048 | |
| 4096 | |
