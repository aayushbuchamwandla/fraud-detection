# Benchmark summary (auto-generated from benchmarks/results/*.json -- do not hand-edit)

Mean latency (ms) by batch size:

| Implementation | batch=1 | batch=32 | batch=128 | batch=512 | batch=1024 |
|---|---:|---:|---:|---:|---:|
| CPU (PyTorch) | 0.070 ms | 0.075 ms | 0.090 ms | 0.138 ms | 0.161 ms |
| PyTorch GPU | 0.239 ms | 0.234 ms | 0.237 ms | 0.238 ms | 0.274 ms |
| Custom CUDA kernel | 0.091 ms | 0.091 ms | 0.101 ms | 0.110 ms | 0.118 ms |
| C++ (persistent buffers) | 0.094 ms | 0.082 ms | 0.088 ms | 0.097 ms | 0.129 ms |
| TensorRT | 0.159 ms | 0.174 ms | 0.168 ms | 0.157 ms | 0.185 ms |

Throughput (samples/sec) by batch size:

| Implementation | batch=1 | batch=32 | batch=128 | batch=512 | batch=1024 |
|---|---:|---:|---:|---:|---:|
| CPU (PyTorch) | 14,268 | 429,023 | 1,425,691 | 3,711,180 | 6,363,845 |
| PyTorch GPU | 4,189 | 136,912 | 541,028 | 2,155,136 | 3,737,745 |
| Custom CUDA kernel | 10,955 | 352,125 | 1,269,773 | 4,666,444 | 8,676,986 |
| C++ (persistent buffers) | 10,676 | 392,078 | 1,449,350 | 5,267,420 | 7,913,940 |
| TensorRT | 6,288 | 184,067 | 763,651 | 3,269,962 | 5,525,190 |
