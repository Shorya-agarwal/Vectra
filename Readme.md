# Vectra: High-Performance SIMD Vector Search Engine

![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg) ![Optimizations](https://img.shields.io/badge/Optimizations-AVX2%20%7C%20OpenMP-red.svg) ![Status](https://img.shields.io/badge/Status-Performance%20Critical-green.svg)

**Vectra** is a header-only, bare-metal vector search engine built from scratch to demonstrate **hardware-sympathetic software design**. It implements both brute-force exact search utilizing **AVX2 intrinsics** and an approximate **HNSW (Hierarchical Navigable Small World)** graph index.

This project was engineered to explore the crossover point where algorithmic complexity ($O(\log N)$) overcomes hardware cache locality benefits ($O(N)$ linear scan).

## 🚀 Key Performance Features

* **SIMD Acceleration:** Hand-written AVX2 intrinsics (`_mm256_fmadd_ps`) process 8 floating-point dimensions per CPU cycle, achieving a **~6x speedup** over scalar execution.
* **Memory Layout:** Utilizes flat, contiguous memory buffers to maximize L1/L2 cache hit rates during linear scans.
* **Algorithmic Optimization:** Implements a custom HNSW graph from first principles for sub-linear search complexity on massive datasets.
* **Parallelism:** OpenMP multi-threading integration for throughput-oriented batch processing.

## 📊 Benchmark Results

*Hardware Environment: x86_64, Single-Socket Consumer CPU*
*Dataset: 10,000 Vectors @ 1024 Dimensions (FP32)*

| Implementation | Latency (Batch 100) | Speedup vs Baseline | Analysis |
| :--- | :--- | :--- | :--- |
| **Scalar (Baseline)** | `2108ms` | 1.0x | Bottlenecked by scalar ALU throughput. |
| **SIMD (AVX2)** | `380ms` | **5.5x** | **Fastest.** Dataset fits in L3 cache; linear scan dominates due to prefetching. |
| **SIMD + OpenMP** | `547ms` | 3.8x | Thread forking overhead outweighed compute gains for $N=10k$. |
| **HNSW Graph** | `669ms` | 3.1x | Slower due to random memory access patterns (pointer chasing) causing cache misses at low $N$. |

### 🧠 Engineering Insight
For small datasets ($N < 100k$), **Cache Locality is King**. 
While HNSW provides superior asymptotic complexity ($O(\log N)$), the constant factors related to random heap access degrade performance compared to a highly optimized AVX2 linear scan which fully saturates the CPU's memory bandwidth. Vectra demonstrates that **Brute Force is viable** when optimized for the hardware's vector units.

## 🛠 Build & Run

**Requirements:**
* GCC/Clang with C++17 support
* CPU supporting AVX2 (Haswell or newer)

**Compilation:**
```bash
# The -O3 and -mavx2 flags are non-negotiable for performance
g++ -O3 -mavx2 -fopenmp main.cpp -o vectra
```
**Execution:**
```Bash

./vectra
```
## 📂 Architecture
### **1. The SIMD Kernel**
The core distance metric (Dot Product) bypasses the standard library to communicate directly with CPU vector registers.
```C++

// 8-way parallel float multiplication
__m256 sum_vec = _mm256_setzero_ps();
for (int i = 0; i < dim; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
}
```
### **2. HNSW Graph Index**
Implements a multi-layered skip-list structure where:

* Layer 0: Contains all nodes (Ground Truth).
* Layer N: Sparse "Express Lanes" for fast traversal.
* Search Strategy: Greedy traversal minimizing distance to target at each layer before descending.

## 🔮 Future Roadmap
* Quantization: Implement Product Quantization (PQ) to compress vectors from FP32 to uint8 for 4x memory bandwidth reduction.
* Prefetching: Add _mm_prefetch instructions to HNSW traversal to hide memory latency during graph hops.
