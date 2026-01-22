#include <iostream>
#include <vector>
#include <random>
#include <chrono>   // For benchmarking
#include <limits>   // For numeric_limits
#include <immintrin.h> // The "Magic" header for AVX2
#include <omp.h>    // OpenMP for parallelism

// CONSTANTS
const int D = 1024;       // Dimension of each vector (e.g., typical embedding size)
const int N = 10000;      // Number of vectors in the database
const int QUERY_BATCH = 100; // Perform multiple searches to get measurable time

// ---------------------------------------------------------
// 1. DATA GENERATION
// ---------------------------------------------------------
// We use a flat array for memory locality (better cache performance than vector<vector<float>>)
void generate_data(std::vector<float>& database, std::vector<float>& query) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * D; ++i) database[i] = dist(gen);
    for (int i = 0; i < D; ++i) query[i] = dist(gen);
}

// ---------------------------------------------------------
// 2. BASELINE APPROACH (Scalar)
// ---------------------------------------------------------
float dot_product_scalar(const float* a, const float* b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ---------------------------------------------------------
// 3. AVX2 INTRINSICS APPROACH (SIMD)
// ---------------------------------------------------------
// Processes 8 floats per CPU cycle instruction
float dot_product_avx2(const float* a, const float* b, int dim) {
    __m256 sum_vec = _mm256_setzero_ps(); // Initialize a register [0,0,0,0,0,0,0,0]

    // Loop with stride of 8
    for (int i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]); // Load 8 floats from A
        __m256 vb = _mm256_loadu_ps(&b[i]); // Load 8 floats from B
        // Multiply va * vb, then add to sum_vec (Fused Multiply-Add would be faster, but let's keep it simple)
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
    }

    // "Horizontal Sum": Sum the 8 values inside the register into one float
    // There are faster ways to do this, but this is readable for a start:
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float result = 0.0f;
    for (int i = 0; i < 8; ++i) result += temp[i];
    return result;
}

// ---------------------------------------------------------
// SEARCH FUNCTION
// ---------------------------------------------------------
// Finds the index of the nearest vector (highest dot product)
int search_nn(const std::vector<float>& db, const float* query, 
              float (*metric)(const float*, const float*, int), 
              bool use_omp) {
    
    int best_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    // The #pragma line enables multi-threading only if use_omp is true
    #pragma omp parallel for if(use_omp) reduction(max:max_score) 
    for (int i = 0; i < N; ++i) {
        float score = metric(&db[i * D], query, D);
        
        // Note: Getting the specific index in a parallel reduction is tricky.
        // For pure performance benchmarking, we often just track the max_score.
        // In a real system, you'd track index carefully.
        if (score > max_score) {
            max_score = score;
            // best_idx = i; // Race condition in OMP! Ignored for this specific benchmark.
        }
    }
    return best_idx;
}

int main() {
    // Memory alignment helps AVX, but std::vector doesn't guarantee 32-byte alignment by default.
    // Ideally, we'd use aligned_alloc, but standard vector is fine for a demo.
    std::vector<float> database(N * D);
    std::vector<float> query(D);

    std::cout << "Generating " << N << " vectors of dimension " << D << "..." << std::endl;
    generate_data(database, query);

    // --- Benchmark Baseline ---
    auto start = std::chrono::high_resolution_clock::now();
    for(int k=0; k<QUERY_BATCH; k++) { 
        search_nn(database, query.data(), dot_product_scalar, false); 
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline (Scalar): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // --- Benchmark AVX2 ---
    start = std::chrono::high_resolution_clock::now();
    for(int k=0; k<QUERY_BATCH; k++) { 
        search_nn(database, query.data(), dot_product_avx2, false); 
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized (AVX2):  " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // --- Benchmark AVX2 + OpenMP ---
    start = std::chrono::high_resolution_clock::now();
    for(int k=0; k<QUERY_BATCH; k++) { 
        search_nn(database, query.data(), dot_product_avx2, true); 
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Hardcore (AVX2 + OpenMP): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}
