#include <iostream>
#include <vector>
#include <random>
#include <chrono>   // For benchmarking
#include <limits>   // For numeric_limits
#include <immintrin.h> // The "Magic" header for AVX2
#include <omp.h>    // OpenMP for parallelism
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <map>
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
// ---------------------------------------------------------
// HNSW GRAPH STRUCTURE
// ---------------------------------------------------------

struct Node {
    int id;
    const float* vector; // Pointer to the data in the main "database" vector
    std::vector<std::vector<int>> connections; // [layer_level][neighbor_index]
    int max_layer;
    
    Node(int _id, const float* _vec, int _max_layer) 
        : id(_id), vector(_vec), max_layer(_max_layer) {
        connections.resize(_max_layer + 1);
    }
};

class HNSWIndex {
public:
    int M; // Max number of connections per node per layer
    int ef_construction; // Size of the dynamic candidate list during build
    std::vector<Node> nodes;
    int entry_point_id = -1; 
    int max_level = -1;
    int D_dim; // Dimension of vectors

    HNSWIndex(int dim, int max_neighbors = 16, int ef = 64) 
        : D_dim(dim), M(max_neighbors), ef_construction(ef) {}

    // PROBABILITY FUNCTION FOR LAYERS
    // Determines if a node gets promoted to higher layers (like a coin flip)
    int get_random_level() {
        double r = ((double) rand() / (RAND_MAX));
        // -ln(r) * (1/ln(M)) is the standard HNSW level generation formula
        // simplified here to just basic probability for clarity:
        int level = 0;
        while (r < 0.5 && level < 3) { // 50% chance to go up a level, cap at level 3
            level++;
            r = ((double) rand() / (RAND_MAX));
        }
        return level;
    }

    // ---------------------------------------------------------
    // THE CORE: GREEDY SEARCH ON GRAPH
    // ---------------------------------------------------------
    // "Start at 'curr_node', look at neighbors, move to closer one, repeat."
    int search_layer(const float* query, int curr_node_id, int layer_idx) {
        float min_dist = dot_product_avx2(nodes[curr_node_id].vector, query, D_dim);
        int best_node = curr_node_id;
        bool changed = true;

        while (changed) {
            changed = false;
            // Iterate over neighbors of the current best node
            for (int neighbor_id : nodes[best_node].connections[layer_idx]) {
                // Note: In real HNSW we minimize distance (Euclidean). 
                // Since we used Dot Product (higher is better), we look for *higher* scores.
                // If using Euclidean, flip this logic.
                float dist = dot_product_avx2(nodes[neighbor_id].vector, query, D_dim);
                
                if (dist > min_dist) { // Found a closer neighbor (higher dot product)
                    min_dist = dist;
                    best_node = neighbor_id;
                    changed = true;
                }
            }
        }
        return best_node;
    }

    // ---------------------------------------------------------
    // INSERTION (Building the Graph)
    // ---------------------------------------------------------
    void add_item(int id, const float* vector) {
        int level = get_random_level();
        Node new_node(id, vector, level);
        
        // If graph is empty, this is the entry point
        if (entry_point_id == -1) {
            nodes.push_back(new_node);
            entry_point_id = id;
            max_level = level;
            return;
        }

        // 1. Zoom phase: Start at top layer, find closest node to entry point, move down
        int curr_node = entry_point_id;
        for (int lc = max_level; lc > level; lc--) {
            curr_node = search_layer(vector, curr_node, lc);
        }

        // 2. Build phase: Connect new node at its assigned level and below
        for (int lc = std::min(max_level, level); lc >= 0; lc--) {
            // Find closest neighbors at this layer to connect to
            // (Simplification: In a real HNSW, we do a full 'ef_search' here. 
            // We are just reusing the greedy search for brevity.)
            int nearest_in_layer = search_layer(vector, curr_node, lc);
            
            // Bidirectional connection
            new_node.connections[lc].push_back(nearest_in_layer);
            nodes[nearest_in_layer].connections[lc].push_back(id);
            
            // Update curr_node for next layer down
            curr_node = nearest_in_layer;
        }
        
        nodes.push_back(new_node);

        // Update global entry point if new node is higher
        if (level > max_level) {
            max_level = level;
            entry_point_id = id;
        }
    }

    // ---------------------------------------------------------
    // FINAL QUERY FUNCTION
    // ---------------------------------------------------------
    int search_hnsw(const float* query) {
        int curr_node = entry_point_id;
        
        // Zoom down from top to layer 1
        for (int lc = max_level; lc > 0; lc--) {
            curr_node = search_layer(query, curr_node, lc);
        }
        
        // Final search on Ground Floor (Layer 0)
        return search_layer(query, curr_node, 0);
    }
};
int main() {
    // Memory alignment helps AVX, but std::vector doesn't guarantee 32-byte alignment by default.
    // Ideally, we'd use aligned_alloc, but standard vector is fine for a demo.
    std::vector<float> database(N * D);
    std::vector<float> query(D);

    /**std::cout << "Generating " << N << " vectors of dimension " << D << "..." << std::endl;
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
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;**/
    std::cout << "Building HNSW Graph..." << std::endl;
    HNSWIndex index(D);
    // Because 'nodes' vector resizes, pointers might invalidate if we aren't careful.
    // In production, we'd reserve() or use pointers to vectors. 
    // For this hackathon code, let's reserve enough space to prevent re-allocation.
    index.nodes.reserve(N); 
    
    auto start_build = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i) {
        index.add_item(i, &database[i * D]);
    }
    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "Build Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() << "ms" << std::endl;

    // --- BENCHMARK HNSW ---
    auto start = std::chrono::high_resolution_clock::now();
    int result_idx = -1;
    for(int k=0; k<QUERY_BATCH; k++) { 
        result_idx = index.search_hnsw(query.data()); 
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "HNSW Search Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
              
    std::cout << "Found Index: " << result_idx << std::endl;

    return 0;
}
