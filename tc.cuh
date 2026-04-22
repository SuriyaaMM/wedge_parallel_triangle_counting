#pragma once
#include "core.cuh"
#include "definitions.cuh"
#include "search.cuh"
#include "utils.cuh"

__global__ void tc_GPU_kernel(const uint32_t *g_Ap, const uint32_t *g_Ai,
                              const uint64_t *g_wedgeSum,
                              const uint64_t wedgeSum_total,
                              const uint32_t num_vertices,
                              uint64_t *g_total_count, const uint32_t spread,
                              const uint32_t *g_adjacency_matrix,
                              const uint32_t adjacency_matrix_len,
                              const uint64_t adjacency_matrix_size) {

    const uint64_t tid =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    extern __shared__ uint32_t sdata[];
    // Size: blockDim.x X spread
    uint32_t *shared_src = sdata;
    // Size: blockDim.x X spread
    uint32_t *shared_dst = &sdata[blockDim.x * spread];
    // Size: 1
    uint32_t *shared_count = &sdata[2 * blockDim.x * spread];

    uint32_t v = 0;
    uint32_t w = 0;
    uint32_t u = 0;
    uint32_t vb = 0;     // Start index of adj(v)
    uint32_t ve = 0;     // End index of adj(v)
    uint32_t d_v = 0;    // Degree of v
    uint32_t w_i = 0;    // Index of w in adj(v)
    uint32_t u_i = 0;    // Index of u in adj(v)
    uint32_t wedges = 0; // Number of wedges of v
    uint32_t i_v = 0;    // Index of current wedge in wedges(v)

    const uint64_t i_start = tid * spread;
    const uint64_t i_end = min2(i_start + spread, wedgeSum_total);

    uint32_t s_i = threadIdx.x * spread;

    for (uint64_t i = i_start; i < i_end; i++, i_v++, s_i++) {

        // =----- FIRST WEDGE -----=
        if (i == i_start) {

#if BINSEARCH_CONSTANT
            v = binary_search_closest_ULONG_constant_GPU(g_wedgeSum, 0,
                                                         num_vertices, i_start);
#else
            v = binary_search_closest_ULONG_GPU(g_wedgeSum, 0, num_vertices,
                                                i_start);
#endif
            // start index in csr
            vb = g_Ap[v];
            // end index in csr
            ve = g_Ap[v + 1];
            // degree
            d_v = ve - vb;
            // (d(v) * (d(v) - 1)) // 2
            wedges = (d_v * (d_v - 1)) >> 1;
            // local index
            i_v = i_start - g_wedgeSum[v];

            // convert to upper triangular index using triangle formula
            w_i = d_v - 2 -
                  (uint32_t)(sqrt((double(wedges - i_v) - 0.875) * 2) - 0.5);
            u_i = i_v + w_i + 1 - wedges +
                  (((d_v - w_i) * ((d_v - w_i) - 1)) >> 1);

            w = g_Ai[vb + w_i];

        } else if (i_v >= wedges) {
            /*
            we have exhausted all the wedges in previous vertex v`
            now the next vertex is always going to be v + 1 because of
            how we spread it

            unless it is an 2 degree vertex, it is going to be v + 1
            why? skips binary search + sqrt operation
            */
            do {
                v++;
                vb = ve;
                ve = g_Ap[v + 1];
                d_v = ve - vb;
            } while (d_v < 2);

            wedges = (d_v * (d_v - 1)) >> 1;
            i_v = 0;
            w_i = 0;
            u_i = 1;

            w = g_Ai[vb];
        } else {
            /*
            else we're just processing the next wedge
            simple logic
            */
            u_i++;

            if (u_i >= d_v) {
                w_i++;
                u_i = w_i + 1;
                w = g_Ai[vb + w_i];
            }
        }

        shared_src[s_i] = w;
        shared_dst[s_i] = g_Ai[vb + u_i];
    }

    if (threadIdx.x == 0)
        *shared_count = 0;

    __syncthreads();

    /* Index into the shared 'transposed' matrix (spread X blockDim.x) */
    for (s_i = threadIdx.x; s_i < (blockDim.x * spread); s_i += blockDim.x) {
        /* Check bounds. */
#if CHECK_BOUNDS
        if (s_i >=
            (wedgeSum_total - (((uint64_t)blockIdx.x * blockDim.x) * spread)))
            break;
#endif

        w = shared_src[s_i];
        u = shared_dst[s_i];

        if (w >=
            (max2(num_vertices, adjacency_matrix_len) - adjacency_matrix_len)) {
            uint64_t adjacency_i = (adjacency_matrix_size -
                                    (((uint64_t)(num_vertices - w) *
                                      (uint64_t)((num_vertices - w) - 1)) >>
                                     1)) +
                                   u - w - 1;

#if UINT_WIDTH == 32
            bool found = (g_adjacency_matrix[adjacency_i >> 5] &
                          (1 << (adjacency_i & 31))) > 0;
#else
            bool found = (g_adjacency_matrix[adjacency_i / UINT_WIDTH] &
                          (1 << (adjacency_i % UINT_WIDTH))) > 0;
#endif
            if (found) {
                atomicAdd_block(shared_count, 1);
            }
        } else {
            uint32_t wb = g_Ap[w];
            uint32_t we = g_Ap[w + 1];

            if (we - wb < 2) {
                if (linear_search_GPU(g_Ai, wb, we, u) >= 0) {
                    atomicAdd_block(shared_count, 1);
                }
            } else {
                if (binary_search_GPU(g_Ai, wb, we, u) >= 0) {
                    atomicAdd_block(shared_count, 1);
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd((unsigned long long int *)g_total_count, *shared_count);
}

__global__ void populateWedgeSums(uint64_t *wedgeSum, const uint32_t *d_Ap,
                                  uint32_t numVertices) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {
        uint64_t dv = d_Ap[tid + 1] - d_Ap[tid];
        if (dv < 2) {
            wedgeSum[tid] = 0;
        } else {
            wedgeSum[tid] = (dv * (dv - 1)) / 2;
        }
    }

    else if (tid == numVertices) {
        wedgeSum[tid] = 0;
    }
}

uint64_t tc_GPU(const graph_t *graph, uint32_t spread,
                uint32_t adjacency_matrix_len, GPU_time *t) {
    uint32_t *d_Ap;
    uint32_t *d_Ai;
    uint64_t *d_wedgeSum;
    uint32_t *d_adjacency_matrix;
    uint64_t *d_total_count;

    cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
    float GPU_copy_elapsed, GPU_exec_elapsed;
    checkCudaErrors(cudaEventCreate(&GPU_copy_start));
    checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
    checkCudaErrors(cudaEventCreate(&GPU_exec_start));
    checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

    /*
    =----- Adjacency Matrix -----=
    - for any vertex v, after the relabelling operation, the number of wedges,
      for lower numbered vertices the number of outgoing edges is going
      to be higher, but for higher numbered vertices the number of incoming
      edges is going to be higher, thus instead of binary search on those higher
      numbered vertices, we'll store an bitmap adjacency matrix to make the
      lookup O(1)

    - length of adjacency matrix is predefined
    */
    uint64_t adjacency_matrix_size = (((uint64_t)adjacency_matrix_len) *
                                      ((uint64_t)adjacency_matrix_len - 1)) /
                                     2;
    uint32_t *h_adjacency_matrix = (uint32_t *)calloc(
        adjacency_matrix_size / UINT_WIDTH, sizeof(uint32_t));
    assert_malloc(h_adjacency_matrix);

    uint32_t v_begin =
        graph->numVertices - min2(graph->numVertices, adjacency_matrix_len);

    for (uint32_t v = v_begin; v < graph->numVertices; v++) {
        for (uint32_t i = graph->rowPtr[v]; i < graph->rowPtr[v + 1]; i++) {

            uint32_t w = graph->colInd[i];
            uint64_t adjacency_i =
                (adjacency_matrix_size -
                 (((uint64_t)(graph->numVertices - v) *
                   (uint64_t)((graph->numVertices - v) - 1)) /
                  2)) +
                w - v - 1;

#if UINT_WIDTH == 32
            h_adjacency_matrix[adjacency_i >> 5] |= (1 << (adjacency_i & 31));
#else
            h_adjacency_matrix[adjacency_i / UINT_WIDTH] |=
                (1 << (adjacency_i % UINT_WIDTH));
#endif
        }
    }

    checkCudaErrors(cudaEventRecord(GPU_copy_start));

    checkCudaErrors(cudaMalloc((void **)&d_Ap,
                               (graph->numVertices + 1) * sizeof(uint32_t)));
    checkCudaErrors(
        cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **)&d_wedgeSum,
                               (graph->numVertices + 1) * sizeof(uint64_t)));
    checkCudaErrors(
        cudaMalloc((void **)&d_adjacency_matrix,
                   (adjacency_matrix_size / 32) * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **)&d_total_count, 1 * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr,
                               (graph->numVertices + 1) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd,
                               graph->numEdges * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

    // =----- WEDGE SUM -----=
    uint64_t *h_wedgeSum =
        (uint64_t *)malloc((graph->numVertices + 1) * sizeof(uint64_t));
    assert_malloc(h_wedgeSum);
    h_wedgeSum[0] = 0;

    for (uint32_t v = 0; v < graph->numVertices; v++) {
        uint32_t d_v = graph->rowPtr[(v + 1)] - graph->rowPtr[v];
        if (d_v < 2) {
            h_wedgeSum[v + 1] = h_wedgeSum[v];
        } else {
            h_wedgeSum[v + 1] = h_wedgeSum[v] + ((d_v * (d_v - 1)) / 2);
        }
    }

    uint64_t wedgeSum_total = h_wedgeSum[graph->numVertices];

#if BINSEARCH_CONSTANT
    uint64_t *h_wedgeSum_cache =
        (uint64_t *)malloc(BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(uint64_t));
    assert_malloc(h_wedgeSum_cache);
    build_binary_search_cache(h_wedgeSum, h_wedgeSum_cache, 0,
                              BINSEARCH_CONSTANT_LEVELS, 0, 0,
                              graph->numVertices);
#endif

    checkCudaErrors(cudaMemcpy(d_wedgeSum, h_wedgeSum,
                               (graph->numVertices + 1) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix,
                               (adjacency_matrix_size / 32) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

#if BINSEARCH_CONSTANT
    checkCudaErrors(
        cudaMemcpyToSymbol(c_binary_search_cache, h_wedgeSum_cache,
                           BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(uint64_t)));
#endif

    checkCudaErrors(cudaMemset(d_total_count, 0, 1 * sizeof(uint64_t)));

    checkCudaErrors(cudaEventRecord(GPU_copy_stop));
    checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
    checkCudaErrors(
        cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
    t->copy += GPU_copy_elapsed;

    uint32_t num_threads = 128;
    uint64_t num_blocks = (wedgeSum_total / (spread * num_threads)) + 1;

    if (num_blocks > (((uint64_t)1 << 31) - 1)) {
        fprintf(stderr, "ERROR: maximum grid size reached.\n");
        exit(EXIT_FAILURE);
    }

    dim3 grid(num_blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    int smem_size = (2 * num_threads * spread + 1) * sizeof(uint32_t);

    checkCudaErrors(cudaEventRecord(GPU_exec_start));

    tc_GPU_kernel<<<grid, threads, smem_size>>>(
        d_Ap, d_Ai, d_wedgeSum, wedgeSum_total, graph->numVertices,
        d_total_count, spread, d_adjacency_matrix, adjacency_matrix_len,
        adjacency_matrix_size);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(GPU_exec_stop));
    checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
    checkCudaErrors(
        cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
    t->exec += GPU_exec_elapsed;

    uint64_t h_total_count = 0;
    checkCudaErrors(cudaMemcpy(&h_total_count, d_total_count,
                               1 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_Ap));
    checkCudaErrors(cudaFree(d_Ai));
    checkCudaErrors(cudaFree(d_wedgeSum));
    checkCudaErrors(cudaFree(d_adjacency_matrix));
    checkCudaErrors(cudaFree(d_total_count));

    checkCudaErrors(cudaEventDestroy(GPU_copy_start));
    checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
    checkCudaErrors(cudaEventDestroy(GPU_exec_start));
    checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

    free(h_wedgeSum);
    free(h_adjacency_matrix);

#if BINSEARCH_CONSTANT
    free(h_wedgeSum_cache);
#endif

#if RESET_DEVICE
    checkCudaErrors(cudaDeviceReset());
#endif
    return h_total_count;
}

uint64_t tc_GPU2(const graph_t *graph, uint32_t spread,
                 uint32_t adjacency_matrix_len, GPU_time *t) {
    uint32_t *d_Ap = graph->d_Ap;
    uint32_t *d_Ai = graph->d_Ai;
    uint64_t *d_wedgeSum;
    uint64_t *d_wedgeSumOut;
    uint32_t *d_adjacency_matrix;
    uint64_t *d_total_count;

    cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
    float GPU_copy_elapsed, GPU_exec_elapsed;
    checkCudaErrors(cudaEventCreate(&GPU_copy_start));
    checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
    checkCudaErrors(cudaEventCreate(&GPU_exec_start));
    checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

    /*
    =----- Adjacency Matrix -----=
    - for any vertex v, after the relabelling operation, the number of wedges,
      for lower numbered vertices the number of outgoing edges is going
      to be higher, but for higher numbered vertices the number of incoming
      edges is going to be higher, thus instead of binary search on those higher
      numbered vertices, we'll store an bitmap adjacency matrix to make the
      lookup O(1)

    - length of adjacency matrix is predefined
    */
    uint64_t adjacency_matrix_size = (((uint64_t)adjacency_matrix_len) *
                                      ((uint64_t)adjacency_matrix_len - 1)) /
                                     2;
    uint32_t *h_adjacency_matrix = (uint32_t *)calloc(
        adjacency_matrix_size / UINT_WIDTH, sizeof(uint32_t));
    assert_malloc(h_adjacency_matrix);

    uint32_t v_begin =
        graph->numVertices - min2(graph->numVertices, adjacency_matrix_len);

    for (uint32_t v = v_begin; v < graph->numVertices; v++) {
        for (uint32_t i = graph->rowPtr[v]; i < graph->rowPtr[v + 1]; i++) {

            uint32_t w = graph->colInd[i];
            uint64_t adjacency_i =
                (adjacency_matrix_size -
                 (((uint64_t)(graph->numVertices - v) *
                   (uint64_t)((graph->numVertices - v) - 1)) /
                  2)) +
                w - v - 1;

#if UINT_WIDTH == 32
            h_adjacency_matrix[adjacency_i >> 5] |= (1 << (adjacency_i & 31));
#else
            h_adjacency_matrix[adjacency_i / UINT_WIDTH] |=
                (1 << (adjacency_i % UINT_WIDTH));
#endif
        }
    }

    checkCudaErrors(cudaEventRecord(GPU_copy_start));
    checkCudaErrors(cudaMalloc((void **)&d_wedgeSum,
                               (graph->numVertices + 1) * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **)&d_wedgeSumOut,
                               (graph->numVertices + 1) * sizeof(uint64_t)));
    checkCudaErrors(
        cudaMalloc((void **)&d_adjacency_matrix,
                   (adjacency_matrix_size / 32) * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **)&d_total_count, 1 * sizeof(uint64_t)));

    // =----- perform wedge sum on gpu -----=

    dim3 blockDim(1024);
    dim3 gridDim((graph->numVertices + 1 + 1023) / 1024);

    uint32_t numVertices = graph->numVertices;
    void *args[] = {&d_wedgeSum, &d_Ap, &numVertices};

    // fills the wedge sums arrays
    cudaLaunchKernel(populateWedgeSums, gridDim, blockDim, args);
    cudaDeviceSynchronize();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_wedgeSum, d_wedgeSumOut,
                                  graph->numVertices + 1);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_wedgeSum, d_wedgeSumOut,
                                  graph->numVertices + 1);

    uint64_t wedgeSum_total;
    checkCudaErrors(cudaMemcpy(&wedgeSum_total,
                               d_wedgeSumOut + graph->numVertices,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // printf("wedge sum total = %lu", wedgeSum_total);

#if BINSEARCH_CONSTANT
    uint64_t *h_wedgeSum_cache =
        (uint64_t *)malloc(BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(uint64_t));
    assert_malloc(h_wedgeSum_cache);
    build_binary_search_cache(h_wedgeSum, h_wedgeSum_cache, 0,
                              BINSEARCH_CONSTANT_LEVELS, 0, 0,
                              graph->numVertices);
#endif

    checkCudaErrors(cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix,
                               (adjacency_matrix_size / 32) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

#if BINSEARCH_CONSTANT
    checkCudaErrors(
        cudaMemcpyToSymbol(c_binary_search_cache, h_wedgeSum_cache,
                           BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(uint64_t)));
#endif

    checkCudaErrors(cudaMemset(d_total_count, 0, 1 * sizeof(uint64_t)));

    checkCudaErrors(cudaEventRecord(GPU_copy_stop));
    checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
    checkCudaErrors(
        cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
    t->copy += GPU_copy_elapsed;

    uint32_t num_threads = 128;
    uint64_t num_blocks = (wedgeSum_total / (spread * num_threads)) + 1;

    if (num_blocks > (((uint64_t)1 << 31) - 1)) {
        fprintf(stderr, "ERROR: maximum grid size reached.\n");
        exit(EXIT_FAILURE);
    }

    dim3 grid(num_blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    int smem_size = (2 * num_threads * spread + 1) * sizeof(uint32_t);

    checkCudaErrors(cudaEventRecord(GPU_exec_start));

    tc_GPU_kernel<<<grid, threads, smem_size>>>(
        d_Ap, d_Ai, d_wedgeSumOut, wedgeSum_total, graph->numVertices,
        d_total_count, spread, d_adjacency_matrix, adjacency_matrix_len,
        adjacency_matrix_size);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(GPU_exec_stop));
    checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
    checkCudaErrors(
        cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
    t->exec += GPU_exec_elapsed;

    uint64_t h_total_count = 0;
    checkCudaErrors(cudaMemcpy(&h_total_count, d_total_count,
                               1 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_wedgeSum));
    checkCudaErrors(cudaFree(d_adjacency_matrix));
    checkCudaErrors(cudaFree(d_total_count));
    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaFree(d_wedgeSumOut));

    checkCudaErrors(cudaEventDestroy(GPU_copy_start));
    checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
    checkCudaErrors(cudaEventDestroy(GPU_exec_start));
    checkCudaErrors(cudaEventDestroy(GPU_exec_stop));
    free(h_adjacency_matrix);

#if BINSEARCH_CONSTANT
    free(h_wedgeSum_cache);
#endif

#if RESET_DEVICE
    checkCudaErrors(cudaDeviceReset());
#endif
    return h_total_count;
}