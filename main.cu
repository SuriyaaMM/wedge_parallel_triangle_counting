#include "tc.cuh"

__global__ void split_edges_kernel(const edge_t *unique_edges, uint32_t *srcs,
                                   uint32_t *dsts, uint32_t num_edges) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        srcs[tid] = unique_edges[tid].src;
        dsts[tid] = unique_edges[tid].dst;
    }
}

void usage() {
    printf("Wedge Parallel Triangle Counting\n\n");
    printf("Usage:\n\n");
    printf("Either one of these must be selected:\n");
    printf(" -m <filename>	[Input graph in Matrix Market "
           "format]\n");
    printf(" -e <filename>	[Input graph in edge list format]\n");
    printf("Required arguments:\n");
    printf(" -s <num>		 	 	[Spread, a.k.a. "
           "wedges/thread]\n");
    printf(" -a <num>				[Adjacency "
           "matrix length] (must "
           "be divisble by 32)\n");
    printf("Optional arguments:\n");
    printf(" -l <num>				[Loop "
           "count]\n");
    printf(" -z						"
           "	[Input graph is "
           "zero-indexed]\n");
    printf(" -p						"
           "	[Preprocessing style, 0:CPU, "
           "1:GPU, 2:GPU low-memory (default)]\n");
    printf("\n");
    printf("Example:\n");
    printf("./tc -m Amazon0302.mtx -s 5 -a 8192 -l 10\n");
    exit(EXIT_FAILURE);
}

static int compareInt_t(const void *a, const void *b) {
    uint32_t arg1 = *(const uint32_t *)a;
    uint32_t arg2 = *(const uint32_t *)b;
    if (arg1 < arg2)
        return -1;
    if (arg1 > arg2)
        return 1;
    return 0;
}

static int compareEdge_t(const void *a, const void *b) {
    edge_t arg1 = *(const edge_t *)a;
    edge_t arg2 = *(const edge_t *)b;
    if (arg1.src < arg2.src)
        return -1;
    if (arg1.src > arg2.src)
        return 1;
    if ((arg1.src == arg2.src) && (arg1.dst < arg2.dst))
        return -1;
    if ((arg1.src == arg2.src) && (arg1.dst > arg2.dst))
        return 1;
    return 0;
}

static int compare_vertex_degree_ascending(const void *a, const void *b) {
    preprocess_vertex_t arg1 = *(const preprocess_vertex_t *)a;
    preprocess_vertex_t arg2 = *(const preprocess_vertex_t *)b;
    if (arg1.num_edges < arg2.num_edges)
        return -1;
    if (arg1.num_edges > arg2.num_edges)
        return 1;
    return 0;
}

struct edge_decomposer_t {
    __host__ __device__ ::cuda::std::tuple<unsigned int &, unsigned int &>
    operator()(edge_t &key) const {
        return {key.src, key.dst};
    }
};

struct preprocess_vertex_decomposer_t {
    __host__ __device__ ::cuda::std::tuple<unsigned int &>
    operator()(preprocess_vertex_t &key) const {
        return {key.num_edges};
    }
};

edge_t *sort_edges_GPU(edge_t *d_in, edge_t *d_out, const uint32_t num_edges,
                       bool use_double_buffer) {
    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    if (use_double_buffer) {
        cub::DoubleBuffer<edge_t> d_keys(d_in, d_out);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_edges, edge_decomposer_t{});
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_edges, edge_decomposer_t{});
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_keys.Current();
    } else {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in,
                                       d_out, num_edges, edge_decomposer_t{});
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in,
                                       d_out, num_edges, edge_decomposer_t{});
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_out;
    }
}

preprocess_vertex_t *sort_vertices_GPU(preprocess_vertex_t *d_in,
                                       preprocess_vertex_t *d_out,
                                       const uint32_t num_vertices,
                                       bool use_double_buffer) {
    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    if (use_double_buffer) {
        cub::DoubleBuffer<preprocess_vertex_t> d_keys(d_in, d_out);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_vertices,
                                       preprocess_vertex_decomposer_t{});
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_vertices,
                                       preprocess_vertex_decomposer_t{});
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_keys.Current();
    } else {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in,
                                       d_out, num_vertices,
                                       preprocess_vertex_decomposer_t{});
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in,
                                       d_out, num_vertices,
                                       preprocess_vertex_decomposer_t{});
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_out;
    }
}

uint32_t *sort_colInd_GPU(uint32_t *d_rowPtr, uint32_t *d_colInd_in,
                          uint32_t *d_colInd_out, const uint32_t num_vertices,
                          const uint32_t num_edges, bool use_double_buffer) {
    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    if (use_double_buffer) {
        cub::DoubleBuffer<uint32_t> d_keys(d_colInd_in, d_colInd_out);
        cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                           d_keys, num_edges, num_vertices,
                                           d_rowPtr, d_rowPtr + 1);
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                           d_keys, num_edges, num_vertices,
                                           d_rowPtr, d_rowPtr + 1);
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_keys.Current();
    } else {
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out,
            num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
        checkCudaErrors(cudaMalloc((void **)&d_temp_storage,
                                   temp_storage_bytes * sizeof(std::uint8_t)));
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out,
            num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
        checkCudaErrors(cudaFree(d_temp_storage));
        return d_colInd_out;
    }
}

graph_t *read_graph(char *filename, bool matrix_market, bool zero_indexed,
                    preprocess_t preprocess_style) {
    FILE *infile = fopen(filename, "r");
    if (infile == NULL) {
        fprintf(stderr, "ERROR: unable to open graph file.\n");
        usage();
    }

    graph_t *graph = (graph_t *)malloc(sizeof(graph_t));
    char line[256];

    /* Skip any header lines */
    do {
        if (fgets(line, sizeof(line), infile) == NULL)
            usage();
    } while (line[0] < '0' || line[0] > '9');

    /* Skip line if the file is in Matrix Market format. We do not use the given
     * vertex/edge counts. */
    if (matrix_market) {
        if (fgets(line, sizeof(line), infile) == NULL)
            usage();
    }

    uint32_t vertex_count = 0;
    uint32_t edge_count = 0;
    size_t size = 10240;
    edge_t *edges = (edge_t *)malloc(size * sizeof(edge_t));
    assert_malloc(edges);

    uint32_t max_vertex = 0;
    uint32_t v, w;

    if (sscanf(line, "%d %d\n", &v, &w) == 2) {
        do {
            if (edge_count >= size) {
                size += 10240;
                edge_t *new_edges =
                    (edge_t *)realloc(edges, size * sizeof(edge_t));
                assert_malloc(new_edges);
                edges = new_edges;
            }

            if ((!zero_indexed) && (v == 0 || w == 0)) {
                fprintf(stderr, "ERROR: zero vertex "
                                "id detected but "
                                "-z was not set.\n");
                usage();
            }

            v -= (zero_indexed ? 0 : 1);
            w -= (zero_indexed ? 0 : 1);

            /* Remove self-loops. */
            if (v != w) {
                max_vertex = max2(max_vertex, max2(v, w));

                /* v->w */
                edges[edge_count].src = v;
                edges[edge_count].dst = w;
                edge_count++;
                /* w->v */
                edges[edge_count].src = w;
                edges[edge_count].dst = v;
                edge_count++;
            }
        } while (fscanf(infile, "%d %d\n", &v, &w) == 2);
    }

    fclose(infile);

    vertex_count = max_vertex + 1;

    /* Sort edges (in order to remove duplicates). */
    if (preprocess_style != PREPROCESS_CPU) {
        edge_t *d_edges;
        edge_t *d_edges_alt;
        edge_t *d_out;

        checkCudaErrors(
            cudaMalloc((void **)&d_edges, edge_count * sizeof(edge_t)));
        checkCudaErrors(
            cudaMalloc((void **)&d_edges_alt, edge_count * sizeof(edge_t)));
        checkCudaErrors(cudaMemcpy(d_edges, edges, edge_count * sizeof(edge_t),
                                   cudaMemcpyHostToDevice));

        if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
            d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, true);
        else
            d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, false);

        checkCudaErrors(cudaMemcpy(edges, d_out, edge_count * sizeof(edge_t),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_edges));
        checkCudaErrors(cudaFree(d_edges_alt));
    } else {
        qsort(edges, edge_count, sizeof(edge_t), compareEdge_t);
    }

    uint32_t *rowPtr = (uint32_t *)calloc(vertex_count + 1, sizeof(uint32_t));
    assert_malloc(rowPtr);

    uint32_t edge_count_no_dup = 1;

    edge_t lastedge;
    lastedge.src = edges[0].src;
    lastedge.dst = edges[0].dst;

    uint32_t *colInd =
        (uint32_t *)edges; /* colInd overwrites the edges array. Possible
                              because sizeof(edge_t) > sizeof(uint32_t). */
    colInd[0] = lastedge.dst;
    rowPtr[lastedge.src + 1]++;

    /* Remove duplicate edges. */
    for (uint32_t i = 1; i < edge_count; i++) {
        if (compareEdge_t(&lastedge, &edges[i]) != 0) {
            colInd[edge_count_no_dup++] = edges[i].dst;
            rowPtr[edges[i].src + 1]++;
            lastedge.src = edges[i].src;
            lastedge.dst = edges[i].dst;
        }
    }

    /* Free excess memory from the colInd/edges array. */
    uint32_t *new_colInd =
        (uint32_t *)realloc(colInd, edge_count_no_dup * sizeof(uint32_t));

    for (uint32_t v = 1; v <= vertex_count; v++) {
        rowPtr[v] += rowPtr[v - 1];
    }

    graph->numVertices = vertex_count;
    graph->numEdges = edge_count_no_dup;
    graph->rowPtr = rowPtr;
    graph->colInd = new_colInd;

    return graph;
}

graph_t *read_graph2(char *filename, bool matrix_market, bool zero_indexed,
                     preprocess_t preprocess_style) {
    FILE *infile = fopen(filename, "r");
    if (infile == NULL) {
        fprintf(stderr, "ERROR: unable to open graph file.\n");
        usage();
    }

    graph_t *graph = (graph_t *)malloc(sizeof(graph_t));
    char line[256];

    /* Skip any header lines */
    do {
        if (fgets(line, sizeof(line), infile) == NULL)
            usage();
    } while (line[0] < '0' || line[0] > '9');

    /* Skip line if the file is in Matrix Market format. We do not use the given
     * vertex/edge counts. */
    if (matrix_market) {
        if (fgets(line, sizeof(line), infile) == NULL)
            usage();
    }

    uint32_t vertex_count = 0;
    uint32_t edge_count = 0;
    size_t size = 10240;
    edge_t *edges = (edge_t *)malloc(size * sizeof(edge_t));
    assert_malloc(edges);

    uint32_t max_vertex = 0;
    uint32_t v, w;

    if (sscanf(line, "%d %d\n", &v, &w) == 2) {
        do {
            if (edge_count >= size) {
                size += 10240;
                edge_t *new_edges =
                    (edge_t *)realloc(edges, size * sizeof(edge_t));
                assert_malloc(new_edges);
                edges = new_edges;
            }

            if ((!zero_indexed) && (v == 0 || w == 0)) {
                fprintf(stderr, "ERROR: zero vertex "
                                "id detected but "
                                "-z was not set.\n");
                usage();
            }

            v -= (zero_indexed ? 0 : 1);
            w -= (zero_indexed ? 0 : 1);

            /*
            prevents self loops
            */
            if (v != w) {
                max_vertex = max2(max_vertex, max2(v, w));

                edges[edge_count].src = min2(v, w);
                edges[edge_count].dst = max2(v, w);
                edge_count++;
            }
        } while (fscanf(infile, "%d %d\n", &v, &w) == 2);
    }

    fclose(infile);

    vertex_count = max_vertex + 1;

    // allocate device side variables
    edge_t *d_edges;
    checkCudaErrors(cudaMalloc((void **)&d_edges, edge_count * sizeof(edge_t)));
    checkCudaErrors(cudaMemcpy(d_edges, edges, edge_count * sizeof(edge_t),
                               cudaMemcpyHostToDevice));

    // sorted edges
    edge_t *d_edges_sorted;
    checkCudaErrors(
        cudaMalloc((void **)&d_edges_sorted, edge_count * sizeof(edge_t)));

    if (preprocess_style == PREPROCESS_GPU_CONSTRAINED) {
        sort_edges_GPU(d_edges, d_edges_sorted, edge_count, true);
    } else {
        sort_edges_GPU(d_edges, d_edges_sorted, edge_count, false);
    }
    checkCudaErrors(cudaFree(d_edges));

    // keep unique edges only
    edge_t *d_unique_edges;
    uint32_t *d_num_unique;
    checkCudaErrors(
        cudaMalloc((void **)&d_unique_edges, edge_count * sizeof(edge_t)));
    checkCudaErrors(cudaMalloc((void **)&d_num_unique, sizeof(uint32_t)));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              d_edges_sorted, d_unique_edges, d_num_unique,
                              edge_count);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                              d_edges_sorted, d_unique_edges, d_num_unique,
                              edge_count);
    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaFree(d_edges_sorted));

    uint32_t edge_count_no_dup = 0;
    checkCudaErrors(cudaMemcpy(&edge_count_no_dup, d_num_unique,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_num_unique));

    uint32_t *d_srcs;
    uint32_t *d_Ai;
    checkCudaErrors(
        cudaMalloc((void **)&d_srcs, edge_count_no_dup * sizeof(uint32_t)));
    checkCudaErrors(
        cudaMalloc((void **)&d_Ai, edge_count_no_dup * sizeof(uint32_t)));

    uint32_t threads = 1024;
    uint32_t blocks = (edge_count_no_dup + threads - 1) / threads;
    split_edges_kernel<<<blocks, threads>>>(d_unique_edges, d_srcs, d_Ai,
                                            edge_count_no_dup);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_unique_edges));

    // histogram to compute degrees
    uint32_t *d_degrees;
    checkCudaErrors(
        cudaMalloc((void **)&d_degrees, (vertex_count + 1) * sizeof(uint32_t)));
    checkCudaErrors(
        cudaMemset(d_degrees, 0, (vertex_count + 1) * sizeof(uint32_t)));

    d_temp_storage = NULL;
    temp_storage_bytes = 0;

    int num_levels = vertex_count + 1;
    uint32_t lower_level = 0;
    uint32_t upper_level = vertex_count;

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes, d_srcs, d_degrees, num_levels,
        lower_level, upper_level, edge_count_no_dup);

    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes, d_srcs, d_degrees, num_levels,
        lower_level, upper_level, edge_count_no_dup);

    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaFree(d_srcs));

    // prefix sum to get the offsets array
    uint32_t *d_Ap;
    checkCudaErrors(
        cudaMalloc((void **)&d_Ap, (vertex_count + 1) * sizeof(uint32_t)));

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_degrees,
                                  d_Ap, vertex_count + 1);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_degrees,
                                  d_Ap, vertex_count + 1);
    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaFree(d_degrees));

    uint32_t *h_rowPtr =
        (uint32_t *)malloc((vertex_count + 1) * sizeof(uint32_t));
    uint32_t *h_colInd =
        (uint32_t *)malloc(edge_count_no_dup * sizeof(uint32_t));

    checkCudaErrors(cudaMemcpy(h_rowPtr, d_Ap,
                               (vertex_count + 1) * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_colInd, d_Ai,
                               edge_count_no_dup * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    graph->numVertices = vertex_count;
    graph->numEdges = edge_count_no_dup;

    graph->rowPtr = h_rowPtr;
    graph->colInd = h_colInd;
    graph->d_Ap = d_Ap;
    graph->d_Ai = d_Ai;

    return graph;
}

graph_t *preprocess(const graph_t *original_graph,
                    preprocess_t preprocess_style) {
    preprocess_vertex_t *vertices = (preprocess_vertex_t *)malloc(
        original_graph->numVertices * sizeof(preprocess_vertex_t));
    assert_malloc(vertices);

    for (uint32_t v = 0; v < original_graph->numVertices; v++) {
        vertices[v].id = v;
        vertices[v].edges = &original_graph->colInd[original_graph->rowPtr[v]];
        vertices[v].num_edges =
            original_graph->rowPtr[v + 1] - original_graph->rowPtr[v];
    }

    if (preprocess_style != PREPROCESS_CPU) {
        preprocess_vertex_t *d_vertices;
        preprocess_vertex_t *d_vertices_alt;
        preprocess_vertex_t *d_out;

        checkCudaErrors(
            cudaMalloc((void **)&d_vertices, original_graph->numVertices *
                                                 sizeof(preprocess_vertex_t)));
        checkCudaErrors(cudaMalloc((void **)&d_vertices_alt,
                                   original_graph->numVertices *
                                       sizeof(preprocess_vertex_t)));
        checkCudaErrors(cudaMemcpy(d_vertices, vertices,
                                   original_graph->numVertices *
                                       sizeof(preprocess_vertex_t),
                                   cudaMemcpyHostToDevice));

        if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
            d_out = sort_vertices_GPU(d_vertices, d_vertices_alt,
                                      original_graph->numVertices, true);
        else
            d_out = sort_vertices_GPU(d_vertices, d_vertices_alt,
                                      original_graph->numVertices, false);

        checkCudaErrors(cudaMemcpy(vertices, d_out,
                                   original_graph->numVertices *
                                       sizeof(preprocess_vertex_t),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_vertices));
        checkCudaErrors(cudaFree(d_vertices_alt));
    } else {
        qsort(vertices, original_graph->numVertices,
              sizeof(preprocess_vertex_t), compare_vertex_degree_ascending);
    }

    uint32_t *reverse =
        (uint32_t *)malloc(original_graph->numVertices * sizeof(uint32_t));
    assert_malloc(reverse);

    for (uint32_t v = 0; v < original_graph->numVertices; v++) {
        reverse[vertices[v].id] = v;
    }

    graph_t *graph = (graph_t *)malloc(sizeof(graph_t));
    assert_malloc(graph);

    graph->numVertices = original_graph->numVertices;
    graph->numEdges = original_graph->numEdges / 2;

    graph->rowPtr =
        (uint32_t *)malloc((graph->numVertices + 1) * sizeof(uint32_t));
    assert_malloc(graph->rowPtr);
    graph->colInd = (uint32_t *)malloc(graph->numEdges * sizeof(uint32_t));
    assert_malloc(graph->colInd);

    uint32_t edge_count = 0;

    graph->rowPtr[0] = 0;

    for (uint32_t v = 0; v < original_graph->numVertices; v++) {
        uint32_t new_degree = 0;

        for (int32_t j = 0; j < vertices[v].num_edges; j++) {
            uint32_t w = vertices[v].edges[j];
            uint32_t w_new = reverse[w];

            if (w_new > v) {
                graph->colInd[edge_count++] = w_new;
                new_degree++;
            }
        }

        graph->rowPtr[v + 1] = graph->rowPtr[v] + new_degree;

        if (preprocess_style == PREPROCESS_CPU) {
            qsort(&graph->colInd[graph->rowPtr[v]], new_degree,
                  sizeof(uint32_t), compareInt_t);
        }
    }

    free(vertices);
    free(reverse);

    if (preprocess_style != PREPROCESS_CPU) {
        uint32_t *d_rowPtr;
        uint32_t *d_colInd;
        uint32_t *d_colInd_alt;
        uint32_t *d_colInd_out;

        checkCudaErrors(cudaMalloc(
            (void **)&d_rowPtr, (graph->numVertices + 1) * sizeof(uint32_t)));
        checkCudaErrors(
            cudaMalloc((void **)&d_colInd, graph->numEdges * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc((void **)&d_colInd_alt,
                                   graph->numEdges * sizeof(uint32_t)));
        checkCudaErrors(cudaMemcpy(d_rowPtr, graph->rowPtr,
                                   (graph->numVertices + 1) * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_colInd, graph->colInd,
                                   graph->numEdges * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));

        if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
            d_colInd_out =
                sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt,
                                graph->numVertices, graph->numEdges, true);
        else
            d_colInd_out =
                sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt,
                                graph->numVertices, graph->numEdges, false);

        checkCudaErrors(cudaMemcpy(graph->colInd, d_colInd_out,
                                   graph->numEdges * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_rowPtr));
        checkCudaErrors(cudaFree(d_colInd));
        checkCudaErrors(cudaFree(d_colInd_alt));
    }

    return graph;
}

void free_graph(graph_t *graph) {
    free(graph->rowPtr);
    free(graph->colInd);
    free(graph);
}

void print_degrees(graph_t *graph, const char *filename, uint32_t num,
                   bool oneify) {
    FILE *outfile = fopen(filename, "w");

    printf("n=%u step=%u\n", graph->numVertices,
           max2(1, (graph->numVertices / num)));

    for (uint32_t v = 0; v < graph->numVertices;
         v += max2(1, (graph->numVertices / num))) {
        if (oneify) {
            uint32_t degree = 0;
            for (uint32_t i = graph->rowPtr[v]; i < graph->rowPtr[v + 1]; i++) {
                if (graph->colInd[i] > v)
                    degree++;
            }
            fprintf(outfile, "%u %u\n", v, degree);
        } else {
            fprintf(outfile, "%u %u\n", v,
                    graph->rowPtr[v + 1] - graph->rowPtr[v]);
        }
    }

    fclose(outfile);
}

int main(int argc, char **argv) {
    char *graph_filename = NULL;
    bool graph_mm = false;
    bool graph_zero_indexed = false;
    uint32_t loop_cnt = 1;

    preprocess_t preprocess_style = PREPROCESS_GPU_CONSTRAINED;

    uint32_t spread = 0;
    uint32_t adjacency_matrix_len = 0;

    while ((argc > 1) && (argv[1][0] == '-')) {
        switch (argv[1][1]) {
        case 'm':
            graph_mm = true;
        case 'e':
            if (argc < 3)
                usage();
            graph_filename = argv[2];
            if (graph_filename == NULL)
                usage();
            argv += 2;
            argc -= 2;
            break;
        case 'z':
            graph_zero_indexed = true;
            argv++;
            argc--;
            break;
        case 'a':
            /* Adjacency matrix lengths:
             *	 8192 ~= 4MiB
             *	 16384 ~= 16MiB
             *	 32768 ~= 64MiB
             *	 65536 ~= 265MiB
             *	 131072 ~= 1024MiB
             */
            if (argc < 3)
                usage();
            adjacency_matrix_len = atoi(argv[2]);
            if (adjacency_matrix_len % 32 != 0)
                usage();
            argv += 2;
            argc -= 2;
            break;
        case 's':
            if (argc < 3)
                usage();
            spread = atoi(argv[2]);
            if (spread <= 0)
                usage();
            argv += 2;
            argc -= 2;
            break;
        case 'l':
            if (argc < 3)
                usage();
            loop_cnt = atoi(argv[2]);
            argv += 2;
            argc -= 2;
            break;
        case 'p':
            if (argc < 3)
                usage();
            if (atoi(argv[2]) < PREPROCESS_CPU ||
                atoi(argv[2]) > PREPROCESS_GPU_CONSTRAINED)
                usage();
            preprocess_style = (preprocess_t)atoi(argv[2]);
            argv += 2;
            argc -= 2;
            break;
        default:
            usage();
        }
    }

    if (graph_filename == NULL || spread == 0)
        usage();

    graph_t *original_graph = read_graph2(graph_filename, graph_mm,
                                          graph_zero_indexed, preprocess_style);
    double t_preprocessing = get_seconds();

    /*
    already we did the processing in read_graph2, eliminate this entirely
    */

    // graph_t *graph = preprocess(original_graph, preprocess_style);
    t_preprocessing = get_seconds() - t_preprocessing;
    // free_graph(original_graph);

    printf("%-60s %16s %16s %16s %16s %16s %16s %16s %16s %16s "
           "%16s\n",
           "graph", "n", "m", "s", "a", "triangles", "prepro (s)",
           "GPU copy (s)", "GPU exec (s)", "GPU total (s)", "CPU+GPU (s)");

    bool warmed_up = false;

    for (uint32_t i = 0; i < (loop_cnt + 1); i++) {
        double t_cpu = get_seconds();
        GPU_time t_gpu = {.copy = 0.0, .exec = 0.0};

        uint64_t triangles =
            tc_GPU2(original_graph, spread, adjacency_matrix_len, &t_gpu);

        t_cpu = get_seconds() - t_cpu;

        t_gpu.copy /= (double)1000;
        t_gpu.exec /= (double)1000;

        if (warmed_up) {
            printf("%-60s %16d %16d %16d %16d %16lu "
                   "%16.6f %16.6f "
                   "%16.6f %16.6f %16.6f\n",
                   graph_filename, original_graph->numVertices,
                   original_graph->numEdges, spread, adjacency_matrix_len,
                   triangles, t_preprocessing, t_gpu.copy, t_gpu.exec,
                   t_gpu.copy + t_gpu.exec, t_cpu);
        } else {
            warmed_up = true;
        }
    }

    free_graph(original_graph);

    return EXIT_SUCCESS;
}