#pragma once
#include <stdint.h>

enum preprocess_t {
    PREPROCESS_CPU = 0,
    PREPROCESS_GPU,
    PREPROCESS_GPU_CONSTRAINED
};

typedef struct {
    uint32_t numVertices;
    uint32_t numEdges;
    uint32_t *rowPtr;
    uint32_t *colInd;
    uint32_t *d_Ap;
    uint32_t *d_Ai;
} graph_t;

typedef struct edge_t {
    uint32_t src;
    uint32_t dst;
    __host__ __device__ __forceinline__ bool
    operator==(const edge_t &other) const {
        return (src == other.src) && (dst == other.dst);
    }
} edge_t;

typedef struct {
    uint32_t id;
    uint32_t *edges;
    uint32_t num_edges;
} preprocess_vertex_t;

typedef struct {
    double copy;
    double exec;
} GPU_time;
