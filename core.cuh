#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_select.cuh>

#define CHECK_BOUNDS 1
#define RESET_DEVICE 0
#define BINSEARCH_CONSTANT 0

#define BINSEARCH_CONSTANT_LEVELS 12
#define BINSEARCH_CONSTANT_CACHE_SIZE ((1 << BINSEARCH_CONSTANT_LEVELS) - 1)

#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min2(a, b) ((a) < (b) ? (a) : (b))

static struct timeval tp;
static struct timezone tzp;

#define get_seconds()                                                          \
    (gettimeofday(&tp, &tzp),                                                  \
     (double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)

#define checkCudaErrors(call)                                                  \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#if BINSEARCH_CONSTANT
__constant__ uint64_t c_binary_search_cache[BINSEARCH_CONSTANT_CACHE_SIZE];
#endif