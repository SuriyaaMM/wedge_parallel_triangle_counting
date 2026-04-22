#pragma once
#include "core.cuh"

__device__ int32_t linear_search_GPU(const uint32_t *list, const uint32_t start,
                                     const uint32_t end,
                                     const uint32_t target) {
    for (uint32_t i = start; i < end; i++) {
        if (list[i] == target) {
            return i;
        } else if (list[i] > target) {
            break;
        }
    }

    return -1;
}

__device__ int32_t binary_search_GPU(const uint32_t *list, const uint32_t start,
                                     const uint32_t end,
                                     const uint32_t target) {
    uint32_t s = start, e = end, mid;
    while (s < e) {
        mid = (s + e) >> 1;
        if (list[mid] == target) return mid;

        if (list[mid] < target)
            s = mid + 1;
        else
            e = mid;
    }
    return -1;
}


__device__ uint32_t binary_search_closest_ULONG_GPU(const uint64_t *list,
                                                    const uint32_t start,
                                                    const uint32_t end,
                                                    const uint64_t target) {
    uint32_t s = start, e = end, mid;
    while (s < e) {
        mid = (s + e) >> 1;

        if (list[mid] < target + 1) {
            s = mid + 1;
        } else {
            e = mid;
        }
    }

    return max2(start, (s > 0) ? s - 1 : 0);
}

#if BINSEARCH_CONSTANT
__device__ uint32_t binary_search_closest_ULONG_constant_GPU(
    const uint64_t *list, const uint32_t start, const uint32_t end,
    const uint64_t target) {
    uint64_t mid;

    uint32_t g_s = start;
    uint32_t g_e = end;
    uint32_t g_mid;

    uint32_t c_index = 0;

#pragma unroll
    for (uint32_t iter = 0; iter < BINSEARCH_CONSTANT_LEVELS; iter++) {
        mid = c_binary_search_cache[c_index];
        g_mid = (g_s + g_e) >> 1;

        c_index *= 2;
        c_index += 1;

        if (mid < target + 1) {
            c_index += 1;
            g_s = g_mid + 1;
        } else {
            g_e = g_mid;
        }
    }

    g_s = max2(start, (g_s > 0) ? g_s - 1 : 0);
    return binary_search_closest_ULONG_GPU(list, g_s, g_e, target);
}
#endif