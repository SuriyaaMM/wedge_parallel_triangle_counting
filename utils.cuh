#pragma once
#include "core.cuh"

static void assert_malloc(const void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "ERROR: failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }
}

void build_binary_search_cache(uint64_t *src, uint64_t *cache, uint32_t level,
                               uint32_t max_level, uint32_t i, uint32_t s,
                               uint32_t e) {
    if (level < max_level) {
        uint32_t mid = (s + e) / 2;
        cache[i] = src[mid];
        build_binary_search_cache(src, cache, level + 1, max_level, i * 2 + 1,
                                  s, mid);
        build_binary_search_cache(src, cache, level + 1, max_level, i * 2 + 2,
                                  mid + 1, e);
    }
}