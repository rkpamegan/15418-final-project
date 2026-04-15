#pragma once

#ifndef __CUDACC__
    #define __host__
    #define __device__
#endif

__host__ __device__ inline float cu_min(float x, float y) {
    return x < y ? x : y;
}
__host__ __device__ inline float cu_max(float x, float y) {
    return x > y ? x : y;
}

__host__ __device__ inline float cu_abs(float x) {
    return x < 0 ? -x : x;
}

__host__ __device__ inline float cu_sqrt(float x) {
    #ifdef __CUDA_ARCH__
        return sqrtf(x);
    #else
        return std::sqrt(x);
    #endif
}