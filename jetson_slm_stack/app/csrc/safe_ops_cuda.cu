#include <cuda_runtime_api.h>
#include <vector>

namespace {
bool try_alloc_mb(int64_t size_mb) {
    if (size_mb <= 0) {
        return false;
    }

    void* ptr = nullptr;
    size_t bytes = static_cast<size_t>(size_mb) * 1024ULL * 1024ULL;
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaFree(ptr);
    cudaDeviceSynchronize();
    return true;
}
}

std::vector<int64_t> get_cuda_mem_info() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return {0, 0};
    }
    return {
        static_cast<int64_t>(free_bytes),
        static_cast<int64_t>(total_bytes),
    };
}

std::vector<int64_t> probe_cuda_budget(int64_t target_mb, int64_t reserve_mb, int64_t step_mb, int64_t min_mb) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return {0, 0, 0};
    }

    int64_t free_mb = static_cast<int64_t>(free_bytes / (1024ULL * 1024ULL));
    int64_t total_mb = static_cast<int64_t>(total_bytes / (1024ULL * 1024ULL));
    int64_t capped_target_mb = target_mb;
    if (reserve_mb > 0) {
        capped_target_mb = std::min(capped_target_mb, std::max<int64_t>(0, free_mb - reserve_mb));
    }

    if (step_mb <= 0) {
        step_mb = 64;
    }
    if (min_mb <= 0) {
        min_mb = 256;
    }

    // Warm up CUDA allocator/context with a tiny allocation.
    try_alloc_mb(1);

    for (int64_t probe_mb = capped_target_mb; probe_mb >= min_mb; probe_mb -= step_mb) {
        if (try_alloc_mb(probe_mb)) {
            return {free_mb, total_mb, probe_mb};
        }
    }

    return {free_mb, total_mb, 0};
}

std::vector<int64_t> probe_llama_cuda_split(
    int64_t target_mb,
    int64_t reserve_mb,
    int64_t step_mb,
    int64_t min_mb,
    int64_t total_layers
) {
    auto budget = probe_cuda_budget(target_mb, reserve_mb, step_mb, min_mb);
    if (budget.size() < 3) {
        return {0, 0, 0, 0};
    }

    int64_t free_mb = budget[0];
    int64_t total_mb = budget[1];
    int64_t safe_budget_mb = budget[2];
    int64_t gpu_layers = 0;

    if (safe_budget_mb > 0 && total_layers > 0) {
        // Jetson allocator가 작은 파편에도 민감하므로 레이어 수를 강하게 보수적으로 잡는다.
        constexpr int64_t kRuntimeHeadroomMb = 256;
        constexpr int64_t kPerLayerBudgetMb = 320;

        int64_t usable_mb = std::max<int64_t>(0, safe_budget_mb - kRuntimeHeadroomMb);
        gpu_layers = usable_mb / kPerLayerBudgetMb;

        if (gpu_layers <= 0 && safe_budget_mb >= (kRuntimeHeadroomMb + (kPerLayerBudgetMb / 2))) {
            gpu_layers = 1;
        }

        if (gpu_layers > total_layers) {
            gpu_layers = total_layers;
        }
    }

    return {free_mb, total_mb, safe_budget_mb, gpu_layers};
}