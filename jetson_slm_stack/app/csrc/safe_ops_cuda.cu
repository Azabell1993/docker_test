#include <cuda_runtime_api.h>
#include <vector>

// CUDA 런타임 API를 직접 호출해 현재 장비에서 실제로 사용할 수 있는
// GPU 메모리 범위를 보수적으로 추정하기 위한 보조 구현이다.
// Python 레벨에서 단순히 free memory만 읽는 것보다,
// 실제 cudaMalloc 성공 여부를 기준으로 판단하는 데 목적이 있다.

namespace {
// 지정한 크기(MB)의 GPU 메모리를 실제로 할당해 볼 수 있는지 테스트한다.
// 성공하면 즉시 해제하며, "이 크기까지는 현재 시점에서 안전하게 할당 가능"하다는
// 신호로 사용한다.
bool try_alloc_mb(int64_t size_mb) {
    // 0 이하 크기는 의미가 없으므로 바로 실패 처리한다.
    if (size_mb <= 0) {
        return false;
    }

    void* ptr = nullptr;

    // MB 단위를 CUDA 런타임이 요구하는 byte 단위로 변환한다.
    size_t bytes = static_cast<size_t>(size_mb) * 1024ULL * 1024ULL;

    // 현재 CUDA 컨텍스트에서 실제 GPU 메모리 할당을 시도한다.
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        // 실패한 경우 CUDA 내부 에러 상태를 정리해 다음 호출에 영향이 남지 않게 한다.
        cudaGetLastError();
        return false;
    }

    // 탐색 목적의 임시 할당이므로 성공 즉시 해제한다.
    cudaFree(ptr);

    // 해제까지 완료된 시점을 보장하기 위해 동기화한다.
    // 탐색 과정에서 allocator 상태를 가능한 한 안정적으로 유지하려는 목적이다.
    cudaDeviceSynchronize();
    return true;
}
}

// 현재 GPU 메모리의 free/total 값을 byte 단위로 조회한다.
// 반환 형식은 {free_bytes, total_bytes} 이다.
std::vector<int64_t> get_cuda_mem_info() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    // CUDA 런타임이 보고하는 현재 가용 메모리와 전체 메모리를 읽는다.
    auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        // 조회 실패 시 호출 측이 안전하게 실패를 감지할 수 있도록 0 값을 반환한다.
        return {0, 0};
    }

    // Python 바인딩에서 다루기 쉽도록 int64_t 벡터로 변환해 반환한다.
    return {
        static_cast<int64_t>(free_bytes),
        static_cast<int64_t>(total_bytes),
    };
}

// 현재 환경에서 "안전하게 시도할 수 있는 CUDA 메모리 예산(MB)"을 보수적으로 탐색한다.
// 반환 형식은 {free_mb, total_mb, safe_budget_mb} 이다.
std::vector<int64_t> probe_cuda_budget(int64_t target_mb, int64_t reserve_mb, int64_t step_mb, int64_t min_mb) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    // 먼저 현재 시점의 free/total 메모리를 확인한다.
    auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        // 메모리 조회 자체가 실패하면 안전 예산도 계산할 수 없으므로 모두 0 반환.
        return {0, 0, 0};
    }

    // 탐색 계산에 사용하기 위해 byte 단위를 MB 단위로 변환한다.
    int64_t free_mb = static_cast<int64_t>(free_bytes / (1024ULL * 1024ULL));
    int64_t total_mb = static_cast<int64_t>(total_bytes / (1024ULL * 1024ULL));

    // 사용자가 원하는 목표 예산. 이후 reserve 조건에 맞게 상한을 조정한다.
    int64_t capped_target_mb = target_mb;
    if (reserve_mb > 0) {
        // reserve_mb는 CUDA 컨텍스트, KV cache, 런타임 여유분으로 남겨둘 메모리다.
        // 따라서 실제 탐색은 free_mb - reserve_mb 범위를 넘지 않도록 제한한다.
        capped_target_mb = std::min(capped_target_mb, std::max<int64_t>(0, free_mb - reserve_mb));
    }

    // step이 비정상 값이면 64MB 단위로 감소 탐색하도록 보정한다.
    if (step_mb <= 0) {
        step_mb = 64;
    }

    // min이 비정상 값이면 너무 작은 예산까지 내려가지 않도록 256MB를 하한으로 둔다.
    if (min_mb <= 0) {
        min_mb = 256;
    }

    // 아주 작은 할당으로 CUDA 컨텍스트와 allocator를 먼저 깨운다.
    // 첫 호출 오버헤드나 lazy initialization 영향을 줄여 실제 탐색 결과를 더 안정화한다.
    try_alloc_mb(1);

    // 큰 예산부터 시작해 step_mb 단위로 줄여가며 실제 할당 가능한 최대 크기를 찾는다.
    // 단순 free memory 수치가 아니라 cudaMalloc 성공 여부를 기준으로 하므로,
    // fragmentation이나 allocator 상태를 일부 반영할 수 있다.
    for (int64_t probe_mb = capped_target_mb; probe_mb >= min_mb; probe_mb -= step_mb) {
        if (try_alloc_mb(probe_mb)) {
            // 첫 성공 지점이 현재 시점에서의 안전 예산으로 간주된다.
            return {free_mb, total_mb, probe_mb};
        }
    }

    // 어떤 후보도 성공하지 못하면 safe budget은 0으로 반환한다.
    return {free_mb, total_mb, 0};
}

// Llama 계열 모델을 GPU/CPU 하이브리드로 적재할 때,
// 현재 안전 예산에서 GPU에 올릴 수 있는 레이어 수를 매우 보수적으로 계산한다.
// 반환 형식은 {free_mb, total_mb, safe_budget_mb, gpu_layers} 이다.
std::vector<int64_t> probe_llama_cuda_split(
    int64_t target_mb,
    int64_t reserve_mb,
    int64_t step_mb,
    int64_t min_mb,
    int64_t total_layers
) {
    // 먼저 현재 장비에서 안전하게 확보 가능한 CUDA 예산을 계산한다.
    auto budget = probe_cuda_budget(target_mb, reserve_mb, step_mb, min_mb);
    if (budget.size() < 3) {
        return {0, 0, 0, 0};
    }

    int64_t free_mb = budget[0];
    int64_t total_mb = budget[1];
    int64_t safe_budget_mb = budget[2];

    // 기본값은 GPU에 올릴 수 있는 레이어가 0개라는 의미다.
    int64_t gpu_layers = 0;

    if (safe_budget_mb > 0 && total_layers > 0) {
        // Jetson allocator는 fragmentation에 민감하므로,
        // 실제 모델 레이어 적재에 앞서 런타임 여유 메모리를 크게 남긴다.
        constexpr int64_t kRuntimeHeadroomMb = 256;

        // 레이어당 메모리 예산도 넉넉하게 잡아,
        // "조금이라도 불안하면 덜 올린다"는 방향으로 계산한다.
        constexpr int64_t kPerLayerBudgetMb = 320;

        // 런타임 헤드룸을 제외하고 실제 레이어 적재에 쓸 수 있는 예산 계산.
        int64_t usable_mb = std::max<int64_t>(0, safe_budget_mb - kRuntimeHeadroomMb);

        // 남은 예산을 레이어당 예산으로 나눠 GPU 적재 가능 레이어 수를 추정한다.
        gpu_layers = usable_mb / kPerLayerBudgetMb;

        // 예산이 아주 작더라도 최소 1개 레이어는 올릴 수 있는 경우를 허용한다.
        // 완전 CPU fallback 대신 "아주 제한적인 offload" 가능성을 남겨두기 위한 보정이다.
        if (gpu_layers <= 0 && safe_budget_mb >= (kRuntimeHeadroomMb + (kPerLayerBudgetMb / 2))) {
            gpu_layers = 1;
        }

        // 계산 결과가 실제 모델의 전체 레이어 수를 넘지 않도록 상한을 건다.
        if (gpu_layers > total_layers) {
            gpu_layers = total_layers;
        }
    }

    // free/total 메모리, 안전 예산, GPU 적재 가능 레이어 수를 함께 반환한다.
    return {free_mb, total_mb, safe_budget_mb, gpu_layers};
}