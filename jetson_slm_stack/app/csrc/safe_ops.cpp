#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

// CUDA 관련 실제 구현은 safe_ops_cuda.cu 에 있고,
// 이 파일은 Python에서 불러 쓸 CPU 런타임 제어 함수와
// CUDA helper 심볼을 pybind11 모듈로 묶는 얇은 진입점 역할을 한다.
std::vector<int64_t> get_cuda_mem_info();
std::vector<int64_t> probe_cuda_budget(int64_t target_mb, int64_t reserve_mb, int64_t step_mb, int64_t min_mb);
std::vector<int64_t> probe_llama_cuda_split(
    int64_t target_mb,
    int64_t reserve_mb,
    int64_t step_mb,
    int64_t min_mb,
    int64_t total_layers
);

// PyTorch CPU 스레드 수를 서버 시작 시 한 번 정리해 두면,
// Python 레벨 torch.set_num_threads 호출보다 예외 처리가 단순하고
// GIL 없이 빠르게 적용할 수 있다.
std::vector<int64_t> configure_runtime(int64_t intra_threads, int64_t interop_threads) {
    // 스레드 설정 자체는 Python 객체를 만지지 않으므로 GIL을 잠시 풀어 둔다.
    py::gil_scoped_release no_gil;

    if (intra_threads > 0) {
        at::set_num_threads(static_cast<int>(intra_threads));
    }

    if (interop_threads > 0) {
        try {
            // 일부 빌드/런타임에서는 inter-op thread 변경이 제한될 수 있으므로
            // 서버 부팅 실패로 이어지지 않게 방어적으로 처리한다.
            at::set_num_interop_threads(static_cast<int>(interop_threads));
        } catch (...) {
        }
    }

    // 실제 적용된 값을 다시 반환해 Python 쪽 health 출력과 로그에 그대로 사용한다.
    return {
        static_cast<int64_t>(at::get_num_threads()),
        static_cast<int64_t>(at::get_num_interop_threads()),
    };
}

// TORCH_EXTENSION_NAME 은 setup.py / torch extension 빌드 시 주입된다.
// server.py 는 이 모듈을 import 하여 CPU 스레드 설정과 CUDA 메모리 프로브를 호출한다.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "safe_ops: lightweight CPU runtime tuning helpers";
    m.def("configure_runtime", &configure_runtime,
          "Configure PyTorch intra-op and inter-op CPU threads");
    m.def("get_cuda_mem_info", &get_cuda_mem_info,
          "Return CUDA free/total memory in bytes");
    m.def("probe_cuda_budget", &probe_cuda_budget,
        "Probe a safe CUDA allocation budget in MB using real cudaMalloc attempts");
    m.def("probe_llama_cuda_split", &probe_llama_cuda_split,
        "Probe a conservative fixed llama CUDA split plan using real cudaMalloc attempts");
}
