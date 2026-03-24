#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

std::vector<int64_t> get_cuda_mem_info();
std::vector<int64_t> probe_cuda_budget(int64_t target_mb, int64_t reserve_mb, int64_t step_mb, int64_t min_mb);
std::vector<int64_t> probe_llama_cuda_split(
    int64_t target_mb,
    int64_t reserve_mb,
    int64_t step_mb,
    int64_t min_mb,
    int64_t total_layers
);

std::vector<int64_t> configure_runtime(int64_t intra_threads, int64_t interop_threads) {
    py::gil_scoped_release no_gil;

    if (intra_threads > 0) {
        at::set_num_threads(static_cast<int>(intra_threads));
    }

    if (interop_threads > 0) {
        try {
            at::set_num_interop_threads(static_cast<int>(interop_threads));
        } catch (...) {
        }
    }

    return {
        static_cast<int64_t>(at::get_num_threads()),
        static_cast<int64_t>(at::get_num_interop_threads()),
    };
}

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
