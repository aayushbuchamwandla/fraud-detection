// PyTorch C++/CUDA extension binding: exposes launch_fraud_mlp_forward
// (from fraud_kernel.cu) as a Python-callable torch op, so the custom
// kernel plugs into the same predict()/benchmark harness used for the
// CPU and PyTorch-CUDA backends.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "fraud_kernel.cuh"

torch::Tensor fraud_mlp_forward(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor w3,
    torch::Tensor b3
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == FRAUD_INPUT_DIM,
                "x must be [batch_size, ", FRAUD_INPUT_DIM, "]");
    TORCH_CHECK(w1.is_cuda() && w2.is_cuda() && w3.is_cuda(), "weights must be CUDA tensors");

    x = x.contiguous();
    w1 = w1.contiguous();
    b1 = b1.contiguous();
    w2 = w2.contiguous();
    b2 = b2.contiguous();
    w3 = w3.contiguous();
    b3 = b3.contiguous();

    const int batch_size = x.size(0);
    auto out = torch::empty({batch_size}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_fraud_mlp_forward(
        x.data_ptr<float>(),
        w1.data_ptr<float>(),
        b1.data_ptr<float>(),
        w2.data_ptr<float>(),
        b2.data_ptr<float>(),
        w3.data_ptr<float>(),
        b3.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fraud_mlp_forward", &fraud_mlp_forward,
          "Fused FraudMLP forward pass (custom CUDA kernel)");
}
