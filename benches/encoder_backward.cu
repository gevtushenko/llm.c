#include <cooperative_groups/reduce.h>
#include <nvbench/nvbench.cuh>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <cuda/atomic>
#include <thrust/random/uniform_int_distribution.h>

__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        cuda::atomic_ref<float, cuda::thread_scope_device> dwte_ix_ref(*dwte_ix);
        cuda::atomic_ref<float, cuda::thread_scope_device> dwte_tc_ref(*dwpe_tc);

        // TODO accumulate in shared memory first, then do a single atomicAdd
        dwte_ix_ref.fetch_add(*dout_btc, cuda::memory_order_relaxed);
        dwte_tc_ref.fetch_add(*dout_btc, cuda::memory_order_relaxed);
    }
}


void base(nvbench::state &state)
{
  int B = 8;              // batch size
  int T = 1024;           // sequence length
  int V = 50257;          // vocab size
  int C = 768;

  thrust::host_vector<int> h_inp(B * T);
  thrust::host_vector<float> h_wte(V * C);
  thrust::host_vector<float> h_wpe(T * C);
  thrust::host_vector<float> h_out(B * T * C);

  thrust::default_random_engine gen(42);
  {
    thrust::uniform_int_distribution<int> dis(0, V);
    thrust::generate(h_inp.begin(), h_inp.end(), [&] { return dis(gen); });
  }

  {
    thrust::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    thrust::generate(h_wte.begin(), h_wte.end(), [&] { return dis(gen); });
    thrust::generate(h_wpe.begin(), h_wpe.end(), [&] { return dis(gen); });
    thrust::generate(h_out.begin(), h_out.end(), [&] { return dis(gen); });
  }

  thrust::device_vector<int> d_inp(h_inp);
  thrust::device_vector<float> d_wte(h_wte);
  thrust::device_vector<float> d_wpe(h_wpe);
  thrust::device_vector<float> d_out(h_out);

  const int N = B * T * C;
  const int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;

  state.add_global_memory_reads<float>(d_wte.size() + d_wpe.size() + d_inp.size() + d_out.size());
  state.add_global_memory_writes<float>(d_wte.size() + d_wpe.size());

  state.exec([&](nvbench::launch &launch) {
    encoder_backward_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(d_wte.data()),
      thrust::raw_pointer_cast(d_wpe.data()),
      thrust::raw_pointer_cast(d_out.data()),
      thrust::raw_pointer_cast(d_inp.data()),
      B, T, C);
  });
}

NVBENCH_BENCH(base);
