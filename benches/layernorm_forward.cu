#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvbench/nvbench.cuh>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>


__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

void kernel2(nvbench::state &state)
{
  int B = 8;
  int T = 1024;
  int C = 768;

  thrust::host_vector<float> h_inp(B * T * C);
  thrust::host_vector<float> h_weight(C);
  thrust::host_vector<float> h_bias(C);

  thrust::default_random_engine gen(42);
  thrust::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  thrust::generate(h_inp.begin(), h_inp.end(), [&] { return dis(gen); });
  thrust::generate(h_weight.begin(), h_weight.end(), [&] { return dis(gen); });
  thrust::generate(h_bias.begin(), h_bias.end(), [&] { return dis(gen); });

  thrust::device_vector<float> d_out(B * T * C);
  thrust::device_vector<float> d_mean(B * T);
  thrust::device_vector<float> d_rstd(B * T);
  thrust::device_vector<float> d_inp(h_inp);
  thrust::device_vector<float> d_weight(h_weight);
  thrust::device_vector<float> d_bias(h_bias);

  const int N = B * T;
  const int block_size = state.get_int64("block_size");

  state.add_global_memory_reads<float>(d_inp.size() + d_weight.size() + d_bias.size());
  state.add_global_memory_writes<float>(d_out.size() + d_mean.size() + d_rstd.size());

  const int normalization_block_size = 256;
  const int normalization_grid_size =
      (B * T * C + normalization_block_size - 1) / normalization_block_size;

  state.exec([&](nvbench::launch &launch) {
    cudaStream_t stream = launch.get_stream();
    mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(
      thrust::raw_pointer_cast(d_mean.data()), 
      thrust::raw_pointer_cast(d_inp.data()), 
      N, C, block_size);
    rstd_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(
      thrust::raw_pointer_cast(d_rstd.data()), 
      thrust::raw_pointer_cast(d_inp.data()), 
      thrust::raw_pointer_cast(d_mean.data()), 
      N, C, block_size);
    normalization_kernel<<<normalization_grid_size, normalization_block_size>>>(
      thrust::raw_pointer_cast(d_out.data()), 
      thrust::raw_pointer_cast(d_inp.data()), 
      thrust::raw_pointer_cast(d_mean.data()), 
      thrust::raw_pointer_cast(d_rstd.data()), 
      thrust::raw_pointer_cast(d_weight.data()), 
      thrust::raw_pointer_cast(d_bias.data()), 
      B, T, C);
  });
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}
NVBENCH_BENCH(kernel2).add_int64_axis("block_size", {32, 64, 128, 256, 512, 1024});

void kernel3(nvbench::state &state)
{
  int B = 8;
  int T = 1024;
  int C = 768;

  thrust::host_vector<float> h_inp(B * T * C);
  thrust::host_vector<float> h_weight(C);
  thrust::host_vector<float> h_bias(C);

  thrust::default_random_engine gen(42);
  thrust::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  thrust::generate(h_inp.begin(), h_inp.end(), [&] { return dis(gen); });
  thrust::generate(h_weight.begin(), h_weight.end(), [&] { return dis(gen); });
  thrust::generate(h_bias.begin(), h_bias.end(), [&] { return dis(gen); });

  thrust::device_vector<float> d_out(B * T * C);
  thrust::device_vector<float> d_mean(B * T);
  thrust::device_vector<float> d_rstd(B * T);
  thrust::device_vector<float> d_inp(h_inp);
  thrust::device_vector<float> d_weight(h_weight);
  thrust::device_vector<float> d_bias(h_bias);

  const int N = B * T;
  const int block_size = state.get_int64("block_size");
  const int grid_size = (N * 32 + block_size - 1) / block_size;

  state.add_global_memory_reads<float>(d_inp.size() + d_weight.size() + d_bias.size());
  state.add_global_memory_writes<float>(d_out.size() + d_mean.size() + d_rstd.size());

  state.exec([&](nvbench::launch &launch) {
    cudaStream_t stream = launch.get_stream();
    layernorm_forward_kernel3<<<grid_size, block_size, 0, stream>>>(
      thrust::raw_pointer_cast(d_out.data()), 
      thrust::raw_pointer_cast(d_mean.data()), 
      thrust::raw_pointer_cast(d_rstd.data()), 
      thrust::raw_pointer_cast(d_inp.data()), 
      thrust::raw_pointer_cast(d_weight.data()),
      thrust::raw_pointer_cast(d_bias.data()),
      N, C);
  });
}
NVBENCH_BENCH(kernel3).add_int64_axis("block_size", {32, 64, 128, 256, 512, 1024});
