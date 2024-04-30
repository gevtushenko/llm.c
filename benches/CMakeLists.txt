
add_executable(bench_layernorm layernorm_forward.cu)
target_link_libraries(bench_layernorm CCCL::CCCL CUDA::cublas CUDA::cublasLt nvbench::main)
target_compile_options(bench_layernorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)

add_executable(encoder_backward encoder_backward.cu)
target_link_libraries(encoder_backward CCCL::CCCL CUDA::cublas CUDA::cublasLt nvbench::main)
target_compile_options(encoder_backward PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)