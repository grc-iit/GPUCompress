# Synthetic GPU benchmark data generator
add_executable(synthetic_generator
    main.cu
    generator.cu
)

target_link_libraries(synthetic_generator PRIVATE
    gpucompress
    CUDA::cudart
)

set_target_properties(synthetic_generator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
