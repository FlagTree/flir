add_triton_library(MLIRTritonNPUUtils
        Utils.cpp
        InterleaveOptimization.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
target_include_directories(MLIRTritonNPUUtils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/triton-shared
    )
