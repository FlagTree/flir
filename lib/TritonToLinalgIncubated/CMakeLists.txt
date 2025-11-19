add_triton_library(TritonToLinalgIncubated
	TritonToLinalgIncubatedPass.cpp
        LoadStoreConverter.cpp
	        FunctionConverter.cpp
	ArgMinMaxConverter.cpp
	DescriptorConverter.cpp
	TritonOpConverter.cpp
        BlockPtrAnalysis.cpp
        MaskAnalysis.cpp
        UseAnalysis.cpp

  DEPENDS
  TritonToLinalgIncubatedPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
)

target_compile_definitions(TritonToLinalgIncubated PRIVATE 
  MLIR_HAS_VECTOR_TRANSFORMS_OPTIONS=0
  MLIR_HAS_GPU_AMD_RUNTIME=0
  MLIR_HAS_EMITC_LANGUAGE_TARGET=0
  MLIR_HAS_SPIRV_CLIENT_API=0)

target_compile_options(TritonToLinalgIncubated PRIVATE
  "-include${CMAKE_CURRENT_SOURCE_DIR}/../../include/triton-shared/mlir_compat.h"
)

target_include_directories(TritonToLinalgIncubated
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/triton-shared
    ${CMAKE_CURRENT_BINARY_DIR}/../../../../python/build/cmake.linux-aarch64-cpython-3.11/third_party/flir/include/triton-shared/TritonToLinalgIncubated
    )
