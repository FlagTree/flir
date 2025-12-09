add_triton_library(TritonLinearize
  TritonLinearize.cpp
  TritonLinearizePass.cpp
  PtrAnalysis.cpp
  #  MaskAnalysis.cpp
  # OpFoldResultUtils.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonLinearizePassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
  TritonTransforms
  TritonStructuredIR
  TritonAnalysis
)
target_include_directories(TritonLinearize
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/triton-shared
    ${PROJECT_SOURCE_DIR}/third_party/flir/lib/Analysis
    )

