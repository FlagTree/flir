add_triton_library(MemrefCopyToDMAFlagTree
  MemrefCopyToDMAFlagTree.cpp
  MemrefCopyToDMAFlagTreePass.cpp

  DEPENDS
  MemrefCopyToDMAFlagTreeConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
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
  TritonTilingExtIR
  TritonStructuredIR
)
