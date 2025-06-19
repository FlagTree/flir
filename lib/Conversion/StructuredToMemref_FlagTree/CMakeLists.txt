add_triton_library(StructuredToMemrefFlagTree
  StructuredToMemrefFlagTree.cpp
  StructuredToMemrefFlagTreePass.cpp

  DEPENDS
  StructuredToMemrefFlagTreeConversionPassIncGen

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
