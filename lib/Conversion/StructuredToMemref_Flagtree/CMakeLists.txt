add_triton_library(StructuredToMemrefFlagtree
  StructuredToMemrefFlagtree.cpp
  StructuredToMemrefFlagtreePass.cpp

  DEPENDS
  StructuredToMemrefFlagtreeConversionPassIncGen

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
