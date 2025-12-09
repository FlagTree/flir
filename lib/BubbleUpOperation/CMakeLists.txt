add_triton_library(BubbleUpOperation
        BubbleUpOperation.cpp

  DEPENDS
  BubbleUpOperationConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonAnalysis
  MLIRSCFTransforms
)
