add_triton_library(SharedMemoryFlagTree
  SharedMemoryFlagTree.cpp
  SharedMemoryFlagTreePass.cpp

  DEPENDS
  SharedMemoryFlagTreeConversionPassIncGen
  Registrar
  Common

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
