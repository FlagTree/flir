add_mlir_dialect_library(MLIRMathExtDialect
  MathExtOps.cpp
  MathExtDialect.cpp

  DEPENDS
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
