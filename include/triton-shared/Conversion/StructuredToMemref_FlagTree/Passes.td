#ifndef STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemrefFlagTree : Pass<"structured-to-memref-flagtree", "mlir::ModuleOp"> {
  let summary = "Convert Copy to DMA";
}

#endif
