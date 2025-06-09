#ifndef STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemrefFlagtree : Pass<"structured-to-memref-flagtree", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}

#endif
