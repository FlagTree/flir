#ifndef SHARED_MEMORY_FLAGTREE_CONVERSION_PASSES
#define SHARED_MEMORY_FLAGTREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def SharedMemoryFlagTree : Pass<"shared-memory-flagtree", "mlir::ModuleOp"> {
  let summary = "Add Shared Memory tag";
}

#endif
