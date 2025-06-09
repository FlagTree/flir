#ifndef TRITON_STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES_H
#define TRITON_STRUCTURED_TO_MEMREF_FLAGTREE_CONVERSION_PASSES_H

#include "triton-shared/Conversion/StructuredToMemref_Flagtree/StructuredToMemrefFlagtree.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/StructuredToMemref_Flagtree/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
